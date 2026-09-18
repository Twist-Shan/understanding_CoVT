"""Colab-only balanced size/depth pilot and intermediate-residual interventions."""
import gc
import json
import os
from pathlib import Path
import time
import zipfile

if not Path('/content').is_dir() or 'COLAB_RELEASE_TAG' not in os.environ:
    raise RuntimeError('Run experiments on Colab only.')

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from covt_pilot.io import read_json, read_jsonl, write_json, write_jsonl, sha256
from covt_pilot.shift_scenes import render
from covt_pilot.scenes import annulus
from covt_pilot.model import CoVTModel, reset_rope, language_norm, extract_answer
from covt_pilot.decoder import NativeDepthDecoder, map_to_input
from covt_pilot.expert import DepthExpert
from covt_pilot.metrics import raw_difference, score_difference, cluster_interval

root=Path('/content/covt-pilot')
out=root/'runs'/time.strftime('balanced-intervention-%Y%m%d-%H%M%S',time.gmtime())
out.mkdir()
write_json(root/'runs/latest-balanced-intervention.json',{'run_dir':str(out)})
prior=Path(read_json(root/'runs/latest-colab.json')['run_dir'])
source=Path(read_json(root/'runs/latest-followup.json')['run_dir'])
audit=read_json(prior/'resources/report.json')
cal=read_json(source/'e0/calibration.json')
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
rng=np.random.default_rng(1902)
rows=[]
try:
    font=ImageFont.truetype('DejaVuSans-Bold.ttf',24)
except OSError:
    font=ImageFont.load_default(size=24)
scenes=out/'scenes'; scenes.mkdir()

# Eight independent families, depth order x projected-size order fully crossed.
# Angular radii and x/z stay fixed during depth exchange. World radii change;
# this is a scene intervention, not a claim of an isolated physical depth cause.
for family in range(8):
    colors=rng.uniform(.3,.95,(2,3))
    near=float(rng.uniform(4.2,4.6)); far=float(rng.uniform(6.4,6.9))
    small=float(rng.uniform(.070,.078)); large=float(rng.uniform(.110,.118))
    for depth_order in range(2):
        for size_order in range(2):
            z=np.array([near,far])[::(-1 if depth_order else 1)]
            angular=np.array([large,small])[::(-1 if size_order else 1)]
            radius=z*angular
            axes=np.repeat(radius[:,None],3,axis=1)
            centers=np.column_stack([np.array([-.20,.20])*z,1.3-radius,z])
            image,depth,ids,marks=render(centers,axes,['ellipsoid']*2,colors,.9,1.3)
            order=[0,1] if family%2==0 else [1,0]
            points=[marks[i] for i in order]
            for point,obj in zip(points,order):
                x,y=point
                assert 20<x<291 and 20<y<311
                assert np.all(ids[annulus(depth.shape,point)]==obj)
            depths=[float(depth[y,x]) for x,y in points]
            assert abs(depths[0]-depths[1])>.4
            label='A' if depths[0]<depths[1] else 'B'
            areas=[int(np.sum(ids==i)) for i in order]
            size_label='A' if areas[0]>areas[1] else 'B'
            cue='aligned' if size_label==label else 'conflict'
            sample=f'balanced_{family:03d}_d{depth_order}_s{size_order}'
            folder=scenes/sample; folder.mkdir()
            image.save(folder/'unmarked.png')
            draw=ImageDraw.Draw(image)
            for letter,(x,y) in zip('AB',points):
                draw.ellipse((x-3,y-3,x+3,y+3),fill='white',outline='black')
                draw.text((x+12,y-18),letter,font=font,fill='white',stroke_width=2,stroke_fill='black')
            image.save(folder/'image.png')
            np.savez_compressed(folder/'geometry.npz',depth=depth,object_ids=ids)
            rows.append(dict(id=sample,family=f'balanced_{family:03d}',split='pilot',
                depth_order=depth_order,size_order=size_order,cue=cue,label=label,
                size_label=size_label,areas=areas,points=points,point_depths=depths,
                image=f'{sample}/image.png',camera={'size':[336,336]},inner_radius=5,outer_radius=9,
                centers=centers.tolist(),axes=axes.tolist(),colors=colors.tolist(),
                question='Which marked point is closer to the camera, A or B? Answer only A or B.'))
assert sum(r['cue']=='conflict' for r in rows)==16
assert sum(r['label']=='A' for r in rows)==16
write_jsonl(scenes/'manifest.jsonl',rows)
write_json(out/'plan.json',dict(seed=1902,images=32,families=8,calibration=cal,
    conditions=['baseline','identity','opposite_depth','same_depth_other_family','token_shuffle'],
    site='Input residual of middle transformer block at four naturally generated depth-pad positions.',
    rollout='Truncate immediately after last depth pad; recompute full prefix with use_cache=False on every step. No target answer tokens retained.',
    limits='Exploratory single-layer pilot; depth exchange also changes world radii and floor-contact positions. No causal claim about a pure depth concept.'))
canvas=Image.new('RGB',(4*336,2*366),'white')
for i,row in enumerate(rows[:8]):
    canvas.paste(Image.open(scenes/row['image']),((i%4)*336,(i//4)*366+30))
    ImageDraw.Draw(canvas).text(((i%4)*336+5,(i//4)*366+5),f"{row['id']} {row['cue']} GT={row['label']}",fill='black')
canvas.save(out/'contact_sheet.png')
print('BALANCED_GEOMETRY_READY',str(out),flush=True)

cal_rows=[r for r in read_jsonl(source/'e0/results.jsonl') if r['split']=='calibration' and r['e0_checks_pass']]
hs=[]
for row in cal_rows:
    path=source/'e0'/row['artifact']; assert sha256(path)==row['artifact_sha256']
    with np.load(path,allow_pickle=False) as z: hs.append(torch.from_numpy(z['hidden'].copy()))
fixed=torch.stack(hs).mean(0)
decoder=NativeDepthDecoder.restore(prior/'decoder','cuda').float()
teacher_info=read_json(prior/'expert_download.json')
from huggingface_hub import hf_hub_download
checkpoint=hf_hub_download(teacher_info['model_id'],'depth_anything_v2_vitl.pth',revision=teacher_info['revision'])
expert=DepthExpert(root/'third_party/CoVT',checkpoint,audit['code']['revision'],'cuda')
features_dir=out/'features'; features_dir.mkdir()

def score(raw,row,key='decoder'):
    mapped=map_to_input(raw,(336,336)).cpu().numpy()
    result=score_difference(raw_difference(mapped,row),cal[key])
    return dict(decoder_label=result['label'],decoder_margin=result['margin'],
                decoder_correct=float(result['label']==row['label']),abstain=float(result['abstain']))

frozen=[]
with torch.inference_mode():
    for i,row in enumerate(rows):
        features,teacher,patch_hw,native_hw=expert.encode(Image.open(scenes/row['image']).convert('RGB'))
        _,raw=decoder(fixed,features,patch_hw,native_hw)
        path=features_dir/f"{row['id']}.npz"
        np.savez_compressed(path,features=np.stack([f[0].cpu().numpy() for f in features]),patch_hw=patch_hw,native_hw=native_hw)
        frozen.append(dict(row,feature_sha256=sha256(path),fixed=score(raw,row),expert=score(teacher,row,'expert')))
        write_jsonl(out/'frozen.jsonl',frozen)
        print('BALANCED_EXPERT',i+1,len(rows),flush=True)
del expert,features,teacher,raw
gc.collect(); torch.cuda.empty_cache()
model=CoVTModel(audit['model_id'],audit['model_revision'],'cuda','float32','eager')
modules=dict(model.model.named_modules())
layer_names=[n for n in modules if n.startswith('model.layers.') and n.count('.')==2]
assert layer_names, 'Unrecognized transformer layer graph'
layer_index=len(layer_names)//2
site=f'model.layers.{layer_index}'
layer=modules[site]; _,norm=language_norm(model.model)
write_json(out/'intervention_site.json',dict(site=site,layers=len(layer_names),provenance=model.provenance))

def inputs_for(row):
    image=Image.open(scenes/row['image']).convert('RGB')
    prompt=model.processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':row['question']}]}],tokenize=False,add_generation_prompt=True)
    return model.processor(text=[prompt],images=[image],return_tensors='pt').to(model.input_device)

@torch.inference_mode()
def forward(inputs,ids,positions,replacement=None):
    middle=[]; terminal=[]
    def pre(module,args,kwargs):
        h=args[0] if args else kwargs['hidden_states']
        p=positions.to(h.device)
        middle.append(h[:,p].detach().clone())
        if replacement is not None:
            h=h.clone(); h[:,p]=replacement.to(h.device,h.dtype)
            if args: args=(h,)+args[1:]
            else: kwargs=dict(kwargs,hidden_states=h)
        return args,kwargs
    def post(module,args,h): terminal.append(h[:,positions.to(h.device)].detach().clone())
    a=layer.register_forward_pre_hook(pre,with_kwargs=True); b=norm.register_forward_hook(post)
    try:
        reset_rope(model.model)
        result=model.model(**dict(inputs,input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True))
        assert len(middle)==len(terminal)==1
        return result.logits[:,-1].detach(),middle[0],terminal[0]
    finally:
        a.remove(); b.remove()

@torch.inference_mode()
def rollout(inputs,ids,positions,replacement=None):
    ids=ids.clone(); tokens=[]; first_h=None; first_logits=None
    eos=model.model.generation_config.eos_token_id
    eos={eos} if isinstance(eos,int) else set(eos)
    for step in range(128):
        logits,_,hidden=forward(inputs,ids,positions,replacement)
        if step==0: first_h=hidden.cpu(); first_logits=logits.cpu()
        token=int(logits.argmax(-1).item()); tokens.append(token)
        ids=torch.cat([ids,torch.tensor([[token]],device=ids.device)],dim=1)
        if token in eos: break
    raw=model.processor.tokenizer.decode(tokens,skip_special_tokens=False)
    clean=model.processor.tokenizer.decode(tokens,skip_special_tokens=True)
    return dict(answer=extract_answer(clean),text=raw,ids=tokens,complete=tokens[-1] in eos),first_h,first_logits

native=[]; states={}; contexts={}
for i,row in enumerate(rows):
    pred,h=model.run(Image.open(scenes/row['image']).convert('RGB'),row['question'])
    record=dict(row,**pred)
    record['eligible']=bool(h is not None and pred.get('identity_pass') and pred['cached_uncached_tokens_equal'] and pred['generation_complete'])
    if record['eligible']:
        inputs=inputs_for(row)
        all_ids=torch.cat([inputs.input_ids,torch.tensor([pred['generated_ids']],device=model.input_device)],dim=1)
        positions=torch.tensor(pred['depth_positions'],device=model.input_device)
        prefix=all_ids[:,:int(positions[-1])+1]
        # Reject any answer already present before the intervention boundary.
        before=model.processor.tokenizer.decode(prefix[0,inputs.input_ids.shape[1]:],skip_special_tokens=False)
        assert '<answer>' not in before
        _,mid,terminal=forward(inputs,prefix,positions)
        states[row['id']]=mid.cpu(); contexts[row['id']]=(prefix.cpu(),positions.cpu())
        np.save(out/f"{row['id']}_middle.npy",mid.cpu().numpy())
        record['terminal_prefix_matches_full']=bool(torch.allclose(terminal[0].cpu(),h,atol=.02,rtol=.01))
    native.append(record); write_jsonl(out/'native.jsonl',native)
    print('BALANCED_NATIVE',i+1,len(rows),record['eligible'],flush=True)

index={r['id']:r for r in native}; frozen_index={r['id']:r for r in frozen}
interventions=[]
for i,row in enumerate(native):
    if not row['eligible'] or not row['terminal_prefix_matches_full']: continue
    inputs=inputs_for(row); prefix,positions=contexts[row['id']]
    prefix=prefix.to(model.input_device); positions=positions.to(model.input_device)
    base,h,logits=rollout(inputs,prefix,positions)
    same,identity_h,identity_logits=rollout(inputs,prefix,positions,states[row['id']])
    expected=row['generated_ids'][int(positions[-1])+1-row['prefix_length']:]
    identity_pass=bool(base['ids']==same['ids']==expected and torch.equal(h,identity_h) and torch.equal(logits,identity_logits))
    write_json(out/f"{row['id']}_identity.json",dict(identity_pass=identity_pass,baseline=base,identity=same,expected=expected,
        hidden_max_abs=float((h-identity_h).abs().max()),logits_max_abs=float((logits-identity_logits).abs().max())))
    if not identity_pass:
        print('IDENTITY_FAILED',row['id'],flush=True); continue
    opposite=f"{row['family']}_d{1-row['depth_order']}_s{row['size_order']}"
    controls=[r for r in native if r['family']!=row['family'] and r['label']==row['label'] and r['cue']==row['cue'] and r['eligible']]
    cases=[('baseline',None,None),('identity',states[row['id']],row['id']),('token_shuffle',states[row['id']][:,[1,2,3,0]],row['id'])]
    if opposite in states: cases.append(('opposite_depth',states[opposite],opposite))
    if controls: cases.append(('same_depth_other_family',states[controls[0]['id']],controls[0]['id']))
    path=features_dir/f"{row['id']}.npz"; assert sha256(path)==frozen_index[row['id']]['feature_sha256']
    with np.load(path,allow_pickle=False) as z:
        features=[torch.from_numpy(f.copy()).unsqueeze(0).cuda() for f in z['features']]
        patch_hw=z['patch_hw'].tolist(); native_hw=z['native_hw'].tolist()
    for condition,replacement,donor in cases:
        if condition=='baseline': answer,terminal=base,h
        elif condition=='identity': answer,terminal=same,identity_h
        else: answer,terminal,_=rollout(inputs,prefix,positions,replacement)
        with torch.inference_mode(): _,raw=decoder(terminal[0],features,patch_hw,native_hw)
        result=dict(id=row['id'],family=row['family'],cue=row['cue'],label=row['label'],condition=condition,donor=donor,
            **answer,**score(raw,row),answer_correct=float(answer['answer']==row['label']),
            answer_changed=float(answer['answer']!=base['answer']),valid=float(answer['answer'] in ('A','B')),
            terminal_max_abs=float((terminal-h).abs().max()))
        interventions.append(result); write_jsonl(out/'interventions.jsonl',interventions)
    print('INTERVENTION_DONE',i+1,len(native),row['id'],flush=True)

summary={'attempted':len(native),'native_eligible':sum(r['eligible'] for r in native),'site':site,
         'identity_passed':sum(r['condition']=='baseline' for r in interventions),'results':{}}
for cue in ('all','aligned','conflict'):
    summary['results'][cue]={}
    for condition in sorted({r['condition'] for r in interventions}):
        selected=[r for r in interventions if r['condition']==condition and (cue=='all' or r['cue']==cue)]
        summary['results'][cue][condition]=dict(n=len(selected),**{k:cluster_interval(selected,k) for k in ('answer_correct','answer_changed','decoder_correct','valid')})
write_json(out/'summary.json',summary)
archive=Path('/content')/(out.name+'-reports.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in out.rglob('*'):
        if p.is_file() and p.suffix!='.npz': z.write(p,p.relative_to(root))
    z.write(Path(__file__),'scripts/balanced_intervention.py')
meta=dict(path=str(archive),sha256=sha256(archive),bytes=archive.stat().st_size)
write_json(out/'archive.json',meta)
print('BALANCED_INTERVENTION_COMPLETED',json.dumps(summary),flush=True)
print('ARCHIVE',json.dumps(meta),flush=True)
