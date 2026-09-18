"""Colab-only sequential block/all-layer and conditional thought-prefix interventions."""
import json
import os
import re
import time
import zipfile
from pathlib import Path
if not Path('/content').is_dir() or 'COLAB_RELEASE_TAG' not in os.environ:
    raise RuntimeError('Run on Colab only.')
import torch
from PIL import Image
from covt_pilot.io import read_json, read_jsonl, write_json, write_jsonl, sha256
from covt_pilot.model import CoVTModel, reset_rope, extract_answer
from covt_pilot.metrics import cluster_interval

root=Path('/content/covt-pilot')
source=Path(read_json(root/'runs/latest-balanced-intervention.json')['run_dir'])
selected={r['id'] for r in read_json(source/'paired_analysis.json')['matched_donors']['all']['rows']}
rows=[r for r in read_jsonl(source/'native.jsonl') if r['id'] in selected]
assert len(rows)==20
lookup={r['id']:r for r in rows}
donors={}
for r in rows:
    opposite=f"{r['family']}_d{1-r['depth_order']}_s{r['size_order']}"
    candidates=sorted(x['id'] for x in rows if x['family']!=r['family'] and
        all(x[k]==r[k] for k in ('label','depth_order','size_order','cue')))
    assert opposite in lookup and candidates
    same=next((x for x in candidates if x>r['id']),candidates[0])
    donors[r['id']]={'opposite_depth':opposite,'same_depth':same}
    for donor in (opposite,same):
        d=lookup[donor]
        assert r['prompt']==d['prompt'] and r['prefix_length']==d['prefix_length']
        assert r['depth_positions']==d['depth_positions']
        stop=r['depth_positions'][-1]+1-r['prefix_length']
        assert r['generated_ids'][:stop]==d['generated_ids'][:stop], 'Thought-prefix token mismatch'

stages=[('early',list(range(0,9)),'pads'),('middle',list(range(9,18)),'pads'),
        ('late',list(range(18,28)),'pads'),('all_layers',list(range(28)),'pads')]
out=root/'runs'/time.strftime('block-intervention-%Y%m%d-%H%M%S',time.gmtime()); out.mkdir()
write_json(root/'runs/latest-block-intervention.json',{'run_dir':str(out)})
write_json(out/'plan.json',dict(source=str(source),source_sha256=sha256(source/'native.jsonl'),
    targets=list(lookup),donors=donors,stages=stages,
    conditional_extension='If all_layers opposite_depth yields no valid changed answer, add all 28 layers at all generated thought-prefix positions from prompt end through last depth pad.',
    interventions='Simultaneous clamping to each corresponding donor layer input, reapplied during every full-prefix recomputation. No old KV and no donor answer tokens.',
    same_depth_control='Other scene family, same true A/B label, depth_order, size_order and cue; deterministic cyclic ID selection. Family identity remains a control limitation.',
    probability_scope='Teacher-forced original continuation immediately before A/B answer; distinct from free rollout.',
    limitations='Six selected scene families, one seed, exploratory repeated interventions and donor reuse; no equivalence claim. Expanded thought-prefix effect is not depth-specific.'))
audit=read_json(Path(read_json(root/'runs/latest-colab.json')['run_dir'])/'resources/report.json')
torch.manual_seed(0); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
model=CoVTModel(audit['model_id'],audit['model_revision'],'cuda','float32','eager')
modules=dict(model.model.named_modules())
for k in range(28): assert f'model.layers.{k}' in modules
tok=model.processor.tokenizer
ab=[tok.encode(x,add_special_tokens=False) for x in 'AB']; assert all(len(x)==1 for x in ab)
ab=[x[0] for x in ab]
eos=model.model.generation_config.eos_token_id; eos={eos} if isinstance(eos,int) else set(eos)
write_json(out/'provenance.json',dict(model=model.provenance,script_sha256=sha256(Path(__file__))))

def context(r):
    inputs=model.processor(text=[r['prompt']],images=[Image.open(source/'scenes'/r['image']).convert('RGB')],return_tensors='pt').to(model.input_device)
    assert inputs.input_ids.shape[1]==r['prefix_length']
    full=torch.cat([inputs.input_ids,torch.tensor([r['generated_ids']],device=model.input_device)],dim=1)
    hits=[i for i,t in enumerate(r['generated_ids']) if t in ab]
    assert len(hits)==1 and r['generated_ids'][hits[0]]==ab['AB'.index(r['answer'])]
    stop=r['depth_positions'][-1]+1
    positions=torch.arange(r['prefix_length'],stop,device=model.input_device)
    assert r['prefix_length']+hits[0]>stop
    return inputs,full[:,:stop],full[:,:r['prefix_length']+hits[0]],positions

@torch.inference_mode()
def forward(inputs,ids,positions,patch=None,capture=False):
    handles=[]; states={}; touched=[]
    def make_pre(k):
        def pre(module,args,kwargs):
            h=args[0] if args else kwargs['hidden_states']
            if capture: states[k]=h[:,positions].detach().cpu().clone()
            if patch is not None and k in patch:
                h=h.clone(); h[:,positions]=patch[k].to(h.device,h.dtype); touched.append(k)
                if args: args=(h,)+args[1:]
                else: kwargs=dict(kwargs,hidden_states=h)
            return args,kwargs
        return pre
    for k in (range(28) if capture else sorted(patch or {})):
        handles.append(modules[f'model.layers.{k}'].register_forward_pre_hook(make_pre(k),with_kwargs=True))
    try:
        reset_rope(model.model)
        result=model.model(**dict(inputs,input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True))
        assert sorted(touched)==sorted(patch or {}), 'Hook coverage mismatch'
        return result.logits[:,-1].float().cpu(),states
    finally:
        for h in handles: h.remove()

def probability(logits):
    x=logits[0]; pair=x[ab].softmax(0); full=x.softmax(0)
    return dict(logit_A_minus_B=float(x[ab[0]]-x[ab[1]]),conditional_p_A=float(pair[0]),
        p_A=float(full[ab[0]]),p_B=float(full[ab[1]]),argmax=int(x.argmax()))

def parse(text):
    tags=re.findall(r'<answer>\s*([AB])\s*</answer>',text)
    letters=set(re.findall(r'(?<![A-Za-z])[AB](?![A-Za-z])',text))
    strict=len(tags)==1 and letters=={tags[0]}
    return dict(answer=tags[0] if strict else None,format_valid=bool(strict),ambiguous_AB=len(letters)>1,
        unique_letter=next(iter(letters)) if len(letters)==1 else None)

def rollout(inputs,prefix,positions,patch=None):
    ids=prefix.clone(); generated=[]
    for _ in range(128):
        logits,_=forward(inputs,ids,positions,patch)
        token=int(logits.argmax(-1).item()); generated.append(token)
        ids=torch.cat([ids,torch.tensor([[token]],device=ids.device)],dim=1)
        if token in eos: break
    text=tok.decode(generated,skip_special_tokens=False)
    return dict(ids=generated,text=text,**parse(text),legacy_answer=extract_answer(tok.decode(generated,skip_special_tokens=True)),
        complete=generated[-1] in eos)

bank={}; answer_bank={}; original_logits={}; baselines={}; checks=[]
for i,r in enumerate(rows):
    inputs,prefix,answer_prefix,positions=context(r)
    _,bank[r['id']]=forward(inputs,prefix,positions,capture=True)
    logits,answer_bank[r['id']]=forward(inputs,answer_prefix,positions,capture=True)
    cc={str(k):dict(max_abs=float((bank[r['id']][k]-answer_bank[r['id']][k]).abs().max()),
        close=torch.allclose(bank[r['id']][k],answer_bank[r['id']][k],atol=1e-3,rtol=1e-5)) for k in range(28)}
    checks.append(dict(id=r['id'],layers=cc)); write_json(out/'prefix_checks.json',checks)
    assert all(c['close'] for c in cc.values()), 'Cross-length diagnostic failed'
    baseline=rollout(inputs,prefix,positions)
    assert baseline['ids']==r['generated_ids'][prefix.shape[1]-r['prefix_length']:]
    assert baseline['complete'] and baseline['format_valid'] and baseline['answer']==r['answer']
    assert int(logits.argmax(-1))==ab['AB'.index(r['answer'])]
    original_logits[r['id']]=logits; baselines[r['id']]=baseline
    write_json(out/f"{r['id']}_baseline.json",dict(id=r['id'],**baseline,**probability(logits)))
    print('CAPTURED',i+1,20,flush=True)

results=[]; identities=[]; summaries={}
def run_stage(name,layers,scope):
    for i,r in enumerate(rows):
        inputs,prefix,answer_prefix,all_positions=context(r)
        indexes=list(range(len(all_positions))) if scope=='thought_prefix' else [p-r['prefix_length'] for p in r['depth_positions']]
        positions=all_positions[indexes]
        def patches(store,donor): return {k:store[donor][k][:,indexes] for k in layers}
        identity,_=forward(inputs,answer_prefix,positions,patches(answer_bank,r['id']))
        own=rollout(inputs,prefix,positions,patches(bank,r['id']))
        passed=torch.equal(identity,original_logits[r['id']]) and own['ids']==baselines[r['id']]['ids']
        identities.append(dict(id=r['id'],stage=name,pass_identity=bool(passed),logits_max_abs=float((identity-original_logits[r['id']]).abs().max())))
        write_jsonl(out/'identity.jsonl',identities)
        assert passed, 'Identity failed; stop without dropping targets'
        original=probability(original_logits[r['id']])
        for condition,donor in donors[r['id']].items():
            patch=patches(bank,donor); answer_patch=patches(answer_bank,donor)
            logits,_=forward(inputs,answer_prefix,positions,answer_patch)
            metrics=probability(logits); answer=rollout(inputs,prefix,positions,patch)
            distances={str(k):dict(max_abs=float((patch[k]-bank[r['id']][k][:,indexes]).abs().max()),
                relative_l2=float((patch[k]-bank[r['id']][k][:,indexes]).norm()/bank[r['id']][k][:,indexes].norm().clamp_min(1e-12))) for k in layers}
            delta=metrics['conditional_p_A']-original['conditional_p_A']
            result=dict(id=r['id'],family=r['family'],cue=r['cue'],label=r['label'],baseline_answer=r['answer'],
                stage=name,layers=layers,scope=scope,positions=positions.cpu().tolist(),condition=condition,donor=donor,
                **answer,**metrics,delta_p_A=delta,abs_delta_p_A=abs(delta),
                delta_toward_opposite_label=delta*(1 if r['label']=='B' else -1),
                delta_logit_A_minus_B=metrics['logit_A_minus_B']-original['logit_A_minus_B'],
                valid_flip=float(answer['answer'] is not None and answer['answer']!=r['answer']),
                correct=float(answer['answer']==r['label']),distances=distances)
            results.append(result); write_jsonl(out/'results.jsonl',results)
        print('PROGRESS',name,i+1,20,flush=True)
    stage_summary={}
    for condition in ('opposite_depth','same_depth'):
        rr=[x for x in results if x['stage']==name and x['condition']==condition]
        stage_summary[condition]=dict(n=len(rr),valid_flips=sum(x['valid_flip'] for x in rr),
            correct=sum(x['correct'] for x in rr),invalid=sum(not x['format_valid'] for x in rr),
            incomplete=sum(not x['complete'] for x in rr),max_abs_delta_p_A=max(x['abs_delta_p_A'] for x in rr),
            min_AB_mass=min(x['p_A']+x['p_B'] for x in rr),
            statistics={m:cluster_interval(rr,m) for m in ('valid_flip','abs_delta_p_A','delta_toward_opposite_label')})
    paired=[]
    for r in rows:
        a=next(x for x in results if x['stage']==name and x['id']==r['id'] and x['condition']=='opposite_depth')
        b=next(x for x in results if x['stage']==name and x['id']==r['id'] and x['condition']=='same_depth')
        paired.append(dict(id=r['id'],family=r['family'],flip_difference=a['valid_flip']-b['valid_flip'],
            directional_probability_difference=a['delta_toward_opposite_label']-b['delta_toward_opposite_label']))
    write_json(out/f'{name}_paired.json',paired)
    stage_summary['paired']={m:cluster_interval(paired,m) for m in ('flip_difference','directional_probability_difference')}
    summaries[name]=stage_summary; write_json(out/'stage_summaries.json',summaries)
    print('STAGE_DONE',name,json.dumps(stage_summary),flush=True)

for stage in stages: run_stage(*stage)
extend=not any(x['valid_flip'] for x in results if x['stage']=='all_layers' and x['condition']=='opposite_depth')
write_json(out/'extension_decision.json',dict(extend=extend,criterion='No valid opposite-depth answer flip in all_layers'))
if extend: run_stage('all_layers_thought_prefix',list(range(28)),'thought_prefix')
summary=dict(targets=20,families=len({r['family'] for r in rows}),baseline_correct=sum(r['answer']==r['label'] for r in rows),
    identity_passed=len(identities),interventions=len(results),extension_executed=extend,stages=summaries)
write_json(out/'summary.json',summary)
archive=Path('/content')/(out.name+'-reports.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in out.rglob('*'):
        if p.is_file(): z.write(p,p.relative_to(root))
    z.write(Path(__file__),'scripts/block_intervention.py')
meta=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha256(archive))
write_json(root/'runs/latest-block-intervention-archive.json',meta)
print('BLOCK_INTERVENTION_COMPLETED',json.dumps(summary),flush=True)
print('ARCHIVE',json.dumps(meta),flush=True)
