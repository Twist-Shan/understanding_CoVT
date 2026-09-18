"""Colab-only seven-layer scan on the existing matched 20-image cohort."""
import json
import os
from pathlib import Path
import time
import zipfile
if not Path('/content').is_dir() or 'COLAB_RELEASE_TAG' not in os.environ:
    raise RuntimeError('Run on Colab only.')
import torch
from PIL import Image
from covt_pilot.io import read_json, read_jsonl, write_json, write_jsonl, sha256
from covt_pilot.model import CoVTModel, reset_rope, language_norm, extract_answer
from covt_pilot.metrics import cluster_interval

root=Path('/content/covt-pilot')
source=Path(read_json(root/'runs/latest-balanced-intervention.json')['run_dir'])
native=read_jsonl(source/'native.jsonl')
matched=read_json(source/'paired_analysis.json')['matched_donors']['all']['rows']
selected={r['id'] for r in matched}
rows=[r for r in native if r['id'] in selected]
assert len(rows)==20
layers=[0,4,8,14,20,24,27]
out=root/'runs'/time.strftime('layer-scan-%Y%m%d-%H%M%S',time.gmtime()); out.mkdir()
write_json(root/'runs/latest-layer-scan.json',{'run_dir':str(out)})
write_json(out/'plan.json',dict(source=str(source),source_sha256=sha256(source/'native.jsonl'),
    targets=[r['id'] for r in rows],layers=layers,conditions=['identity','opposite_depth','zero'],
    probability_scope='Teacher-forced original continuation up to, but excluding, the A/B answer token; P(A|A or B) and full-vocabulary probabilities. Distinct from free rollout.',
    rollout_scope='From immediately after last natural depth pad; no old KV or answer suffix.',
    positive_control='Transplant final-norm answer-predictor state from a sample with opposite emitted answer. Readout pipeline check, not depth-specific positive control.',
    limitations='Single seed, six matched scene families; zero is off-manifold. Fixed-context probability diagnostic is not a total causal effect.'))
audit=read_json(Path(read_json(root/'runs/latest-colab.json')['run_dir'])/'resources/report.json')
torch.manual_seed(0); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
model=CoVTModel(audit['model_id'],audit['model_revision'],'cuda','float32','eager')
modules=dict(model.model.named_modules()); _,norm=language_norm(model.model)
for k in layers: assert f'model.layers.{k}' in modules
tok=model.processor.tokenizer
ab=[tok.encode(letter,add_special_tokens=False) for letter in 'AB']
assert all(len(x)==1 for x in ab)
ab=[x[0] for x in ab]
eos=model.model.generation_config.eos_token_id
eos={eos} if isinstance(eos,int) else set(eos)

def context(row):
    image=Image.open(source/'scenes'/row['image']).convert('RGB')
    inputs=model.processor(text=[row['prompt']],images=[image],return_tensors='pt').to(model.input_device)
    assert inputs.input_ids.shape[1]==row['prefix_length']
    full=torch.cat([inputs.input_ids,torch.tensor([row['generated_ids']],device=model.input_device)],dim=1)
    p=torch.tensor(row['depth_positions'],device=model.input_device)
    hits=[i for i,t in enumerate(row['generated_ids']) if t in ab]
    assert len(hits)==1 and row['generated_ids'][hits[0]]==ab['AB'.index(row['answer'])]
    answer_at=row['prefix_length']+hits[0]
    assert answer_at>int(p[-1])+1
    return inputs,full[:,:int(p[-1])+1],full[:,:answer_at],p

@torch.inference_mode()
def forward(inputs,ids,p,layer_id=None,replacement=None,capture=False,answer_state=None):
    handles=[]; states={}; terminal=[]; predictor=[]
    def make_pre(k):
        def pre(module,args,kwargs):
            h=args[0] if args else kwargs['hidden_states']
            if capture: states[k]=h[:,p].detach().cpu().clone()
            if k==layer_id and replacement is not None:
                h=h.clone(); h[:,p]=replacement.to(h.device,h.dtype)
                if args: args=(h,)+args[1:]
                else: kwargs=dict(kwargs,hidden_states=h)
            return args,kwargs
        return pre
    for k in (layers if capture else [layer_id] if layer_id is not None else []):
        handles.append(modules[f'model.layers.{k}'].register_forward_pre_hook(make_pre(k),with_kwargs=True))
    def post(module,args,h):
        if answer_state is not None:
            h=h.clone(); h[:,-1]=answer_state.to(h.device,h.dtype)
        terminal.append(h[:,p].detach().cpu().clone()); predictor.append(h[:,-1].detach().cpu().clone())
        return h
    handles.append(norm.register_forward_hook(post))
    try:
        reset_rope(model.model)
        result=model.model(**dict(inputs,input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True))
        assert len(terminal)==len(predictor)==1
        return result.logits[:,-1].float().cpu(),states,terminal[0],predictor[0]
    finally:
        for h in handles: h.remove()

def probability(logits):
    x=logits[0]; pair=x[ab].softmax(0); full=x.softmax(0)
    return dict(logit_A_minus_B=float(x[ab[0]]-x[ab[1]]),conditional_p_A=float(pair[0]),
        p_A=float(full[ab[0]]),p_B=float(full[ab[1]]),argmax=int(x.argmax()))

def rollout(inputs,prefix,p,k=None,replacement=None):
    ids=prefix.clone(); generated=[]
    for step in range(128):
        logits,_,_,_=forward(inputs,ids,p,k,replacement)
        token=int(logits.argmax(-1).item()); generated.append(token)
        ids=torch.cat([ids,torch.tensor([[token]],device=ids.device)],dim=1)
        if token in eos: break
    return dict(ids=generated,text=tok.decode(generated,skip_special_tokens=False),
        answer=extract_answer(tok.decode(generated,skip_special_tokens=True)),complete=generated[-1] in eos)

captured={}; captured_at_answer={}; answer_states={}; original_logits={}; baselines={}; prefix_checks=[]
for i,row in enumerate(rows):
    inputs,prefix,answer_prefix,p=context(row)
    _,states,_,_=forward(inputs,prefix,p,capture=True)
    logits,states_at_answer,_,predictor=forward(inputs,answer_prefix,p,capture=True)
    checks={str(k):dict(max_abs=float((states[k]-states_at_answer[k]).abs().max()),
        exact=torch.equal(states[k],states_at_answer[k]),
        close=torch.allclose(states[k],states_at_answer[k],atol=1e-3,rtol=1e-5)) for k in layers}
    prefix_checks.append(dict(id=row['id'],layers=checks,atol=1e-3,rtol=1e-5))
    write_json(out/'prefix_checks.json',prefix_checks)
    print('PREFIX_CHECK',row['id'],json.dumps(checks),flush=True)
    assert all(c['close'] for c in checks.values()), 'Prefix drift exceeds FP32 tolerance'
    baseline=rollout(inputs,prefix,p)
    expected=row['generated_ids'][int(p[-1])+1-row['prefix_length']:]
    assert baseline['ids']==expected and baseline['complete'], 'Baseline rollout mismatch'
    assert int(logits.argmax(-1))==ab['AB'.index(row['answer'])]
    captured[row['id']]=states; captured_at_answer[row['id']]=states_at_answer
    answer_states[row['id']]=predictor; original_logits[row['id']]=logits
    baselines[row['id']]=baseline
    write_json(out/f"{row['id']}_baseline.json",dict(id=row['id'],**baseline,**probability(logits)))
    print('CAPTURED',i+1,len(rows),row['id'],flush=True)

positives=[]
for row in rows:
    donor=next(r for r in rows if r['answer']!=row['answer'])
    inputs,_,answer_prefix,p=context(row)
    logits,_,_,_=forward(inputs,answer_prefix,p,answer_state=answer_states[donor['id']])
    exact=torch.equal(logits,original_logits[donor['id']])
    pred=tok.decode([int(logits.argmax(-1))])
    positives.append(dict(id=row['id'],donor=donor['id'],baseline_answer=row['answer'],prediction=pred,
        donor_logits_equal=exact,pass_control=bool(exact and pred==donor['answer'] and pred!=row['answer'])))
write_jsonl(out/'positive_controls.jsonl',positives)
assert all(r['pass_control'] for r in positives), 'Readout positive control failed'
print('POSITIVE_CONTROL',len(positives),len(positives),flush=True)

results=[]; identities=[]
for k in layers:
    for i,row in enumerate(rows):
        inputs,prefix,answer_prefix,p=context(row)
        own=captured[row['id']][k]
        identity,_,_,_=forward(inputs,answer_prefix,p,k,captured_at_answer[row['id']][k])
        same=rollout(inputs,prefix,p,k,own)
        exact=torch.equal(identity,original_logits[row['id']])
        close=torch.allclose(identity,original_logits[row['id']],atol=1e-4,rtol=1e-5)
        passed=bool(exact and same['ids']==baselines[row['id']]['ids'])
        identities.append(dict(id=row['id'],layer=k,pass_identity=passed,logits_exact=exact,
            logits_max_abs=float((identity-original_logits[row['id']]).abs().max()),atol=1e-4,rtol=1e-5,
            delta_logit_A_minus_B=probability(identity)['logit_A_minus_B']-probability(original_logits[row['id']])['logit_A_minus_B']))
        write_jsonl(out/'identity.jsonl',identities)
        assert passed, 'Layer identity failed; stop rather than silently exclude'
        opposite=f"{row['family']}_d{1-row['depth_order']}_s{row['size_order']}"
        assert opposite in captured
        baseline_probability=probability(original_logits[row['id']])
        for condition,replacement in [('opposite_depth',captured[opposite][k]),('zero',torch.zeros_like(own))]:
            answer_replacement=captured_at_answer[opposite][k] if condition=='opposite_depth' else torch.zeros_like(own)
            logits,_,_,_=forward(inputs,answer_prefix,p,k,answer_replacement)
            metrics=probability(logits)
            answer=rollout(inputs,prefix,p,k,replacement)
            results.append(dict(id=row['id'],family=row['family'],cue=row['cue'],label=row['label'],layer=k,
                condition=condition,donor=opposite if condition=='opposite_depth' else None,**answer,**metrics,
                delta_logit_A_minus_B=metrics['logit_A_minus_B']-baseline_probability['logit_A_minus_B'],
                abs_delta_logit=abs(metrics['logit_A_minus_B']-baseline_probability['logit_A_minus_B']),
                delta_p_A=metrics['conditional_p_A']-baseline_probability['conditional_p_A'],
                abs_delta_p_A=abs(metrics['conditional_p_A']-baseline_probability['conditional_p_A']),
                answer_changed=float(answer['answer']!=row['answer']),answer_correct=float(answer['answer']==row['label']),
                valid=float(answer['answer'] in ('A','B')),replacement_max_abs=float((replacement-own).abs().max())))
            write_jsonl(out/'results.jsonl',results)
        print('SCAN',k,i+1,len(rows),flush=True)
    print('LAYER_DONE',k,flush=True)

summary=dict(targets=len(rows),families=len({r['family'] for r in rows}),identity_passed=len(identities),
    positive_controls_passed=sum(r['pass_control'] for r in positives),layers={})
for k in layers:
    summary['layers'][str(k)]={}
    for condition in ('opposite_depth','zero'):
        selected=[r for r in results if r['layer']==k and r['condition']==condition]
        summary['layers'][str(k)][condition]=dict(n=len(selected),changed=sum(r['answer_changed'] for r in selected),
            correct=sum(r['answer_correct'] for r in selected),invalid=sum(not r['valid'] for r in selected),
            incomplete=sum(not r['complete'] for r in selected),max_abs_delta_logit=max(r['abs_delta_logit'] for r in selected),
            max_abs_delta_p_A=max(r['abs_delta_p_A'] for r in selected),
            statistics={m:cluster_interval(selected,m) for m in ('answer_changed','abs_delta_logit','abs_delta_p_A')})
write_json(out/'summary.json',summary)
archive=Path('/content')/(out.name+'-reports.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in out.rglob('*'):
        if p.is_file(): z.write(p,p.relative_to(root))
    z.write(Path(__file__),'scripts/layer_scan.py')
meta=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha256(archive))
write_json(root/'runs/latest-layer-scan-archive.json',meta)
print('LAYER_SCAN_COMPLETED',json.dumps(summary),flush=True)
print('ARCHIVE',json.dumps(meta),flush=True)
