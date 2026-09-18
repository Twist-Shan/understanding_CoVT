"""Colab-only final audit and compact layer scan tables."""
import json
import os
import re
from pathlib import Path
import zipfile
if not Path('/content').is_dir() or 'COLAB_RELEASE_TAG' not in os.environ:
    raise RuntimeError('Run on Colab only.')
from covt_pilot.io import read_json, read_jsonl, write_json, sha256
from covt_pilot.metrics import cluster_interval

root=Path('/content/covt-pilot')
out=Path(read_json(root/'runs/latest-layer-scan.json')['run_dir'])
summary=read_json(out/'summary.json')
rows=read_jsonl(out/'results.jsonl')
identities=read_jsonl(out/'identity.jsonl')
assert len(rows)==280 and len(identities)==140
assert all(r['pass_identity'] for r in identities)
details=[]
for layer in (0,4,8,14,20,24,27):
    controls=[r for r in identities if r['layer']==layer]
    noise=max(abs(r['delta_logit_A_minus_B']) for r in controls)
    for condition in ('opposite_depth','zero'):
        rr=[r for r in rows if r['layer']==layer and r['condition']==condition]
        assert len(rr)==20 and len({r['id'] for r in rr})==20
        changed=[r for r in rr if r['answer_changed']]
        for r in rr:
            r['delta_toward_opposite_label']=(1 if r['label']=='B' else -1)*r['delta_p_A']
        largest=max(rr,key=lambda r:r['abs_delta_p_A'])
        formats=[]
        for r in rr:
            letters=re.findall(r'(?<![A-Za-z])[AB](?![A-Za-z])',r['text'])
            tags=re.findall(r'<answer>\s*([AB])\s*</answer>',r['text'])
            unique=next(iter(set(letters))) if len(set(letters))==1 else None
            baseline_answer=read_json(out/f"{r['id']}_baseline.json")['answer']
            strict=len(tags)==1 and set(letters)=={tags[0]}
            formats.append(dict(id=r['id'],original_answer=r['answer'],letters=letters,
                single_complete_answer_tag=strict,unique_letter=unique,
                strict_flip=bool(strict and tags[0]!=baseline_answer),
                unique_letter_flip=bool(unique is not None and unique!=baseline_answer),
                ambiguous=len(set(letters))>1,text=r['text']))
        write_json(out/f'format_layer_{layer}_{condition}.json',formats)
        details.append(dict(layer=layer,condition=condition,n=len(rr),
            valid_flips=sum(r['valid'] for r in changed),invalid=sum(not r['valid'] for r in rr),
            incomplete=sum(not r['complete'] for r in rr),correct=sum(r['answer_correct'] for r in rr),
            max_delta_logit=max(r['abs_delta_logit'] for r in rr),
            max_delta_p_A=max(r['abs_delta_p_A'] for r in rr),
            min_full_AB_mass=min(r['p_A']+r['p_B'] for r in rr),
            max_full_AB_mass=max(r['p_A']+r['p_B'] for r in rr),
            format_audit=dict(complete_answer_tags=sum(r['single_complete_answer_tag'] for r in formats),
                strict_flips=sum(r['strict_flip'] for r in formats),
                unique_letter_flips=sum(r['unique_letter_flip'] for r in formats),
                ambiguous_AB=sum(r['ambiguous'] for r in formats),
                no_AB=sum(not r['letters'] for r in formats),
                unique_letter_count=sum(r['unique_letter'] is not None for r in formats)),
            identity_logit_noise=noise,identity_full_logits_max_abs=max(r['logits_max_abs'] for r in controls),
            toward_opposite_label=cluster_interval(rr,'delta_toward_opposite_label'),
            largest_probability_change=dict(id=largest['id'],label=largest['label'],answer=largest['answer'],
                delta_p_A=largest['delta_p_A'],conditional_p_A=largest['conditional_p_A']),
            changes=[dict(id=r['id'],answer=r['answer'],label=r['label'],text=r['text']) for r in changed],
            cue_counts={cue:dict(n=sum(r['cue']==cue for r in rr),
                changed=sum(r['answer_changed'] for r in rr if r['cue']==cue)) for cue in ('aligned','conflict')}))
target_ids={r['id'] for r in rows}
source=Path(read_json(out/'plan.json')['source'])
baseline=[r for r in read_jsonl(source/'native.jsonl') if r['id'] in target_ids]
assert len(baseline)==20
lookup={r['id']:r for r in baseline}
alignment=[]
for r in baseline:
    donor=lookup[f"{r['family']}_d{1-r['depth_order']}_s{r['size_order']}"]
    alignment.append(dict(id=r['id'],donor=donor['id'],prompt_equal=r['prompt']==donor['prompt'],
        prefix_length_equal=r['prefix_length']==donor['prefix_length'],
        depth_positions_equal=r['depth_positions']==donor['depth_positions']))
audit=dict(baseline=dict(n=len(baseline),correct=sum(r['answer']==r['label'] for r in baseline),
    emitted_A=sum(r['answer']=='A' for r in baseline),label_A=sum(r['label']=='A' for r in baseline)),
    donor_alignment=alignment,details=details,identity_exact=sum(r['logits_exact'] for r in identities),
    prefix_max_abs=max(v['max_abs'] for r in read_json(out/'prefix_checks.json') for v in r['layers'].values()))
write_json(out/'final_audit.json',audit)
print('LAYER_SCAN_AUDIT',json.dumps(audit),flush=True)
archive=Path('/content')/(out.name+'-format-audited-reports.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in out.rglob('*'):
        if p.is_file(): z.write(p,p.relative_to(root))
    for name in ('layer_scan.py','summarize_layer_scan.py'):
        z.write(root/'scripts'/name,'scripts/'+name)
    z.write(root/'runs/layer-scan-console.log','logs/layer-scan-console.log')
meta=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha256(archive))
write_json(root/'runs/latest-layer-scan-archive.json',meta)
print('ARCHIVE',json.dumps(meta),flush=True)
