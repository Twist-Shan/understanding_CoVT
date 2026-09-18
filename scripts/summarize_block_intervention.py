"""Colab-only final audit; preserve primary outputs and collect logs in archive."""
import json
import os
import zipfile
from pathlib import Path
if not Path('/content').is_dir() or 'COLAB_RELEASE_TAG' not in os.environ:
    raise RuntimeError('Run on Colab only.')
from covt_pilot.io import read_json, read_jsonl, write_json, sha256
root=Path('/content/covt-pilot')
out=Path(read_json(root/'runs/latest-block-intervention.json')['run_dir'])
summary=read_json(out/'summary.json')
rows=read_jsonl(out/'results.jsonl'); identities=read_jsonl(out/'identity.jsonl')
assert len(rows)==summary['interventions']==40*len(summary['stages'])
assert len(identities)==20*len(summary['stages']) and all(r['pass_identity'] for r in identities)
baseline={p.name.removesuffix('_baseline.json'):read_json(p) for p in out.glob('*_baseline.json')}
assert len(baseline)==20
token_map={b['argmax']:b['answer'] for b in baseline.values()}; assert set(token_map.values())=={'A','B'}
details=[]
for stage in summary['stages']:
    for condition in ('opposite_depth','same_depth'):
        rr=[r for r in rows if r['stage']==stage and r['condition']==condition]
        assert len(rr)==20 and set(r['id'] for r in rr)==set(baseline)
        assert all(r['baseline_answer']==baseline[r['id']]['answer'] for r in rr)
        discordant=[r for r in rr if baseline[r['donor']]['answer']!=r['baseline_answer']]
        details.append(dict(stage=stage,condition=condition,n=len(rr),
            position_counts=sorted(set(len(r['positions']) for r in rr)),
            valid_flips=sum(r['valid_flip'] for r in rr),correct=sum(r['correct'] for r in rr),
            invalid=sum(not r['format_valid'] for r in rr),ambiguous=sum(r['ambiguous_AB'] for r in rr),
            incomplete=sum(not r['complete'] for r in rr),
            fixed_context_valid_flips=sum(r['argmax'] in token_map and token_map[r['argmax']]!=r['baseline_answer'] for r in rr),
            full_token_sequence_changes=sum(r['ids']!=baseline[r['id']]['ids'] for r in rr),
            max_abs_delta_p_A=max(r['abs_delta_p_A'] for r in rr),
            max_abs_delta_logit=max(abs(r['delta_logit_A_minus_B']) for r in rr),
            min_AB_mass=min(r['p_A']+r['p_B'] for r in rr),
            max_relative_state_l2=max(d['relative_l2'] for r in rr for d in r['distances'].values()),
            donor_answer_discordant=dict(n=len(discordant),valid_flips=sum(r['valid_flip'] for r in discordant),
                invalid=sum(not r['format_valid'] for r in discordant),
                max_abs_delta_p_A=max((r['abs_delta_p_A'] for r in discordant),default=None)),
            changes=[dict(id=r['id'],family=r['family'],label=r['label'],baseline=r['baseline_answer'],answer=r['answer'],
                donor=r['donor'],valid=r['format_valid'],text=r['text']) for r in rr if r['valid_flip'] or not r['format_valid']]))
audit=dict(details=details,identity_exact=len(identities),
    prefix_max_abs=max(v['max_abs'] for r in read_json(out/'prefix_checks.json') for v in r['layers'].values()))
write_json(out/'final_audit.json',audit)
print('BLOCK_AUDIT',json.dumps(audit),flush=True)
archive=Path('/content')/(out.name+'-final-reports.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in out.rglob('*'):
        if p.is_file(): z.write(p,p.relative_to(root))
    for name in ('block_intervention.py','summarize_block_intervention.py'):
        z.write(root/'scripts'/name,'scripts/'+name)
    z.write(root/'runs/block-intervention-console.log','logs/block-intervention-console.log')
meta=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha256(archive))
write_json(root/'runs/latest-block-intervention-archive.json',meta)
print('ARCHIVE',json.dumps(meta),flush=True)
