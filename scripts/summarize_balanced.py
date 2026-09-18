"""Colab-only matched-target analysis; no new model inference."""
import os
from pathlib import Path
import json
import zipfile
if not Path('/content').is_dir() or 'COLAB_RELEASE_TAG' not in os.environ:
    raise RuntimeError('Run on Colab only.')
from covt_pilot.io import read_json, read_jsonl, write_json, sha256
from covt_pilot.metrics import cluster_interval
root=Path('/content/covt-pilot')
out=Path(read_json(root/'runs/latest-balanced-intervention.json')['run_dir'])
assert (out/'summary.json').is_file()
native=read_jsonl(out/'native.jsonl'); frozen=read_jsonl(out/'frozen.jsonl')
interventions=read_jsonl(out/'interventions.jsonl')
result={'all_attempted':{},'matched_donors':{},'failures':[
    {'id':r['id'],'reason':r.get('reason'),'cached_equal':r['cached_uncached_tokens_equal']}
    for r in native if not r['eligible']], 'identity_failures':[]}
for p in out.glob('*_identity.json'):
    r=read_json(p)
    if not r['identity_pass']: result['identity_failures'].append(dict(file=p.name,**r))
for cue in ('all','aligned','conflict'):
    a=[r for r in native if cue=='all' or r['cue']==cue]
    f=[r for r in frozen if cue=='all' or r['cue']==cue]
    result['all_attempted'][cue]=dict(n=len(a),language_correct=sum(r['answer']==r['label'] for r in a),
        language_invalid=sum(r['answer'] not in ('A','B') for r in a),
        eligible=sum(r['eligible'] for r in a),
        fixed_correct=sum(r['fixed']['decoder_correct'] for r in f),
        expert_correct=sum(r['expert']['decoder_correct'] for r in f))
lookup={(r['id'],r['condition']):r for r in interventions}
ids={r['id'] for r in interventions}
for cue in ('all','aligned','conflict'):
    matched=[i for i in sorted(ids) if all((i,c) in lookup for c in ('baseline','opposite_depth','same_depth_other_family'))
        and (cue=='all' or lookup[(i,'baseline')]['cue']==cue)]
    rows=[]
    for i in matched:
        b=lookup[(i,'baseline')]; o=lookup[(i,'opposite_depth')]; s=lookup[(i,'same_depth_other_family')]
        rows.append(dict(id=i,family=b['family'],opposite_flip=o['answer_changed'],same_flip=s['answer_changed'],
            flip_difference=o['answer_changed']-s['answer_changed'],
            opposite_answer=o['answer'],same_answer=s['answer'],baseline_answer=b['answer'],
            opposite_decoder_changed=float(o['decoder_label']!=b['decoder_label']),
            same_decoder_changed=float(s['decoder_label']!=b['decoder_label'])))
    result['matched_donors'][cue]=dict(n=len(rows),rows=rows,statistics={k:cluster_interval(rows,k)
        for k in ('opposite_flip','same_flip','flip_difference','opposite_decoder_changed','same_decoder_changed')})
write_json(out/'paired_analysis.json',result)
archive=Path('/content')/(out.name+'-final-reports.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in out.rglob('*'):
        if p.is_file() and p.suffix!='.npz': z.write(p,p.relative_to(root))
    for name in ('balanced_intervention.py','summarize_balanced.py'):
        z.write(root/'scripts'/name,'scripts/'+name)
    z.write(root/'runs/balanced-intervention-console.log','balanced-intervention-console.log')
meta=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha256(archive))
write_json(root/'runs/latest-balanced-archive.json',meta)
print('PAIRED_ANALYSIS',json.dumps(result))
print('ARCHIVE',json.dumps(meta))
