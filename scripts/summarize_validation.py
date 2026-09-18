"""Compute paired diagnostics and export compact results on Colab."""
import hashlib
import json
import os
from pathlib import Path
import zipfile

if not Path('/content').is_dir() or 'COLAB_RELEASE_TAG' not in os.environ:
    raise RuntimeError('Run experiment analysis on Colab only.')

import numpy as np
from PIL import Image, ImageDraw
from covt_pilot.io import read_json, read_jsonl, write_json
from covt_pilot.metrics import cluster_interval

root=Path('/content/covt-pilot')
answer=Path(read_json(root/'runs/latest-answer-diagnostic.json')['run_dir'])
shift=Path(read_json(root/'runs/latest-shift-validation.json')['run_dir'])
assert (shift/'native/summary.json').is_file()
previous=Path(read_json(root/'runs/latest-followup.json')['run_dir'])
old={r['id']:r for r in read_jsonl(previous/'e0/results.jsonl')}
answers=read_jsonl(answer/'results.jsonl')
paired={}
for model in ('covt','backbone'):
    lookup={(r['source_id'],r['task']):r for r in answers if r['model']==model}
    original=[r for r in lookup.values() if r['task']=='original_depth']
    differences=[dict(family=r['family'],difference=lookup[(r['source_id'],'plain_near')]['correct']-r['correct']) for r in original]
    near=[r for r in lookup.values() if r['task']=='plain_near']
    base=[r for r in near if r['variant']=='base']
    paired[model]=dict(near_minus_original=cluster_interval(differences,'difference'),
        near_far_opposite=sum(r['answer']!=lookup[(r['source_id'],'plain_far')]['answer'] for r in near),
        near_far_n=len(near),
        near_label_swap_flip=sum(r['answer']!=lookup[(r['source_id'].replace('_base','_label_swap'),'plain_near')]['answer'] for r in base),
        label_swap_families=len(base),
        side_label_swap_invariant=sum(lookup[(r['source_id'],'side_near')]['answer']==lookup[(r['source_id'].replace('_base','_label_swap'),'side_near')]['answer'] for r in base))
    if model=='covt':
        paired[model]['old_original_token_matches']=sum(r['generated_ids']==old[r['source_id']]['generated_ids'] for r in original)
        paired[model]['old_original_n']=len(original)
write_json(answer/'paired_summary.json',paired)

rows=read_jsonl(shift/'scenes/manifest.jsonl')
geometry=[]
for row in rows:
    with np.load(shift/'scenes'/row['geometry'],allow_pickle=False) as z:
        ids=z['object_ids']
        objects=[int(ids[y,x]) for x,y in row['points']]
        area=[int((ids==obj).sum()) for obj in objects]
    bigger='A' if area[0]>area[1] else 'B' if area[1]>area[0] else None
    geometry.append(dict(id=row['id'],family=row['family'],stratum=row['stratum'],native_audit=row['native_audit'],
        apparent_area=area,bigger_is_nearer=float(bigger==row['label'])))
native=read_jsonl(shift/'native/results.jsonl')
frozen=read_jsonl(shift/'frozen_results.jsonl')
controls=read_jsonl(shift/'native/controls.jsonl')
paired_native=[]
for row in native:
    if not row.get('e0_checks_pass'): continue
    own=next(r for r in controls if r['id']==row['id'] and r['condition']=='native')
    fixed=next(r for r in controls if r['id']==row['id'] and r['condition']=='fixed_mean')
    paired_native.append(dict(id=row['id'],family=row['family'],difference=fixed['correct']-own['correct'],
        label_agreement=float(fixed['prediction']==own['prediction']),abs_margin_delta=fixed['abs_margin_delta']))
summary=dict(geometry_baseline={scope:cluster_interval([r for r in geometry if scope=='all' or r['native_audit']], 'bigger_is_nearer')
        for scope in ('all','native_subset')},
    native_by_stratum={s:{metric:cluster_interval([r for r in native if r.get('e0_checks_pass') and r['stratum']==s],metric)
        for metric in ('correct','answer_correct','abstain')} for s in sorted({r['stratum'] for r in native})},
    paired_fixed_vs_native={metric:cluster_interval(paired_native,metric) for metric in ('difference','label_agreement','abs_margin_delta')},
    abstentions=[r for r in frozen if r['abstain']],
    frozen_direction_errors=[r for r in frozen if not r['abstain'] and not r['correct']],
    native_failures=[r['id'] for r in native if not r.get('e0_checks_pass')])
write_json(shift/'analysis.json',summary)
write_json(shift/'geometry_baseline.json',geometry)

# Fixed first example of each stratum, selected before model inspection.
selected=[r for r in rows if r['variant']=='base'][:5]
sheet=Image.new('RGB',(336*5,370),'white'); draw=ImageDraw.Draw(sheet)
for i,row in enumerate(selected):
    sheet.paste(Image.open(shift/'scenes'/row['image']).convert('RGB'),(336*i,34))
    draw.text((336*i+8,8),row['stratum']+' GT='+row['label'],fill='black')
sheet.save(shift/'scene_contact_sheet.png')
paths=[p for folder in (answer,shift) for p in folder.rglob('*') if p.is_file()
       and (p.suffix not in ('.npy','.npz') or p.name.endswith('_hidden.npy'))]
paths += [p for pattern in ('scripts/*.py','src/covt_pilot/*.py','tests/*.py') for p in root.glob(pattern)]
paths += [root/'runs/answer-diagnostic-console.log',root/'runs/shift-validation-console.log',root/'runs/backbone-revision.json',root/'pyproject.toml']
target=Path('/content/covt-answer-shift-reports-20260918.zip')
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in paths: archive.write(path,path.relative_to(root))
with target.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
manifest={'path':str(target),'bytes':target.stat().st_size,'sha256':digest,
    'contents':'All JSON/JSONL, scene images, hidden states, logs, runtime code. Feature/geometry NPZ and depth-map NPY excluded.'}
write_json(Path('/content/covt-answer-shift-archive.json'),manifest)
print('ANSWER_PAIRED',json.dumps(paired),flush=True)
print('SHIFT_ANALYSIS',json.dumps(summary),flush=True)
print('REPORT_ARCHIVE',json.dumps(manifest),flush=True)
