"""Prespecified behavioral controls for label binding and depth wording."""
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .io import read_jsonl, write_jsonl


def parse_choice(text, choices):
    tags = re.findall(r'<answer>\s*(.*?)\s*</answer>', text, re.I | re.S)
    candidate = tags[-1] if tags else text.strip()
    candidate = re.sub(r'^\\boxed\{(.*?)\}[.!]?$', r'\1', candidate.strip())
    candidate = candidate.strip().rstrip('.!').strip().lower()
    return next((c for c in choices if candidate == c.lower()), None)


def build_cases(source, out):
    """All eight prior pilot families, base and label-swap; no correctness filter."""
    source, out = Path(source), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    rows = read_jsonl(source/'scenes/manifest.jsonl')
    try:
        font = ImageFont.truetype('DejaVuSans-Bold.ttf', 24)
    except OSError:
        font = ImageFont.load_default(size=24)
    for row in rows:
        if row['split'] != 'pilot' or row['variant'] not in ('base', 'label_swap'):
            continue
        ab = row['points']
        left = 'A' if ab[0][0] < ab[1][0] else 'B'
        with np.load(source/'scenes'/row['geometry']) as z:
            ids = z['object_ids']
            object_ab = [int(ids[y,x]) for x,y in ab]
            if min(object_ab) < 0 or len(set(object_ab)) != 2:
                raise ValueError('Markers must refer to two distinct visible objects.')
            areas = [int((ids == obj).sum()) for obj in object_ab]
        larger = 'A' if areas[0] > areas[1] else 'B'
        if min(areas)/max(areas) > .95:
            raise ValueError('Apparent-size diagnostic is ambiguous.')
        image = Image.open(source/'scenes'/row['image']).convert('RGB')
        marked = out/f"{row['id']}_original.png"
        image.save(marked)
        image = Image.open(source/'scenes'/Path(row['image']).parent/'unmarked.png').convert('RGB')
        draw = ImageDraw.Draw(image)
        for label,(x,y) in zip(('A','B'),ab):
            draw.ellipse((x-3,y-3,x+3,y+3), fill='white',outline='black')
            draw.text((x+12,y-18),label,font=font,fill='white',stroke_width=2,stroke_fill='black')
        large = out/f"{row['id']}_large.png"
        image.save(large)
        near = 'Which marked point is closer to the camera, A or B? Answer only A or B.'
        left_q = 'Which label is on the left side of the image, A or B? Answer only A or B.'
        tasks = [
            ('original_depth',row['question'],row['label'],('A','B'),marked),
            ('plain_near',near,row['label'],('A','B'),marked),
            ('plain_far','Which marked point is farther from the camera, A or B? Answer only A or B.',
             'B' if row['label']=='A' else 'A',('A','B'),marked),
            ('label_left',left_q,left,('A','B'),marked),
            ('apparent_size','Which labeled sphere occupies more area in the image, A or B? Answer only A or B.',
             larger,('A','B'),marked),
            ('side_near','Which sphere is closer to the camera, the left sphere or the right sphere? Answer only left or right.',
             'left' if row['label']==left else 'right',('left','right'),marked),
            ('large_plain_near',near,row['label'],('A','B'),large),
            ('large_label_left',left_q,left,('A','B'),large),
        ]
        for task,question,label,choices,path in tasks:
            cases.append(dict(id=f"{row['id']}__{task}",source_id=row['id'],family=row['family'],
                variant=row['variant'],task=task,image=str(path.resolve()),question=question,
                label=label,choices=list(choices),visible_areas=areas))
    write_jsonl(out/'cases.jsonl',cases)
    return cases
