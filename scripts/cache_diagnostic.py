"""Colab-only diagnostic: cache, precision, and final-head sensitivity."""
import gc
import json
import os
import time
from pathlib import Path

if not Path('/content').exists() or 'COLAB_RELEASE_TAG' not in os.environ:
    raise RuntimeError('Run this diagnostic in Colab only.')

import torch
from PIL import Image
from covt_pilot.model import CoVTModel, reset_rope
from covt_pilot.io import read_json, read_jsonl, write_json

torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)
root = Path('/content/covt-pilot')
prior = Path(read_json(root/'runs/latest-colab.json')['run_dir'])
audit = read_json(prior/'resources/report.json')
all_rows = read_jsonl(prior/'scenes/manifest.jsonl')
selected_ids = ['scene_00000_nuisance','scene_00001_base','scene_00000_base','scene_00002_base']
selected = [next(r for r in all_rows if r['id']==i) for i in selected_ids]
out = root/'runs'/time.strftime('cache-diagnostic-%Y%m%d-%H%M%S',time.gmtime())
out.mkdir(parents=True,exist_ok=False)
write_json(root/'runs/latest-diagnostic.json',{'run_dir':str(out)})
records=[]


def run_one(adapter, inputs, use_cache):
    reset_rope(adapter.model)
    with torch.inference_mode():
        result=adapter.model.generate(**inputs,max_new_tokens=192,do_sample=False,
            use_cache=use_cache,return_dict_in_generate=True,output_scores=True,output_logits=True)
    prefix=inputs.input_ids.shape[1]
    ids=result.sequences[0,prefix:].cpu().tolist()
    answer_ids={s:adapter.processor.tokenizer.encode(s,add_special_tokens=False)[0] for s in ('A','B')}
    # Retain the entire per-step top-five and A/B scores; never compare different prefixes silently.
    steps=[]
    for i,(raw,score) in enumerate(zip(result.logits,result.scores)):
        values,indices=torch.topk(score[0].float(),5)
        steps.append({'step':i,'prefix_ids':ids[:i],
            'raw_dtype':str(raw.dtype),'score_dtype':str(score.dtype),
            'A':float(score[0,answer_ids['A']]),'B':float(score[0,answer_ids['B']]),
            'A_minus_B':float(score[0,answer_ids['A']]-score[0,answer_ids['B']]),
            'raw_A_minus_B':float(raw[0,answer_ids['A']]-raw[0,answer_ids['B']]),
            'top5_ids':indices.cpu().tolist(),'top5_scores':values.cpu().tolist()})
    return {'ids':ids,'text':adapter.processor.tokenizer.decode(ids,skip_special_tokens=False),'steps':steps}


for mode,dtype,head_fp32 in [('bf16','bfloat16',False),('bf16_fp32_head','bfloat16',True),
                             ('fp16','float16',False),('fp32','float32',False)]:
    print('MODE',mode,flush=True)
    adapter=CoVTModel(audit['model_id'],audit['model_revision'],'cuda',dtype,'eager')
    head=adapter.model.get_output_embeddings()
    hook=None
    if head_fp32:
        head.float()
        hook=head.register_forward_pre_hook(lambda module,args:(args[0].float(),)+args[1:])
    for row in selected:
        image=Image.open(prior/'scenes'/row['image']).convert('RGB')
        messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':row['question']}]}]
        prompt=adapter.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=adapter.processor(text=[prompt],images=[image],return_tensors='pt').to(adapter.input_device)
        cached=run_one(adapter,inputs,True)
        repeat=run_one(adapter,inputs,True)
        uncached=run_one(adapter,inputs,False)
        diffs=[i for i,(a,b) in enumerate(zip(cached['ids'],uncached['ids'])) if a!=b]
        divergence=diffs[0] if diffs else None
        record={'id':row['id'],'mode':mode,'label':row['label'],
            'cached_repeat_equal':cached['ids']==repeat['ids'],
            'cached_uncached_equal':cached['ids']==uncached['ids'],
            'first_divergence':divergence,'cached':cached,'repeat':repeat,'uncached':uncached,
            'fp32_head':head_fp32,'max_memory_allocated':torch.cuda.max_memory_allocated()}
        records.append(record)
        write_json(out/'results.json',records)
        k=divergence if divergence is not None else 20
        detail={s:r['steps'][k]['A_minus_B'] for s,r in [('cached',cached),('repeat',repeat),('uncached',uncached)]}
        print(json.dumps({'id':row['id'],'mode':mode,'equal':record['cached_uncached_equal'],
            'repeat_equal':record['cached_repeat_equal'],'step':k,'A_minus_B':detail}),flush=True)
    if hook: hook.remove()
    del adapter,head,inputs
    gc.collect()
    torch.cuda.empty_cache()
summary={mode:{'examples':len(rows),'cache_mismatches':sum(not r['cached_uncached_equal'] for r in rows),
    'repeat_mismatches':sum(not r['cached_repeat_equal'] for r in rows)}
    for mode in ('bf16','bf16_fp32_head','fp16','fp32')
    for rows in [[r for r in records if r['mode']==mode]]}
write_json(out/'summary.json',summary)
print('DIAGNOSTIC_SUMMARY',json.dumps(summary),flush=True)
