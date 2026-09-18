"""Run label/wording diagnostics for CoVT and its base model on Colab only."""
import gc
import json
import os
from pathlib import Path
import time

if not Path('/content').is_dir() or 'COLAB_RELEASE_TAG' not in os.environ:
    raise RuntimeError('Run experiments on Colab only.')

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from covt_pilot.diagnostics import build_cases, parse_choice
from covt_pilot.io import read_json, write_json, write_jsonl, sha256
from covt_pilot.metrics import cluster_interval
from covt_pilot.model import reset_rope

root=Path('/content/covt-pilot')
source=Path(read_json(root/'runs/latest-followup.json')['run_dir'])
prior=Path(read_json(root/'runs/latest-colab.json')['run_dir'])
audit=read_json(prior/'resources/report.json')
backbone=read_json(root/'runs/backbone-revision.json')
out=root/'runs'/time.strftime('answer-diagnostic-%Y%m%d-%H%M%S',time.gmtime())
out.mkdir()
write_json(root/'runs/latest-answer-diagnostic.json',{'run_dir':str(out)})
cases=build_cases(source,out/'cases')
write_json(out/'plan.json',{'source_run':str(source),'families':8,'images':16,'cases_per_model':len(cases),
    'selection':'All pilot families, base and label_swap; no accuracy filtering.',
    'models':{'covt':{'model_id':audit['model_id'],'revision':audit['model_revision']},'backbone':backbone},
    'dtype':'float32','attention':'eager','max_new_tokens':192,
    'calibration':'No prompt selection or tuning on these results. Exploratory diagnostic.',
    'cache_audit':'First base example for each of eight tasks, cached vs uncached.'})
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
results=[]
for name,info in [('covt',{'model_id':audit['model_id'],'revision':audit['model_revision']}),('backbone',backbone)]:
    print('LOADING',name,info,flush=True)
    processor=AutoProcessor.from_pretrained(info['model_id'],revision=info['revision'],trust_remote_code=False)
    model,loading=Qwen2_5_VLForConditionalGeneration.from_pretrained(info['model_id'],revision=info['revision'],
        torch_dtype=torch.float32,attn_implementation='eager',device_map={'':'cuda'},output_loading_info=True)
    if loading.get('missing_keys') or loading.get('mismatched_keys'):
        raise ValueError(f'Incomplete model restore: {loading}')
    model.eval()
    write_json(out/f'{name}_loading.json',loading)
    first_family=cases[0]['family']
    with torch.inference_mode():
        for index,case in enumerate(cases):
            image=Image.open(case['image']).convert('RGB')
            messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':case['question']}]}]
            prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inputs=processor(text=[prompt],images=[image],return_tensors='pt').to('cuda')
            reset_rope(model)
            seq=model.generate(**inputs,max_new_tokens=192,do_sample=False,use_cache=True)
            ids=seq[0,inputs.input_ids.shape[1]:].cpu().tolist()
            text=processor.tokenizer.decode(ids,skip_special_tokens=True)
            raw=processor.tokenizer.decode(ids,skip_special_tokens=False)
            eos=model.generation_config.eos_token_id
            eos=[eos] if isinstance(eos,int) else (eos or [])
            answer=parse_choice(text,case['choices'])
            record=case|dict(model=name,prompt=prompt,image_sha256=sha256(case['image']),
                text=text,raw_text=raw,generated_ids=ids,answer=answer,
                correct=float(answer==case['label']),invalid=float(answer is None),
                complete=bool(ids and ids[-1] in eos))
            if case['family']==first_family and case['variant']=='base':
                reset_rope(model)
                other=model.generate(**inputs,max_new_tokens=192,do_sample=False,use_cache=False)
                record['cached_uncached_equal']=bool(torch.equal(seq,other))
                record['uncached_text']=processor.tokenizer.decode(other[0,inputs.input_ids.shape[1]:],skip_special_tokens=False)
            results.append(record)
            write_jsonl(out/'results.jsonl',results)
            print(json.dumps({'model':name,'done':index+1,'total':len(cases),'id':case['id'],
                'answer':answer,'label':case['label'],'correct':record['correct']}),flush=True)
    del model,processor,inputs,seq,other
    gc.collect(); torch.cuda.empty_cache()
summary={'scope':'Behavioral diagnostics only; paired variants are not independent families.',
    'models':{name:{task:{metric:cluster_interval([r for r in results if r['model']==name and r['task']==task],metric)
            for metric in ('correct','invalid')} for task in sorted({r['task'] for r in results})}
            for name in ('covt','backbone')},
    'cache_audit':{name:{'tested':sum(r['model']==name and 'cached_uncached_equal' in r for r in results),
        'mismatched':sum(r['model']==name and r.get('cached_uncached_equal') is False for r in results)}
        for name in ('covt','backbone')},
    'truncated':sum(not r['complete'] for r in results)}
write_json(out/'summary.json',summary)
print('ANSWER_DIAGNOSTIC_COMPLETED',str(out),json.dumps(summary),flush=True)
