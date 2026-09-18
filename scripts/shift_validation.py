"""Colab distribution-shift pilot; 200 frozen-readout cases and 40 native cases."""
import argparse
import json
import os
from pathlib import Path
import time

if not Path('/content').is_dir() or 'COLAB_RELEASE_TAG' not in os.environ:
    raise RuntimeError('Run experiments on Colab only.')

import numpy as np
import torch
from PIL import Image
from covt_pilot.controls import token_derangement, donor_for
from covt_pilot.decoder import NativeDepthDecoder, map_to_input
from covt_pilot.expert import DepthExpert
from covt_pilot.io import read_json, read_jsonl, write_json, write_jsonl, sha256
from covt_pilot.metrics import raw_difference, score_difference, cluster_interval
from covt_pilot.model import CoVTModel
from covt_pilot.shift_scenes import generate_shift

parser=argparse.ArgumentParser()
parser.add_argument('stage',choices=['expert','native'])
args=parser.parse_args()
root=Path('/content/covt-pilot')
prior=Path(read_json(root/'runs/latest-colab.json')['run_dir'])
source=Path(read_json(root/'runs/latest-followup.json')['run_dir'])
audit=read_json(prior/'resources/report.json')
old_run=read_json(source/'e0/run.json')
calibration=read_json(source/'e0/calibration.json')
cal_rows=[r for r in read_jsonl(source/'e0/results.jsonl') if r['split']=='calibration' and r['e0_checks_pass']]
cal_states=[]
for row in cal_rows:
    path=source/'e0'/row['artifact']
    assert sha256(path)==row['artifact_sha256']
    with np.load(path,allow_pickle=False) as z: cal_states.append(torch.from_numpy(z['hidden'].copy()))
fixed_mean=torch.stack(cal_states).mean(0); fixed_example=cal_states[0]
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False


def summarize(rows,condition_key):
    return {stratum:{condition:{metric:cluster_interval([r for r in rows
        if (stratum=='all' or r['stratum']==stratum) and r[condition_key]==condition],metric)
        for metric in ('correct','abstain')} for condition in sorted({r[condition_key] for r in rows})}
        for stratum in ['all']+sorted({r['stratum'] for r in rows})}


def score_map(raw,row,cal):
    evaluated=map_to_input(raw,tuple(reversed(row['camera']['size']))).cpu().numpy()
    scored=score_difference(raw_difference(evaluated,row),cal)
    return evaluated,scored


if args.stage=='expert':
    out=root/'runs'/time.strftime('shift-validation-%Y%m%d-%H%M%S',time.gmtime()); out.mkdir()
    write_json(root/'runs/latest-shift-validation.json',{'run_dir':str(out)})
    rows=generate_shift(out/'scenes',50,1901)
    (out/'features').mkdir(); (out/'maps').mkdir()
    write_json(out/'plan.json',dict(source_run=str(source),source_run_sha256=sha256(source/'e0/run.json'),
        images=200,native_images=40,calibration=calibration,fixed_state_ids=[r['id'] for r in cal_rows],
        expert_device='cuda',decoder_dtype='float32',native_dtype='float32',attention='eager',
        native_selection='First 10 families; two from each of five strata, selected by generator before inference.',
        question='Plain closer-to-camera wording; 24px labels. This changes more than geometry relative to initial pilot.',
        scope='All 200 evaluate frozen states. Native-state comparison only on 40 prespecified images.'))
    teacher_info=read_json(prior/'expert_download.json')
    from huggingface_hub import hf_hub_download
    checkpoint=hf_hub_download(teacher_info['model_id'],'depth_anything_v2_vitl.pth',revision=teacher_info['revision'])
    expert=DepthExpert(root/'third_party/CoVT',checkpoint,audit['code']['revision'],'cuda')
    decoder=NativeDepthDecoder.restore(prior/'decoder','cuda').float()
    write_json(out/'expert_provenance.json',expert.provenance)
    results=[]; cache_records=[]
    with torch.inference_mode():
        for index,row in enumerate(rows):
            image=Image.open(out/'scenes'/row['image']).convert('RGB')
            features,teacher,patch_hw,native_hw=expert.encode(image)
            maps={'expert':teacher}
            for name,state in [('fixed_mean',fixed_mean),('fixed_example',fixed_example)]:
                _,maps[name]=decoder(state,features,patch_hw,native_hw)
            for name,raw in maps.items():
                mapped,scored=score_map(raw,row,calibration['expert' if name=='expert' else 'decoder'])
                results.append({k:row[k] for k in ('id','family','variant','stratum','label','native_audit')}|
                    dict(condition=name,margin=scored['margin'],prediction=scored['label'],
                         correct=float(scored['label']==row['label']),abstain=float(scored['abstain'])))
                np.save(out/'maps'/f"{row['id']}_{name}.npy",mapped)
            if row['native_audit']:
                fresh,fresh_depth,_,_=expert.encode(image)
                passed=all(torch.allclose(a,b,atol=.02,rtol=.01) for a,b in zip(features,fresh)) and torch.allclose(teacher,fresh_depth,atol=.02,rtol=.01)
                cache=out/'features'/f"{row['id']}.npz"
                np.savez_compressed(cache,features=np.stack([f[0].cpu().numpy() for f in features]),
                    patch_hw=patch_hw,native_hw=native_hw)
                cache_records.append(dict(id=row['id'],sha256=sha256(cache),expert_repeat_pass=bool(passed),
                    expert_repeat_max_abs=max(float((a-b).abs().max()) for a,b in zip(features,fresh))))
                write_jsonl(out/'feature_index.jsonl',cache_records)
            write_jsonl(out/'frozen_results.jsonl',results)
            print('FROZEN',index+1,len(rows),row['id'],flush=True)
    summary=summarize(results,'condition')
    write_json(out/'frozen_summary.json',summary)
    print('FROZEN_COMPLETED',str(out),json.dumps(summary),flush=True)

else:
    out=Path(read_json(root/'runs/latest-shift-validation.json')['run_dir'])
    rows=[r for r in read_jsonl(out/'scenes/manifest.jsonl') if r['native_audit']]
    index={r['id']:r for r in read_jsonl(out/'feature_index.jsonl')}
    folder=out/'native'; folder.mkdir(exist_ok=False)
    model=CoVTModel(audit['model_id'],audit['model_revision'],'cuda','float32','eager')
    decoder=NativeDepthDecoder.restore(prior/'decoder','cuda').float()
    results=[]; states={}

    def load_features(row):
        path=out/'features'/f"{row['id']}.npz"
        assert sha256(path)==index[row['id']]['sha256']
        with np.load(path,allow_pickle=False) as z:
            return [torch.from_numpy(f.copy()).unsqueeze(0).cuda() for f in z['features']],z['patch_hw'].tolist(),z['native_hw'].tolist()

    with torch.inference_mode():
        for number,row in enumerate(rows):
            image=Image.open(out/'scenes'/row['image']).convert('RGB')
            record=dict(row)
            try:
                prediction,hidden=model.run(image,row['question'])
                record.update(prediction)
                if hidden is not None:
                    features,patch_hw,native_hw=load_features(row)
                    _,raw=decoder(hidden,features,patch_hw,native_hw)
                    _,same=decoder(hidden.clone(),[f.clone() for f in features],patch_hw,native_hw)
                    mapped,scored=score_map(raw,row,calibration['decoder'])
                    record.update(decoder_identity_pass=bool(torch.allclose(raw,same,atol=.02,rtol=.01)),
                        expert_recompute_pass=index[row['id']]['expert_repeat_pass'],
                        decoder_margin=scored['margin'],decoder_label=scored['label'],
                        correct=float(scored['label']==row['label']),abstain=float(scored['abstain']),
                        answer_correct=float(record['answer']==row['label']))
                    record['e0_checks_pass']=bool(record['identity_pass'] and record['cached_uncached_tokens_equal']
                        and record['generation_complete'] and record['expert_recompute_pass'] and record['decoder_identity_pass'])
                    np.save(folder/f"{row['id']}_hidden.npy",hidden.numpy())
                    np.save(folder/f"{row['id']}_map.npy",mapped)
                    if record['e0_checks_pass']: states[row['id']]=hidden
                else: record['e0_checks_pass']=False
            except torch.cuda.OutOfMemoryError: raise
            except Exception as error: record.update(e0_checks_pass=False,error=f'{type(error).__name__}: {error}')
            results.append(record); write_jsonl(folder/'results.jsonl',results)
            print('NATIVE',number+1,len(rows),row['id'],record.get('e0_checks_pass'),flush=True)
        eligible=[r for r in results if r.get('e0_checks_pass')]
        controls=[]
        for number,row in enumerate(eligible):
            features,patch_hw,native_hw=load_features(row)
            h=states[row['id']]
            _,raw=decoder(h,features,patch_hw,native_hw)
            baseline,baseline_score=score_map(raw,row,calibration['decoder'])
            cases=[('native',0,h,{}),('fixed_mean',0,fixed_mean,{}),('fixed_example',0,fixed_example,{})]
            for seed in range(5):
                donor=donor_for(row,eligible,seed+1009*number)
                cases.append(('cross_image',seed,states[donor['id']],{'donor_id':donor['id']}))
                order=token_derangement(seed)
                cases.append(('token_shuffle',seed,h[list(order)],{'permutation':list(order)}))
            for name,seed,state,details in cases:
                _,raw=decoder(state,features,patch_hw,native_hw)
                mapped,scored=score_map(raw,row,calibration['decoder'])
                controls.append({k:row[k] for k in ('id','family','variant','stratum','label')}|
                    dict(condition=name,seed=seed,**details,margin=scored['margin'],prediction=scored['label'],
                        correct=float(scored['label']==row['label']),abstain=float(scored['abstain']),
                        abs_margin_delta=abs(scored['margin']-baseline_score['margin']),
                        map_rmse=float(np.sqrt(np.mean((mapped-baseline)**2)))))
            write_jsonl(folder/'controls.jsonl',controls)
    summary=dict(attempted=len(results),eligible=len(eligible),failures=len(results)-len(eligible),
        answer_accuracy=cluster_interval(eligible,'answer_correct'),decoder_accuracy=cluster_interval(eligible,'correct'),
        controls=summarize(controls,'condition'),effects={condition:{metric:cluster_interval(
            [r for r in controls if r['condition']==condition],metric) for metric in ('abs_margin_delta','map_rmse')}
            for condition in sorted({r['condition'] for r in controls})},
        scope='Prespecified 40-image native subset only; calibration frozen from previous run.')
    write_json(folder/'summary.json',summary)
    print('SHIFT_NATIVE_COMPLETED',str(out),json.dumps(summary),flush=True)
