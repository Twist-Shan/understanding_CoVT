"""Terminal depth-decoder controls with fixed calibration and explicit donors.

No answer rollout is performed. Zero hidden input still passes through MLP biases;
zero projected tokens is a separate flat-map negative control.
"""
import itertools
from pathlib import Path

import numpy as np
import torch

from .decoder import NativeDepthDecoder, map_to_input, reconstruct
from .io import new_output, read_json, read_jsonl, sha256, write_json, write_jsonl
from .metrics import cluster_interval, raw_difference, score_difference


def token_derangement(seed):
    permutations = [p for p in itertools.permutations(range(4)) if all(i != j for i, j in enumerate(p))]
    return permutations[int(np.random.default_rng(seed).integers(len(permutations)))]


def donor_for(row, candidates, seed):
    choices = sorted((r for r in candidates if r['family'] != row['family']), key=lambda r: r['id'])
    if not choices:
        raise ValueError('Cross-image control requires an eligible donor from another family.')
    return choices[int(np.random.default_rng(seed).integers(len(choices)))]


def run_controls(e0_dir, decoder_dir, out, device='cuda', repeats=5):
    if repeats < 1:
        raise ValueError('At least one shuffle repeat is required.')
    e0_dir=Path(e0_dir)
    original=read_json(e0_dir/'run.json')
    if original['kind'] != 'e0_native':
        raise ValueError('Controls require native E0 results.')
    calibration=read_json(e0_dir/'calibration.json')['decoder']
    all_rows=read_jsonl(e0_dir/'results.jsonl')
    eligible=[r for r in all_rows if r.get('e0_checks_pass')]
    calibrators=[r for r in eligible if r['split']=='calibration']
    targets=[r for r in eligible if r['split']=='pilot']
    if not calibrators or not targets:
        raise ValueError('Eligible calibration and pilot rows are both required.')
    if any(r['family'] in calibration['families'] for r in targets):
        raise ValueError('Calibration families cannot enter evaluation.')
    hidden={}
    for row in eligible:
        path=e0_dir/row['artifact']
        if sha256(path) != row['artifact_sha256']:
            raise ValueError('Modified E0 artifact.')
        with np.load(path,allow_pickle=False) as z:
            hidden[row['id']]=torch.from_numpy(z['hidden'].copy()).float()
    fixed_mean=torch.stack([hidden[r['id']] for r in calibrators]).mean(0)
    fixed_first=hidden[calibrators[0]['id']]
    out=new_output(out)
    write_json(out/'run.json',{'kind':'terminal_state_controls','e0_dir':str(e0_dir.resolve()),
        'e0_run_sha256':sha256(e0_dir/'run.json'),'calibration':calibration,
        'fixed_mean_ids':[r['id'] for r in calibrators],'fixed_first_id':calibrators[0]['id'],
        'repeats':repeats,'device':device,'evaluation_attempted':sum(r['split']=='pilot' for r in all_rows),
        'evaluation_eligible':len(targets),'scope':'Decoder-input interventions only; no answer-use inference.',
        'precision_scope':'native checkpoint dtype and FP32 decoder; E0 hidden/expert caches are unchanged.',
        'aggregation':'Equal-family mean; repeated seeds are averaged within family, never counted as new families.'})
    results=[]
    exclusions=[]
    for precision in ('native','float32'):
        decoder=NativeDepthDecoder.restore(decoder_dir,device)
        if decoder.provenance['file_sha256'] != original['decoder']['file_sha256']:
            raise ValueError('Decoder differs from source E0.')
        if precision=='float32':
            decoder.float()
        with torch.inference_mode():
            for index,row in enumerate(targets):
                with np.load(e0_dir/row['artifact'],allow_pickle=False) as z:
                    features=[torch.from_numpy(f.copy()).unsqueeze(0).to(device) for f in z['features']]
                    patch_hw,native_hw=z['patch_hw'].tolist(),z['native_hw'].tolist()
                h=hidden[row['id']]
                _,base=decoder(h,features,patch_hw,native_hw)
                baseline=map_to_input(base,tuple(reversed(row['camera']['size']))).cpu().numpy()
                base_score=score_difference(raw_difference(baseline,row),calibration)
                cases=[('native_state',0,h,{}),('fixed_calibration_mean',0,fixed_mean,{}),
                    ('fixed_calibration_example',0,fixed_first,{}),('zero_hidden',0,torch.zeros_like(h),{}),
                    ('zero_projected_tokens',0,None,{})]
                for seed in range(repeats):
                    permutation=token_derangement(seed)
                    cases.append(('token_shuffle',seed,h[list(permutation)],{'permutation':list(permutation)}))
                    try:
                        donor=donor_for(row,targets,seed+1009*index)
                        cases.append(('cross_image_state',seed,hidden[donor['id']],{'donor_id':donor['id']}))
                    except ValueError as error:
                        exclusions.append({'id':row['id'],'precision':precision,'seed':seed,
                            'condition':'cross_image_state','reason':str(error)})
                for condition,seed,state,details in cases:
                    if state is None:
                        _,raw=reconstruct(torch.zeros(1,4,features[0].shape[-1],device=device),features,patch_hw,native_hw)
                    else:
                        _,raw=decoder(state,features,patch_hw,native_hw)
                    evaluated=map_to_input(raw,baseline.shape).cpu().numpy()
                    scored=score_difference(raw_difference(evaluated,row),calibration)
                    record={'id':row['id'],'family':row['family'],'variant':row['variant'],'label':row['label'],
                        'precision':precision,'condition':condition,'seed':seed,**details,
                        'decoder_label':scored['label'],'margin':scored['margin'],
                        'baseline_margin':base_score['margin'],'baseline_label':base_score['label'],
                        'correct':float(scored['label']==row['label']),'abstain':float(scored['abstain']),
                        'label_agreement':float(scored['label']==base_score['label']),
                        'abs_margin_delta':abs(scored['margin']-base_score['margin']),
                        'map_rmse':float(np.sqrt(np.mean((evaluated-baseline)**2)))}
                    results.append(record)
                    if seed==0:
                        np.save(out/f"{row['id']}_{precision}_{condition}.npy",evaluated)
                write_jsonl(out/'results.jsonl',results)
                print(f"CONTROL {precision} {row['id']}",flush=True)
        del decoder
    summary={'evaluation_eligible':len(targets),'evaluation_families':len({r['family'] for r in targets}),
        'excluded_controls':len(exclusions),'scope':'Decoder branch only; no equivalence or answer-use claim.',
        'conditions':{precision:{condition:{metric:cluster_interval(
            [r for r in results if r['precision']==precision and r['condition']==condition],metric)
            for metric in ('correct','abstain','label_agreement','abs_margin_delta','map_rmse')}
            for condition in sorted({r['condition'] for r in results})}
            for precision in ('native','float32')}}
    write_jsonl(out/'exclusions.jsonl',exclusions)
    write_json(out/'summary.json',summary)
    return summary
