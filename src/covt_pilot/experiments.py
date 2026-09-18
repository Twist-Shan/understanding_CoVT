"""E0 restoration checks and E1 visualization-only factorial experiment."""
import importlib.metadata
from pathlib import Path

import numpy as np
from PIL import Image

from .io import new_output, read_json, read_jsonl, sha256, write_json, write_jsonl
from .metrics import calibrate, cluster_interval, joint_table, raw_difference, score_difference


def run_e0(manifest, audit_path, decoder_dir, covt_repo, expert_checkpoint, out,
           device="cuda", expert_device="cpu", dtype="bfloat16", max_new_tokens=192,
           attention="eager", max_gpu_memory=None, atol=0.02, rtol=0.01):
    import torch
    from .decoder import NativeDepthDecoder, map_to_input
    from .expert import DepthExpert
    from .model import CoVTModel
    from .reference import verify_reference
    rows = read_jsonl(manifest)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate sample IDs in manifest.")
    audit = read_json(audit_path)
    if audit.get("status") != "tensor_names_present" or not audit.get("code"):
        raise ValueError("A successful model AND source audit is required before E0.")
    parity = verify_reference(Path(audit_path).parent / "covt_qwen2_5_vl.py", audit["code"]["sha256"])
    decoder = NativeDepthDecoder.restore(decoder_dir, expert_device)
    if decoder.provenance["model_revision"] != audit["model_revision"] or decoder.provenance["model_id"] != audit["model_id"]:
        raise ValueError("Decoder and language checkpoint provenance differ.")
    out = new_output(out)
    manifest = Path(manifest).resolve()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    settings = {"kind": "e0_native", "manifest": str(manifest), "manifest_sha256": sha256(manifest),
                "audit": audit, "decoder": decoder.provenance, "reference_parity": parity,
                "device": device, "expert_device": expert_device, "dtype": dtype,
                "attention": attention, "max_new_tokens": max_new_tokens, "atol": atol, "rtol": rtol,
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "Pillow")},
                "shared_ancestor_verified": False, "scope": "E0 only; no answer-path causal claim"}
    write_json(out / "run.json", settings)
    expert = DepthExpert(covt_repo, expert_checkpoint, audit["code"]["revision"], expert_device)
    model = CoVTModel(audit["model_id"], audit["model_revision"], device, dtype, attention, max_gpu_memory)
    settings.update(expert=expert.provenance, model=model.provenance)
    write_json(out / "run.json", settings)
    results = []
    with torch.inference_mode():
        for row in rows:
            record = dict(row)
            try:
                image_path = manifest.parent / row["image"]
                record["image_sha256"] = sha256(image_path)
                with Image.open(image_path) as opened:
                    image = opened.convert("RGB")
                prediction, hidden = model.run(image, row["question"], max_new_tokens, atol, rtol)
                record.update(prediction)
                if hidden is not None:
                    features, expert_depth, patch_hw, native_hw = expert.encode(image)
                    fresh_features, fresh_depth, _, _ = expert.encode(image)
                    feature_error = max(float((a-b).abs().max()) for a, b in zip(features, fresh_features))
                    expert_pass = all(torch.allclose(a, b, atol=atol, rtol=rtol) for a, b in zip(features, fresh_features))
                    expert_pass = expert_pass and torch.allclose(expert_depth, fresh_depth, atol=atol, rtol=rtol)
                    _, raw_depth = decoder(hidden, features, patch_hw, native_hw)
                    _, replay_depth = decoder(hidden.clone(), [f.clone() for f in features], patch_hw, native_hw)
                    input_hw = (image.height, image.width)
                    decoded = map_to_input(raw_depth, input_hw).cpu().numpy()
                    expert_map = map_to_input(expert_depth, input_hw).cpu().numpy()
                    record.update(raw_difference=raw_difference(decoded, row), expert_difference=raw_difference(expert_map, row),
                                  expert_recompute_max_abs=feature_error, expert_recompute_pass=bool(expert_pass),
                                  decoder_identity_max_abs=float((raw_depth-replay_depth).abs().max()),
                                  decoder_identity_pass=bool(torch.allclose(raw_depth, replay_depth, atol=atol, rtol=rtol)))
                    record["e0_checks_pass"] = bool(record["identity_pass"] and record["cached_uncached_tokens_equal"]
                        and record["generation_complete"] and record["expert_recompute_pass"] and record["decoder_identity_pass"])
                    artifact = f"{row['id']}.npz"
                    np.savez_compressed(out / artifact, hidden=hidden.numpy(),
                                        features=np.stack([f[0].float().cpu().numpy() for f in features]),
                                        patch_hw=patch_hw, native_hw=native_hw,
                                        raw_native_map=raw_depth[0, 0].float().cpu().numpy(),
                                        expert_native_map=expert_depth[0, 0].float().cpu().numpy(),
                                        decoded_map=decoded, expert_map=expert_map)
                    record["artifact"] = artifact
                    record["artifact_sha256"] = sha256(out / artifact)
            except torch.cuda.OutOfMemoryError as error:
                record.update(status="fatal_out_of_memory", error=str(error), e0_checks_pass=False)
                results.append(record)
                write_jsonl(out / "results.jsonl", results)
                raise RuntimeError("E0 stopped on GPU OOM; reduce memory use or choose a larger GPU.") from error
            except Exception as error:
                # Keep failures in the denominator; preserve completed records.
                record.update(status="error", error=f"{type(error).__name__}: {error}", e0_checks_pass=False)
            results.append(record)
            write_jsonl(out / "results.jsonl", results)
            print(f"{row['id']}: {record['status']}", flush=True)
    return summarize_e0(out)


def summarize_e0(out):
    out = Path(out)
    rows = read_jsonl(out / "results.jsonl")
    eligible = [r for r in rows if r.get("e0_checks_pass")]
    calibration_rows = [r for r in eligible if r["split"] == "calibration"]
    summary = {"attempted": len(rows), "eligible": len(eligible), "failures_or_ineligible": len(rows)-len(eligible),
               "status": "incomplete", "shared_ancestor_verified": False}
    try:
        decoder_cal = calibrate([r["raw_difference"] for r in calibration_rows], calibration_rows)
        expert_cal = calibrate([r["expert_difference"] for r in calibration_rows], calibration_rows)
    except ValueError as error:
        summary["calibration_error"] = str(error)
        write_json(out / "summary.json", summary)
        return summary
    write_json(out / "calibration.json", {"decoder": decoder_cal, "expert": expert_cal})
    scored = []
    for row in eligible:
        if row["split"] == "calibration":
            continue
        if row["family"] in decoder_cal["families"]:
            raise ValueError("Calibration and evaluation families overlap.")
        d = score_difference(row["raw_difference"], decoder_cal)
        e = score_difference(row["expert_difference"], expert_cal)
        scored.append(row | {"decoder_label": d["label"], "decoder_margin": d["margin"],
                             "expert_label": e["label"], "expert_margin": e["margin"],
                             "decoder_correct": float(d["label"] == row["label"]),
                             "expert_correct": float(e["label"] == row["label"]),
                             "answer_correct": float(row.get("answer") == row["label"])})
    summary.update(status="scored" if scored else "no_evaluation_examples", joint_errors=joint_table(scored),
                   decoder_accuracy=cluster_interval(scored, "decoder_correct"),
                   expert_accuracy=cluster_interval(scored, "expert_correct"),
                   answer_accuracy=cluster_interval(scored, "answer_correct"),
                   evaluation_attempted=sum(r["split"] != "calibration" for r in rows),
                   scope="Accuracy is conditional on E0 eligibility; failures and invalid answers are separately retained.")
    write_jsonl(out / "scored.jsonl", scored)
    write_json(out / "summary.json", summary)
    return summary


def run_e1(e0_dir, decoder_dir, out, device="cpu"):
    import torch
    from .decoder import NativeDepthDecoder, map_to_input
    e0_dir = Path(e0_dir)
    run = read_json(e0_dir / "run.json")
    if run["kind"] != "e0_native":
        raise ValueError("E1 requires native E0 caches, not smoke/oracle outputs.")
    decoder = NativeDepthDecoder.restore(decoder_dir, device)
    if decoder.provenance["file_sha256"] != run["decoder"]["file_sha256"]:
        raise ValueError("E1 decoder differs from E0.")
    calibration = read_json(e0_dir / "calibration.json")["decoder"]
    rows = read_jsonl(e0_dir / "results.jsonl")
    groups = {}
    for row in rows:
        if row["split"] == "pilot":
            if row["family"] in calibration["families"]:
                raise ValueError("Calibration family cannot enter E1 evaluation.")
            groups.setdefault(row["family"], {})[row["variant"]] = row
    out = new_output(out)
    write_json(out / "run.json", {"kind": "e1_visualization_only", "e0_run": str(e0_dir.resolve()),
               "e0_run_sha256": sha256(e0_dir / "run.json"), "calibration": calibration,
               "scope": "Terminal conditional decoder factorial; no answer recomputation or shared-state claim."})
    results, exclusions = [], []
    for family, variants in groups.items():
        for change in ("depth_swap", "nuisance"):
            a, b = variants.get("base"), variants.get(change)
            if not a or not b or not a.get("e0_checks_pass") or not b.get("e0_checks_pass"):
                exclusions.append({"family": family, "change": change, "reason": "missing or failed E0 pair"})
                continue
            cache = []
            for row in (a, b):
                path = e0_dir / row["artifact"]
                if sha256(path) != row["artifact_sha256"]:
                    raise ValueError("Cached artifact was modified after E0.")
                with np.load(path, allow_pickle=False) as archive:
                    cache.append({k: archive[k] for k in archive.files})
            if not np.array_equal(cache[0]["patch_hw"], cache[1]["patch_hw"]):
                raise ValueError("Factorial inputs have incompatible expert grids.")
            margins = {}
            with torch.inference_mode():
                for u in range(2):
                    for v in range(2):
                        hidden = torch.from_numpy(cache[u]["hidden"]).to(device)
                        features = [torch.from_numpy(f).unsqueeze(0).to(device) for f in cache[v]["features"]]
                        _, raw = decoder(hidden, features, cache[v]["patch_hw"], cache[v]["native_hw"])
                        recipient = (a, b)[v]
                        evaluated = map_to_input(raw, tuple(reversed(recipient["camera"]["size"]))).cpu().numpy()
                        value = score_difference(raw_difference(evaluated, recipient), calibration)
                        margins[f"r{u}{v}"] = value["margin"]
                        np.save(out / f"{family}_{change}_H{u}_F{v}.npy", raw[0, 0].float().cpu().numpy())
            state_a = margins["r10"]-margins["r00"]
            state_b = margins["r11"]-margins["r01"]
            orientation = 1 if b["label"] == "A" else -1
            results.append({"family": family, "change": change, **margins,
                            "state_effect_Fa": state_a, "state_effect_Fb": state_b,
                            "abs_state_effect_Fa": abs(state_a), "abs_state_effect_Fb": abs(state_b),
                            "directed_state_effect_Fa": orientation*state_a,
                            "directed_state_effect_Fb": orientation*state_b,
                            "expert_effect_Ha": margins["r01"]-margins["r00"],
                            "expert_effect_Hb": margins["r11"]-margins["r10"],
                            "interaction": state_b-state_a})
    write_jsonl(out / "results.jsonl", results)
    write_jsonl(out / "exclusions.jsonl", exclusions)
    summary = {"pairs": len(results), "excluded_pairs": len(exclusions),
               "scope": "Decoder branch only. Effects are not answer-use evidence.",
               "conditions": {change: {key: cluster_interval([r for r in results if r["change"] == change], key)
                    for key in (("directed_state_effect_Fa", "directed_state_effect_Fb", "abs_state_effect_Fa", "interaction")
                                if change == "depth_swap" else ("abs_state_effect_Fa", "abs_state_effect_Fb", "interaction"))}
                    for change in ("depth_swap", "nuisance")}}
    write_json(out / "summary.json", summary)
    return summary
