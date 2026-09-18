"""Command-line entrypoints. Heavy model dependencies are loaded on demand."""
import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

from .io import new_output, read_jsonl, write_json, write_jsonl


def doctor():
    report = {"python": sys.version, "packages": {}}
    for name in ("numpy", "Pillow", "torch", "torchvision", "transformers", "safetensors", "accelerate"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    try:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        report["gpus"] = [{"name": torch.cuda.get_device_name(i), "memory_gib": torch.cuda.get_device_properties(i).total_memory / 2**30}
                          for i in range(torch.cuda.device_count())]
    except Exception as error:
        report["torch_error"] = str(error)
    report["model_runtime_version_match"] = report["packages"]["transformers"] == "4.50.1"
    report["note"] = "Full checkpoint/E0 inference is not validated by doctor; original runtime pin is transformers 4.50.1."
    return report


def smoke(out):
    """CPU integration exercise with GEOMETRIC ORACLE maps, never model results."""
    import numpy as np
    from .metrics import calibrate, joint_table, raw_difference, score_difference
    from .scenes import generate
    out = new_output(out)
    generated = generate(out / "scenes", families=5, seed=17)
    rows = read_jsonl(generated["manifest"])
    differences = []
    for row in rows:
        with np.load(out / "scenes" / row["geometry"], allow_pickle=False) as archive:
            differences.append(raw_difference(archive["depth"], row))
    selected = [(d, r) for d, r in zip(differences, rows) if r["split"] == "calibration"]
    frozen = calibrate([d for d, _ in selected], [r for _, r in selected])
    scored = [r | {"decoder_label": score_difference(d, frozen)["label"], "answer": r["label"]}
              for d, r in zip(differences, rows) if r["split"] == "pilot"]
    report = {"kind": "oracle_smoke_only", "native_model_executed": False,
              "note": "Both maps and answers come from the renderer. These are software checks, not research evidence.",
              "calibration": frozen, "geometry_checks": joint_table(scored)}
    write_json(out / "report.json", report)
    return report


def parser():
    root = argparse.ArgumentParser(description="CoVT native depth E0/E1 pilot")
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser("doctor", help="Inspect local runtime without downloading models")
    p.add_argument("--out")
    p = commands.add_parser("smoke", help="CPU pipeline check using explicitly labelled oracle data")
    p.add_argument("--out", required=True)
    p = commands.add_parser("make-scenes", help="Render pilot scene families with exact camera-axis depth")
    p.add_argument("--out", required=True)
    p.add_argument("--families", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=int, default=336)
    p = commands.add_parser("audit", help="Fetch metadata/source only and check required tensor names")
    p.add_argument("--out", required=True)
    p.add_argument("--model-id", default="Wakals/CoVT-7B-depth")
    p.add_argument("--revision", default="main")
    p.add_argument("--code-revision", default="main")
    p = commands.add_parser("audit-cache", help="Finish a partial audit from pinned local metadata")
    p.add_argument("--directory", required=True)
    p.add_argument("--source-api-json", required=True)
    p.add_argument("--code-revision", required=True)
    p = commands.add_parser("recover-decoder", help="Download only the four official generator tensors via HTTP ranges")
    p.add_argument("--audit", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--local-checkpoint", help="Optional existing HF snapshots/<revision> directory; avoids network")
    p = commands.add_parser("verify-reference", help="Compare reconstruction with the audited upstream class")
    p.add_argument("--source", required=True)
    p.add_argument("--out")
    p = commands.add_parser("e0", help="Native generation, replay, decoder recovery and calibration")
    p.add_argument("--manifest", required=True)
    p.add_argument("--audit", dest="audit_path", required=True)
    p.add_argument("--decoder-dir", required=True)
    p.add_argument("--covt-repo", required=True)
    p.add_argument("--expert-checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--expert-device", default="cpu")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    p.add_argument("--attention", choices=["eager", "sdpa"], default="eager")
    p.add_argument("--max-gpu-memory")
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--atol", type=float, default=0.02)
    p.add_argument("--rtol", type=float, default=0.01)
    p = commands.add_parser("e1", help="Four-combination visualization-branch factorial on E0 caches")
    p.add_argument("--e0-dir", required=True)
    p.add_argument("--decoder-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    return root


def main(argv=None):
    args = vars(parser().parse_args(argv))
    command = args.pop("command")
    try:
        if command == "doctor":
            result = doctor()
            if args["out"]:
                write_json(args["out"], result)
        elif command == "smoke":
            result = smoke(**args)
        elif command == "make-scenes":
            from .scenes import generate
            result = generate(**args)
        elif command == "audit":
            from .resources import audit
            result = audit(**args)
        elif command == "recover-decoder":
            from .resources import recover
            result = recover(args["audit"], args["out"], args["local_checkpoint"])
        elif command == "audit-cache":
            from .resources import complete_cached_audit
            result = complete_cached_audit(**args)
        elif command == "verify-reference":
            from .reference import verify_reference
            result = verify_reference(args["source"])
            if args["out"]:
                write_json(args["out"], result)
        elif command == "e0":
            from .experiments import run_e0
            result = run_e0(**args)
        elif command == "e1":
            from .experiments import run_e1
            result = run_e1(**args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if command in ("audit", "audit-cache") and (result["status"] != "tensor_names_present" or result["errors"]):
            return 2
        if command == "e0" and result["status"] != "scored":
            return 2
        if command == "e1" and result["pairs"] == 0:
            return 2
        return 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
