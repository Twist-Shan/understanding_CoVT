"""Metadata-first audit and selective recovery of official decoder tensors.

No full VLM shard is downloaded by audit or recover. Range requests must be
honoured; a server ignoring Range is rejected before its body is consumed.
"""
import base64
import hashlib
import json
import re
import struct
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .io import new_output, read_json, write_json

MODEL_ID = "Wakals/CoVT-7B-depth"
CODE_REPO = "Wakals/CoVT"
CODE_PATH = "train/src/training/covt_qwen2_5_vl.py"
REQUIRED_KEYS = [f"depth_token_generator.{layer}.{kind}"
                 for layer in (0, 2) for kind in ("weight", "bias")]
MAX_JSON = 16 * 1024 * 1024


def get_bytes(url, *, start=None, end=None, limit=MAX_JSON, timeout=25):
    headers = {"User-Agent": "covt-pilot/0.1", "Accept-Encoding": "identity"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if start is not None:
            expected = f"bytes {start}-{end}/"
            if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                raise RuntimeError("Server ignored or changed byte range; refusing a full weight download.")
        content = response.read(limit + 1)
    if len(content) > limit:
        raise RuntimeError(f"Response exceeds {limit} bytes: {url}")
    if start is not None and len(content) != end - start + 1:
        raise RuntimeError("Incomplete tensor byte range.")
    return content


def get_json(url):
    return json.loads(get_bytes(url))


def resolve_url(model_id, revision, filename):
    return f"https://huggingface.co/{model_id}/resolve/{revision}/{urllib.parse.quote(filename, safe='/')}"


def remote_header(url):
    size = struct.unpack("<Q", get_bytes(url, start=0, end=7, limit=8))[0]
    if size < 2 or size > MAX_JSON:
        raise ValueError("Invalid safetensors header length.")
    header = json.loads(get_bytes(url, start=8, end=7+size, limit=size))
    return header, 8 + size


def inspect_keys(keys):
    keys = set(keys)
    missing = sorted(set(REQUIRED_KEYS) - keys)
    return {"required_keys": REQUIRED_KEYS, "missing_keys": missing,
            "other_depth_keys": sorted(k for k in keys if "depth" in k and k not in REQUIRED_KEYS),
            "status": "tensor_names_present" if not missing else "missing_native_decoder_weights",
            "runtime_verified": False}


def audit(out, model_id=MODEL_ID, revision="main", code_revision="main"):
    out = new_output(out)
    report = {"model_id": model_id, "requested_revision": revision,
              "checked_at": datetime.now(timezone.utc).isoformat(), "errors": [],
              "status": "unverified", "runtime_verified": False}
    try:
        metadata = get_json(f"https://huggingface.co/api/models/{model_id}/revision/{revision}")
        model_revision = metadata["sha"]
        if not re.fullmatch(r"[0-9a-f]{40}", model_revision):
            raise ValueError("Expected an immutable model commit.")
        report["model_revision"] = model_revision
        files = [f["rfilename"] for f in metadata["siblings"]]
        report["files"] = files
        write_json(out / "model_metadata.json", metadata)
        config = get_json(f"https://huggingface.co/{model_id}/raw/{model_revision}/config.json")
        write_json(out / "config.json", config)
        report["architectures"] = config.get("architectures")
        if "model.safetensors.index.json" in files:
            index = get_json(f"https://huggingface.co/{model_id}/raw/{model_revision}/model.safetensors.index.json")
            mapping = index["weight_map"]
            write_json(out / "model.safetensors.index.json", index)
        elif "model.safetensors" in files:
            header, _ = remote_header(resolve_url(model_id, model_revision, "model.safetensors"))
            mapping = {k: "model.safetensors" for k in header if k != "__metadata__"}
            write_json(out / "model.safetensors.header.json", header)
        else:
            raise RuntimeError("No supported safetensors index; decoder availability remains unknown.")
        report.update(inspect_keys(mapping))
        report["decoder_weight_map"] = {k: mapping[k] for k in REQUIRED_KEYS if k in mapping}
    except Exception as error:
        report["errors"].append({"stage": "model_metadata", "message": str(error)})
    try:
        commit = get_json(f"https://api.github.com/repos/{CODE_REPO}/commits/{code_revision}")["sha"]
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("Expected an immutable code commit.")
        source_url = f"https://raw.githubusercontent.com/{CODE_REPO}/{commit}/{CODE_PATH}"
        blob = get_json(f"https://api.github.com/repos/{CODE_REPO}/contents/{CODE_PATH}?ref={commit}")
        source = base64.b64decode(blob["content"])
        (out / "covt_qwen2_5_vl.py").write_bytes(source)
        report["code"] = {"revision": commit, "path": CODE_PATH, "url": source_url,
                          "sha256": hashlib.sha256(source).hexdigest()}
    except Exception as error:
        report["errors"].append({"stage": "code_metadata", "message": str(error)})
    write_json(out / "report.json", report)
    return report


def recover(audit_path, out, local_checkpoint=None):
    report = read_json(audit_path)
    if report.get("status") != "tensor_names_present":
        raise ValueError("Native decoder tensor availability has not passed audit.")
    if local_checkpoint is not None:
        return recover_local(report, local_checkpoint, out)
    out = new_output(out)
    mapping = report["decoder_weight_map"]
    raw_tensors, specs, records = {}, {}, {}
    for filename in sorted(set(mapping.values())):
        url = resolve_url(report["model_id"], report["model_revision"], filename)
        header, offset = remote_header(url)
        for key in REQUIRED_KEYS:
            if mapping[key] != filename:
                continue
            spec = header[key]
            lo, hi = spec["data_offsets"]
            if hi <= lo or hi-lo > 64 * 1024 * 1024:
                raise ValueError(f"Unexpected decoder tensor size: {key}")
            print(f"Recovering {key}: {hi-lo:,} bytes", flush=True)
            content = get_bytes(url, start=offset+lo, end=offset+hi-1, limit=hi-lo)
            raw_tensors[key], specs[key] = content, spec
            records[key] = {"shard": filename, "shape": spec["shape"], "dtype": spec["dtype"],
                            "sha256": hashlib.sha256(content).hexdigest()}
    # Safetensors is a JSON header + contiguous tensor bytes. Preserve tensor dtypes.
    cursor, header = 0, {"__metadata__": {"source_model": report["model_id"], "revision": report["model_revision"]}}
    for key in REQUIRED_KEYS:
        content, spec = raw_tensors[key], specs[key]
        header[key] = {"dtype": spec["dtype"], "shape": spec["shape"], "data_offsets": [cursor, cursor+len(content)]}
        cursor += len(content)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    target = out / "decoder.safetensors"
    with target.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for key in REQUIRED_KEYS:
            stream.write(raw_tensors[key])
    from .io import sha256
    provenance = {"model_id": report["model_id"], "model_revision": report["model_revision"],
                  "file_sha256": sha256(target), "tensors": records, "code": report.get("code"),
                  "runtime_verified": False}
    write_json(out / "provenance.json", provenance)
    return provenance


def recover_local(report, checkpoint, out):
    """Read selected tensors from an already downloaded HF snapshot, without loading the VLM."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    from .io import sha256
    checkpoint = Path(checkpoint)
    if checkpoint.name != report["model_revision"]:
        raise ValueError("Use the Hugging Face snapshots/<audited-model-revision> directory to avoid version mixing.")
    mapping = report["decoder_weight_map"]
    tensors, records = {}, {}
    for filename in sorted(set(mapping.values())):
        with safe_open(str(checkpoint / filename), framework="pt", device="cpu") as archive:
            for key in REQUIRED_KEYS:
                if mapping[key] == filename:
                    tensors[key] = archive.get_tensor(key).contiguous()
                    records[key] = {"shard": filename, "shape": list(tensors[key].shape), "dtype": str(tensors[key].dtype)}
    out = new_output(out)
    target = out / "decoder.safetensors"
    save_file(tensors, str(target), metadata={"source_model": report["model_id"], "revision": report["model_revision"]})
    provenance = {"model_id": report["model_id"], "model_revision": report["model_revision"],
                  "file_sha256": sha256(target), "tensors": records, "code": report.get("code"),
                  "transport": "user-provided local HF snapshot", "runtime_verified": False}
    write_json(out / "provenance.json", provenance)
    return provenance


def complete_cached_audit(directory, source_api_json, code_revision):
    """Complete an interrupted audit from already downloaded, pinned metadata.

    Required files: model_metadata.json, config.json, model.safetensors.index.json.
    The API source response must point to the explicit code revision; its Git blob
    digest is verified. No inference about trained quality is made from key names.
    """
    directory = Path(directory)
    metadata = read_json(directory / "model_metadata.json")
    config = read_json(directory / "config.json")
    index = read_json(directory / "model.safetensors.index.json")
    blob = read_json(source_api_json)
    if not re.fullmatch(r"[0-9a-f]{40}", code_revision) or not re.fullmatch(r"[0-9a-f]{40}", metadata["sha"]):
        raise ValueError("Immutable code and model revisions are required.")
    if f"/{code_revision}/" not in blob["download_url"] or blob["path"] != CODE_PATH:
        raise ValueError("Source API response does not match the requested revision/path.")
    source = base64.b64decode(blob["content"])
    digest = hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest()
    if digest != blob["sha"]:
        raise ValueError("Upstream source Git blob checksum mismatch.")
    (directory / "covt_qwen2_5_vl.py").write_bytes(source)
    prior = read_json(directory / "report.json") if (directory / "report.json").exists() else {}
    report = {"model_id": metadata["id"], "model_revision": metadata["sha"],
              "checked_at": datetime.now(timezone.utc).isoformat(), "architectures": config.get("architectures"),
              "files": [r["rfilename"] for r in metadata["siblings"]],
              **inspect_keys(index["weight_map"]), "errors": [],
              "prior_download_errors": prior.get("errors", []),
              "audit_mode": "cached pinned metadata; source Git blob verified",
              "decoder_weight_map": {k: index["weight_map"][k] for k in REQUIRED_KEYS if k in index["weight_map"]},
              "code": {"revision": code_revision, "path": CODE_PATH, "url": blob["download_url"],
                       "git_blob": digest, "sha256": hashlib.sha256(source).hexdigest()}}
    write_json(directory / "report.json", report)
    return report
