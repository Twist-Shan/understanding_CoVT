"""Verify compact archived evidence without executing any experiment."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "results/manifest.json").read_text(encoding="utf8"))
for entry in manifest["files"]:
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root / "results"):
        raise ValueError("Evidence path outside results")
    data = path.read_bytes()
    if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"Checksum mismatch: {entry['path']}")
print(f"Verified {len(manifest['files'])} archived evidence files.")
