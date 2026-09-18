"""Exercise artifact/calibration plumbing with explicit test doubles, no VLM."""
import numpy as np
import torch

from covt_pilot.decoder import reconstruct
from covt_pilot.experiments import run_e0
from covt_pilot.io import read_json, read_jsonl, write_json
from covt_pilot.scenes import generate


def test_e0_artifact_pipeline_without_external_models(tmp_path, monkeypatch):
    generate(tmp_path / "data", families=2, seed=88)
    manifest = tmp_path / "data/manifest.jsonl"
    rows = read_jsonl(manifest)
    state = {"index": 0}
    provenance = {"model_id": "fixture", "model_revision": "a"*40, "file_sha256": "fixture"}
    audit = {"status": "tensor_names_present", "model_id": "fixture", "model_revision": "a"*40,
             "code": {"sha256": "fixture", "revision": "b"*40}}
    write_json(tmp_path / "audit/report.json", audit)

    class Model:
        provenance = {"test_double": True}
        def __init__(self, *args): pass
        def run(self, *args):
            state["row"] = rows[state["index"]]
            state["index"] += 1
            return {"answer": state["row"]["label"], "status": "decoded_state_available",
                    "identity_pass": True, "cached_uncached_tokens_equal": True,
                    "generation_complete": True}, torch.ones(4, 1)

    class Expert:
        provenance = {"test_double": True}
        def __init__(self, *args): pass
        def encode(self, image):
            row = state["row"]
            spatial_sign = np.sign(row["points"][1][0] - row["points"][0][0])
            label_sign = 1 if row["label"] == "A" else -1
            field = torch.arange(4, dtype=torch.float32).repeat(4).view(1, 16, 1) * float(spatial_sign*label_sign)
            features = [field.clone() for _ in range(4)]
            _, depth = reconstruct(torch.ones(1, 4, 1), features, (4, 4), (256, 256))
            return features, depth, (4, 4), (256, 256)

    class Decoder:
        def __init__(self): self.provenance = provenance
        def __call__(self, hidden, features, patch_hw, native_hw):
            return reconstruct(hidden.unsqueeze(0), features, patch_hw, native_hw)

    monkeypatch.setattr("covt_pilot.model.CoVTModel", Model)
    monkeypatch.setattr("covt_pilot.expert.DepthExpert", Expert)
    monkeypatch.setattr("covt_pilot.decoder.NativeDepthDecoder.restore", lambda *a: Decoder())
    monkeypatch.setattr("covt_pilot.reference.verify_reference", lambda *a: {"passed": True, "test_double": True})
    monkeypatch.setattr("covt_pilot.experiments.importlib.metadata.version", lambda name: "test-double")
    summary = run_e0(manifest, tmp_path / "audit/report.json", "unused", "unused", "unused", tmp_path / "e0", device="cpu")
    assert summary["status"] == "scored"
    assert summary["attempted"] == summary["eligible"] == 8
    assert summary["shared_ancestor_verified"] is False
    assert summary["joint_errors"]["counts"] == {"decoder_correct__answer_correct": 4}
    assert len(list((tmp_path / "e0").glob("*.npz"))) == 8
    assert read_json(tmp_path / "e0/calibration.json")["decoder"]["families"] == ["scene_00000"]
