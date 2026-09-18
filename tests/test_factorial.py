import numpy as np
import torch

from covt_pilot.decoder import reconstruct
from covt_pilot.experiments import run_e1
from covt_pilot.io import read_jsonl, sha256, write_json, write_jsonl


def test_four_way_factorial_artifacts_and_direction(tmp_path, monkeypatch):
    e0 = tmp_path / "e0"
    e0.mkdir()
    write_json(e0 / "run.json", {"kind": "e0_native", "decoder": {"file_sha256": "fixture"}})
    write_json(e0 / "calibration.json", {"decoder": {"direction": 1, "scale": 1., "tie_margin": .05, "families": ["calibration"]}})
    rows = []
    features = np.tile(np.arange(4, dtype=np.float32), 4).reshape(1, 16, 1)
    features = np.repeat(features, 4, axis=0)
    for name, hidden_sign, feature_sign, label in [("base", -1, 1, "B"), ("depth_swap", 1, -1, "A"), ("nuisance", -1, 1, "B")]:
        artifact = f"{name}.npz"
        np.savez(e0 / artifact, hidden=np.full((4, 1), hidden_sign, dtype=np.float32),
                 features=features * feature_sign, patch_hw=[4, 4], native_hw=[16, 16])
        rows.append({"id": name, "family": "pilot", "split": "pilot", "variant": name,
                     "e0_checks_pass": True, "artifact": artifact, "artifact_sha256": sha256(e0 / artifact),
                     "label": label, "camera": {"size": [16, 16]}, "points": [[3, 8], [12, 8]],
                     "inner_radius": 1, "outer_radius": 2})
    write_jsonl(e0 / "results.jsonl", rows)

    class FixtureDecoder:
        provenance = {"file_sha256": "fixture"}
        def __call__(self, hidden, features, patch_hw, native_hw):
            return reconstruct(hidden.unsqueeze(0), features, patch_hw, native_hw)
    monkeypatch.setattr("covt_pilot.decoder.NativeDepthDecoder.restore", lambda *args: FixtureDecoder())
    result = run_e1(e0, tmp_path / "fixture-decoder", tmp_path / "e1")
    assert result["pairs"] == 2
    assert len(list((tmp_path / "e1").glob("*.npy"))) == 8
    records = {row["change"]: row for row in read_jsonl(tmp_path / "e1/results.jsonl")}
    semantic = records["depth_swap"]
    assert semantic["directed_state_effect_Fa"] > 0
    assert semantic["r00"] == semantic["r11"]
    assert semantic["r10"] == semantic["r01"]
    assert semantic["interaction"] == semantic["state_effect_Fb"] - semantic["state_effect_Fa"]
    assert records["nuisance"]["abs_state_effect_Fa"] == 0


def test_e1_refuses_modified_cached_state(tmp_path, monkeypatch):
    e0 = tmp_path / "e0"
    e0.mkdir()
    write_json(e0 / "run.json", {"kind": "e0_native", "decoder": {"file_sha256": "fixture"}})
    write_json(e0 / "calibration.json", {"decoder": {"families": []}})
    (e0 / "state.npz").write_bytes(b"modified")
    common = {"family": "f", "split": "pilot", "e0_checks_pass": True, "artifact": "state.npz", "artifact_sha256": "old"}
    write_jsonl(e0 / "results.jsonl", [common | {"variant": v} for v in ("base", "depth_swap")])
    monkeypatch.setattr("covt_pilot.decoder.NativeDepthDecoder.restore", lambda *a: type("D", (), {"provenance": {"file_sha256": "fixture"}})())
    import pytest
    with pytest.raises(ValueError, match="modified"):
        run_e1(e0, "unused", tmp_path / "out")
