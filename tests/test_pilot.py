import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from covt_pilot.cli import main
from covt_pilot.decoder import NativeDepthDecoder, reconstruct
from covt_pilot.io import read_json, read_jsonl, write_json
from covt_pilot.metrics import calibrate, cluster_interval, raw_difference, score_difference
from covt_pilot.model import depth_positions, extract_answer, replay
from covt_pilot.resources import REQUIRED_KEYS, get_bytes, inspect_keys, recover
from covt_pilot.scenes import generate


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("scenes") / "data"
    generate(out, families=5, seed=33)
    return out, read_jsonl(out / "manifest.jsonl")


def test_scene_counterfactuals_and_splits(dataset):
    root, rows = dataset
    for family in {r["family"] for r in rows}:
        variants = {r["variant"]: r for r in rows if r["family"] == family}
        assert len({r["split"] for r in variants.values()}) == 1
        assert variants["base"]["label"] != variants["depth_swap"]["label"]
        assert variants["base"]["label"] != variants["label_swap"]["label"]
        assert variants["base"]["label"] == variants["nuisance"]["label"]
        for row in variants.values():
            with np.load(root / row["geometry"]) as archive:
                diff = raw_difference(archive["depth"], row)
                assert (diff > 0) == (row["label"] == "A")
        with np.load(root / variants["base"]["geometry"]) as a, np.load(root / variants["nuisance"]["geometry"]) as b:
            np.testing.assert_array_equal(a["depth"], b["depth"])


def test_cannot_overwrite_dataset(dataset):
    with pytest.raises(ValueError, match="not empty"):
        generate(dataset[0])


def test_inverse_depth_calibration_and_abstention():
    rows = [{"split": "calibration", "label": label, "family": str(i)} for i, label in enumerate(["A", "B"])]
    calibration = calibrate([-2, 4], rows)
    assert calibration["direction"] == -1
    assert score_difference(-1, calibration)["label"] == "A"
    assert score_difference(0, calibration)["abstain"]
    with pytest.raises(ValueError, match="Flat"):
        calibrate([0, 0], rows)
    with pytest.raises(ValueError, match="calibration-only"):
        calibrate([-2, 4], [r | {"split": "pilot"} for r in rows])


def test_correlated_repeats_do_not_inflate_family_count():
    rows = [{"family": "a", "value": 1.0}] * 100
    assert cluster_interval(rows, "value") == {"mean": 1.0, "ci95": None, "families": 1}


def test_decoder_keeps_signed_scale_and_token_layer_pairing():
    tokens = torch.ones(1, 4, 3)
    features = [torch.full((1, 6, 3), float(i-2)) for i in range(4)]
    individual, average = reconstruct(tokens, features, (2, 3), (7, 11))
    assert individual.shape == (1, 4, 7, 11)
    torch.testing.assert_close(individual[0, 0], torch.full((7, 11), -6.0))
    torch.testing.assert_close(average, torch.full((1, 1, 7, 11), -1.5))
    _, doubled = reconstruct(tokens * 2, features, (2, 3), (7, 11))
    torch.testing.assert_close(doubled, average * 2)


def test_decoder_rejects_wrong_grid_and_unrestored_weights():
    with pytest.raises(ValueError, match="grid"):
        reconstruct(torch.ones(1, 4, 3), [torch.ones(1, 5, 3)]*4, (2, 3), (9, 9))
    model = NativeDepthDecoder(hidden_size=8, expert_size=3)
    with pytest.raises(RuntimeError, match="not been restored"):
        model(torch.ones(4, 8), [torch.ones(1, 6, 3)]*4, (2, 3), (9, 9))


def test_availability_requires_active_generator_not_early_projection():
    assert inspect_keys(["depth_projection.weight"])["status"] == "missing_native_decoder_weights"
    assert inspect_keys(REQUIRED_KEYS)["status"] == "tensor_names_present"


def test_range_ignored_never_reads_weight_body(monkeypatch):
    class Response:
        status = 200
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, count): pytest.fail("Full shard body must never be read")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response())
    with pytest.raises(RuntimeError, match="refusing"):
        get_bytes("https://example.test/weights", start=0, end=7, limit=8)


def test_selective_recovery_preserves_tensor_bytes(tmp_path, monkeypatch):
    from safetensors.torch import load_file
    from covt_pilot import resources
    raw = np.arange(8, dtype=np.float32).tobytes()
    header = {key: {"dtype": "F32", "shape": [2], "data_offsets": [i*8, (i+1)*8]}
              for i, key in enumerate(REQUIRED_KEYS)}
    monkeypatch.setattr(resources, "remote_header", lambda url: (header, 256))
    monkeypatch.setattr(resources, "get_bytes", lambda url, start, end, limit: raw[start-256:end-255])
    audit = {"status": "tensor_names_present", "model_id": "fixture", "model_revision": "a"*40,
             "decoder_weight_map": {k: "part.safetensors" for k in REQUIRED_KEYS}}
    write_json(tmp_path / "audit.json", audit)
    recover(tmp_path / "audit.json", tmp_path / "decoder")
    state = load_file(str(tmp_path / "decoder/decoder.safetensors"))
    for index, key in enumerate(REQUIRED_KEYS):
        torch.testing.assert_close(state[key], torch.tensor([index*2, index*2+1], dtype=torch.float32))


def test_local_recovery_checks_revision_and_extracts_only_decoder(tmp_path):
    from safetensors.torch import save_file, load_file
    from covt_pilot.resources import recover_local
    revision = "b"*40
    checkpoint = tmp_path / revision
    checkpoint.mkdir()
    tensors = {k: torch.arange(3, dtype=torch.float32) for k in REQUIRED_KEYS}
    tensors["model.layers.0.unrelated.weight"] = torch.ones(4)
    save_file(tensors, str(checkpoint / "part.safetensors"))
    audit = {"model_id": "fixture", "model_revision": revision,
             "decoder_weight_map": {k: "part.safetensors" for k in REQUIRED_KEYS}}
    with pytest.raises(ValueError, match="snapshots"):
        recover_local(audit, tmp_path, tmp_path / "wrong")
    recover_local(audit, checkpoint, tmp_path / "recovered")
    loaded = load_file(str(tmp_path / "recovered/decoder.safetensors"))
    assert set(loaded) == set(REQUIRED_KEYS)


@pytest.mark.parametrize("text,expected", [("<think>A or B?</think>\n<answer>B</answer>", "B"),
    ("The answer could be A or B.", None), ("A", "A"), (r"\boxed{B}", "B"), ("", None)])
def test_answer_extraction_no_rationale_guessing(text, expected):
    assert extract_answer(text) == expected


def test_depth_positions_are_input_positions_not_previous_logits():
    ids = torch.tensor([[4, 9, 9, 9, 9, 3]])
    assert depth_positions(ids, 9, 1).tolist() == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="exactly 4"):
        depth_positions(ids, 9, 2)


def test_identity_replay_does_not_reuse_cache():
    class MiniModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.norm = nn.LayerNorm(3)
            self.rope_deltas = torch.tensor([999])
        def forward(self, input_ids, attention_mask, use_cache, return_dict):
            assert not use_cache and self.rope_deltas is None
            values = input_ids.float().unsqueeze(-1) * torch.tensor([1., 2., 3.])
            hidden = self.model.norm(values)
            return SimpleNamespace(logits=hidden @ torch.ones(3, 5))
    model = MiniModel()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    positions = torch.tensor([1, 2, 3, 4])
    hidden, logits, _ = replay(model, {}, ids, positions)
    identity, patched_logits, _ = replay(model, {}, ids, positions, hidden)
    torch.testing.assert_close(identity, hidden, rtol=0, atol=0)
    torch.testing.assert_close(patched_logits, logits, rtol=0, atol=0)
    assert not model.model.norm._forward_hooks


def test_smoke_is_explicitly_not_native(tmp_path):
    assert main(["smoke", "--out", str(tmp_path / "smoke")]) == 0
    report = read_json(tmp_path / "smoke/report.json")
    assert report["kind"] == "oracle_smoke_only"
    assert report["native_model_executed"] is False


def test_e1_rejects_smoke_provenance(tmp_path):
    from covt_pilot.experiments import run_e1
    write_json(tmp_path / "smoke/run.json", {"kind": "oracle_smoke_only"})
    with pytest.raises(ValueError, match="native E0"):
        run_e1(tmp_path / "smoke", tmp_path / "missing", tmp_path / "out")
