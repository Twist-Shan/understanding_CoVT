"""Depth reconstruction following the official CoVT generation-stage code.

Source: train/src/training/covt_qwen2_5_vl.py (DepthReconstructor and
depth_token_generator). The feature-alignment cross-attention is NOT used
in the later reconstruction branch. No softmax, sigmoid or per-map scaling.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .io import read_json, sha256
from .resources import REQUIRED_KEYS


def reconstruct(tokens, patch_features, patch_hw, image_hw):
    if tokens.ndim != 3 or tokens.shape[1] != 4 or len(patch_features) != 4:
        raise ValueError("Exactly four depth tokens and four expert feature layers are required.")
    batch, _, channels = tokens.shape
    height, width = map(int, patch_hw)
    if min(height, width, *image_hw) <= 0:
        raise ValueError("Invalid spatial dimensions.")
    maps = []
    for index, features in enumerate(patch_features):
        if tuple(features.shape) != (batch, height * width, channels):
            raise ValueError("Expert layer shape does not match the token width or patch grid.")
        if not torch.isfinite(features).all() or not torch.isfinite(tokens).all():
            raise ValueError("Non-finite decoder input.")
        token = tokens[:, index:index+1].to(features.dtype)
        scores = torch.bmm(token, features.transpose(1, 2)).reshape(batch, 1, height, width)
        maps.append(F.interpolate(scores, size=tuple(image_hw), mode="bilinear", align_corners=False)[:, 0])
    per_token = torch.stack(maps, dim=1)
    return per_token, per_token.mean(dim=1, keepdim=True)


class NativeDepthDecoder(nn.Module):
    def __init__(self, hidden_size=3584, expert_size=1024):
        super().__init__()
        self.generator = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, expert_size))
        self.restored = False
        self.provenance = None

    @classmethod
    def restore(cls, directory, device="cpu"):
        from safetensors.torch import load_file
        directory = Path(directory)
        provenance = read_json(directory / "provenance.json")
        filename = directory / "decoder.safetensors"
        if sha256(filename) != provenance["file_sha256"]:
            raise ValueError("Decoder checksum does not match recovery provenance.")
        state = load_file(str(filename))
        if set(state) != set(REQUIRED_KEYS):
            raise ValueError("Incomplete or unexpected decoder state; random fallback is prohibited.")
        expected = {"depth_token_generator.0.weight": (3584, 3584),
                    "depth_token_generator.0.bias": (3584,),
                    "depth_token_generator.2.weight": (1024, 3584),
                    "depth_token_generator.2.bias": (1024,)}
        if any(tuple(state[k].shape) != shape for k, shape in expected.items()):
            raise ValueError("Unsupported decoder architecture; expected the official 7B depth generator.")
        if any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("Non-finite decoder weights.")
        decoder = cls().to(dtype=state[REQUIRED_KEYS[0]].dtype)
        decoder.generator.load_state_dict({k.removeprefix("depth_token_generator."): v for k, v in state.items()}, strict=True)
        decoder.restored, decoder.provenance = True, provenance
        return decoder.eval().to(device)

    def forward(self, hidden, patch_features, patch_hw, image_hw):
        if not self.restored:
            raise RuntimeError("Native decoder weights have not been restored. No random-weight decoding.")
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
        weight = self.generator[0].weight
        tokens = self.generator(hidden.to(device=weight.device, dtype=weight.dtype))
        return reconstruct(tokens, patch_features, patch_hw, image_hw)


def map_to_input(depth, input_hw):
    """Explicit evaluation resampling; native 256x256 maps are saved separately."""
    return F.interpolate(depth.float(), size=tuple(input_hw), mode="bilinear", align_corners=False)[0, 0]
