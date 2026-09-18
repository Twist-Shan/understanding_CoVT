"""Adapter to the exact DepthAnything copy shipped by CoVT.

Preserves upstream PIL resize(256,256) -> RGB numpy array -> image2tensor.
Do not silently change BGR/RGB handling to improve a result.
"""
import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .io import sha256


class DepthExpert:
    def __init__(self, covt_repo, checkpoint, revision, device="cpu"):
        root = Path(covt_repo).resolve()
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        if actual != revision:
            raise ValueError(f"CoVT checkout {actual} does not match audited code revision {revision}.")
        package = root / "train/src/anchors/DepthAnything"
        if not (package / "depth_anything_v2/dpt.py").is_file():
            raise FileNotFoundError(f"Missing CoVT DepthAnything source in {package}")
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "train/src/anchors/DepthAnything"], cwd=root, text=True)
        if dirty.strip():
            raise ValueError("The audited expert source has local modifications.")
        sys.path.insert(0, str(package))
        module = importlib.import_module("depth_anything_v2.dpt")
        if not Path(module.__file__).resolve().is_relative_to(package):
            raise RuntimeError("A different DepthAnything package is already imported.")
        self.model = module.DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.eval().float().to(device)
        self.device = device
        self.provenance = {"checkpoint_sha256": sha256(checkpoint), "code_revision": actual,
                           "feature_layers": [4, 11, 17, 23], "resize": [256, 256],
                           "input_convention": "upstream PIL RGB array passed directly to image2tensor"}

    @torch.inference_mode()
    def encode(self, image):
        resized = image.convert("RGB").resize((256, 256))
        tensor, native_hw = self.model.image2tensor(np.array(resized))
        tensor = tensor.to(self.device)
        patch_hw = tuple(s // 14 for s in tensor.shape[-2:])
        features = self.model.pretrained.get_intermediate_layers(tensor, [4, 11, 17, 23], return_class_token=True)
        depth = F.relu(self.model.depth_head(features, *patch_hw))
        depth = F.interpolate(depth, size=native_hw, mode="bilinear", align_corners=True)
        return [f[0].detach() for f in features], depth, patch_hw, tuple(native_hw)
