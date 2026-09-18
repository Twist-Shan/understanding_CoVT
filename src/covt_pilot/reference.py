"""Numerically compare the decoder kernel to the audited upstream class."""
import ast
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .decoder import reconstruct
from .io import sha256


def verify_reference(source, expected_sha256=None):
    if expected_sha256 and sha256(source) != expected_sha256:
        raise ValueError("Audited upstream source hash changed.")
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "DepthReconstructor"]
    if len(classes) != 1:
        raise ValueError("Expected exactly one upstream DepthReconstructor.")
    # Execute only the selected official class, not the file's training imports.
    selected = ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[]))
    namespace = {"torch": torch, "nn": nn, "F": F}
    exec(compile(selected, str(source), "exec"), namespace)
    reference = namespace["DepthReconstructor"]().eval()
    generator = torch.Generator().manual_seed(7401)
    errors = []
    for batch, patch_hw, image_hw in [(1, (3, 5), (17, 23)), (2, (4, 3), (11, 19))]:
        tokens = torch.randn(batch, 4, 1024, generator=generator)
        features = [torch.randn(batch, patch_hw[0]*patch_hw[1], 1024, generator=generator) for _ in range(4)]
        expected = reference(tokens, features, patch_hw, image_hw)
        actual = reconstruct(tokens, features, patch_hw, image_hw)
        errors.append(max(float((a-b).abs().max()) for a, b in zip(expected, actual)))
    result = {"source_sha256": sha256(source), "max_absolute_error": max(errors),
              "passed": max(errors) <= 1e-6, "scope": "reconstruction kernel on synthetic tensors; not checkpoint/runtime restoration"}
    if not result["passed"]:
        raise RuntimeError(f"Upstream reconstruction parity failed: {result}")
    return result
