"""Analytic pinhole renderer for an instrumentation pilot, not CLEVR."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .io import new_output, write_json, write_jsonl


def render_spheres(centers, radii, colors, size=336):
    focal = 0.9 * size
    yy, xx = np.mgrid[:size, :size]
    # Unnormalised rays have z=1, so intersection t is CAMERA-AXIS depth.
    rays = np.stack([(xx + 0.5 - size / 2) / focal,
                     (yy + 0.5 - size / 2) / focal, np.ones_like(xx)], axis=-1)
    a = (rays ** 2).sum(-1)
    depth = np.full((size, size), 100.0)
    ids = np.full((size, size), -1, dtype=np.int16)
    rgb = np.full((size, size, 3), [0.55, 0.65, 0.75])
    floor = np.divide(1.3, rays[..., 1], out=np.full_like(a, 100.0), where=rays[..., 1] > 0)
    floor_valid = (floor > 0) & (floor < depth)
    depth[floor_valid] = floor[floor_valid]
    pts = rays * floor[..., None]
    checks = ((np.floor(pts[..., 0]) + np.floor(pts[..., 2])) % 2) * 0.08 + 0.55
    rgb[floor_valid] = np.repeat(checks[..., None], 3, axis=-1)[floor_valid]
    light = np.array([-0.4, -0.8, -0.6])
    light /= np.linalg.norm(light)
    marks = []
    for i, (center, radius, color) in enumerate(zip(centers, radii, colors)):
        center = np.asarray(center)
        dot = np.einsum("hwc,c->hw", rays, center)
        disc = dot ** 2 - a * (center @ center - radius ** 2)
        t = (dot - np.sqrt(np.maximum(disc, 0))) / a
        visible = (disc >= 0) & (t > 0) & (t < depth)
        normals = (rays * t[..., None] - center) / radius
        shading = 0.3 + 0.7 * np.maximum(0, normals @ light)
        rgb[visible] = (np.array(color)[None, None] * shading[..., None])[visible]
        depth[visible] = t[visible]
        ids[visible] = i
        marks.append([int(round(focal * center[0] / center[2] + size / 2 - 0.5)),
                      int(round(focal * center[1] / center[2] + size / 2 - 0.5))])
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)), depth.astype(np.float32), ids, marks


def annulus(shape, xy, inner=5, outer=9):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    distance = (xx - xy[0]) ** 2 + (yy - xy[1]) ** 2
    return (distance >= inner ** 2) & (distance <= outer ** 2)


def generate(out, families=20, seed=0, size=336):
    if families < 2 or size < 224:
        raise ValueError("Use at least 2 families and image size >=224.")
    out = new_output(out)
    rng = np.random.default_rng(seed)
    rows = []
    calibration_count = max(1, families // 5)
    for family in range(families):
        family_id = f"scene_{family:05d}"
        z = np.array([rng.uniform(4.2, 4.8), rng.uniform(6.1, 6.8)])
        if family % 2:
            z = z[::-1]
        radii = rng.uniform(0.68, 0.9, 2)
        colors = rng.uniform(0.25, 0.95, (2, 3))
        label_order = [0, 1] if family % 4 < 2 else [1, 0]
        split = "calibration" if family < calibration_count else "pilot"
        for variant in ("base", "depth_swap", "nuisance", "label_swap"):
            current_z = z[::-1] if variant == "depth_swap" else z
            # Equal image dimensions; object identities are left/right across variants.
            centers = [[-0.18 * current_z[0], 1.3-radii[0], current_z[0]],
                       [0.18 * current_z[1], 1.3-radii[1], current_z[1]]]
            current_colors = 1 - colors * 0.65 if variant == "nuisance" else colors
            image, depth, ids, marks = render_spheres(centers, radii, current_colors, size)
            order = label_order[::-1] if variant == "label_swap" else label_order
            ab = [marks[i] for i in order]
            masks = [annulus(depth.shape, mark) for mark in ab]
            if any(not np.all(ids[mask] == obj) for mask, obj in zip(masks, order)):
                raise ValueError("Marker neighborhood leaves its visible object; enlarge image size.")
            native_z = [float(depth[y, x]) for x, y in ab]
            patch_z = [float(np.median(depth[mask])) for mask in masks]
            if (native_z[0] < native_z[1]) != (patch_z[0] < patch_z[1]):
                raise ValueError("Point and neighborhood labels disagree.")
            sample_id = f"{family_id}_{variant}"
            folder = out / sample_id
            folder.mkdir()
            image.save(folder / "unmarked.png")
            draw = ImageDraw.Draw(image)
            for label, (x, y) in zip(("A", "B"), ab):
                draw.ellipse((x-3, y-3, x+3, y+3), fill="white", outline="black")
                draw.text((x+12, y-18), label, fill="white", stroke_width=1, stroke_fill="black")
            image.save(folder / "image.png")
            np.savez_compressed(folder / "geometry.npz", depth=depth, object_ids=ids)
            rows.append({"id": sample_id, "family": family_id, "split": split,
                         "variant": variant, "image": f"{sample_id}/image.png",
                         "geometry": f"{sample_id}/geometry.npz", "points": ab,
                         "inner_radius": 5, "outer_radius": 9, "point_depths": native_z,
                         "label": "A" if native_z[0] < native_z[1] else "B",
                         "question": "Which marked surface point, A or B, has smaller camera-axis depth (closer to the image plane)? Answer A or B.",
                         "camera": {"focal_px": 0.9 * size, "size": [size, size]},
                         "objects": {"centers": centers, "radii": radii.tolist()},
                         "renderer": "analytic_spheres_v1"})
    write_jsonl(out / "manifest.jsonl", rows)
    write_json(out / "dataset.json", {"seed": seed, "families": families, "size": size,
               "purpose": "instrumentation pilot; not confirmatory evidence",
               "splits": ["calibration", "pilot"], "variants_per_family": 4})
    return {"manifest": str(out / "manifest.jsonl"), "examples": len(rows)}
