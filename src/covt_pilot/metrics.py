"""Metrics on raw maps only; sign and scale are frozen on calibration families."""
import numpy as np

from .scenes import annulus


def raw_difference(depth, row):
    depth = np.asarray(depth)
    expected = tuple(reversed(row["camera"]["size"]))
    if depth.ndim != 2 or depth.shape != expected or not np.isfinite(depth).all():
        raise ValueError(f"Expected a finite raw map of shape {expected}, got {depth.shape}.")
    medians = []
    for xy in row["points"]:
        if not (0 <= xy[0] < depth.shape[1] and 0 <= xy[1] < depth.shape[0]):
            raise ValueError("Marker outside image.")
        mask = annulus(depth.shape, xy, row["inner_radius"], row["outer_radius"])
        if not mask.any():
            raise ValueError("Empty marker neighborhood.")
        medians.append(float(np.median(depth[mask])))
    return medians[1] - medians[0]


def calibrate(differences, rows, tie_margin=0.05):
    if len(rows) != len(differences) or not rows or any(r["split"] != "calibration" for r in rows):
        raise ValueError("Calibration requires matching calibration-only rows.")
    if tie_margin < 0 or not np.isfinite(tie_margin):
        raise ValueError("Invalid tie margin.")
    differences = np.asarray(differences, dtype=float)
    if not np.isfinite(differences).all():
        raise ValueError("Non-finite calibration margins.")
    labels = np.array([1 if row["label"] == "A" else -1 for row in rows])
    usable = differences != 0
    if not usable.any():
        raise ValueError("Flat calibration maps; depth direction cannot be identified.")
    signed_accuracy = np.mean(np.sign(differences[usable]) == labels[usable])
    if signed_accuracy == 0.5:
        raise ValueError("Depth-direction calibration is tied; inspect decoder before continuing.")
    scale = float(np.median(np.abs(differences[usable])))
    return {"direction": 1 if signed_accuracy > 0.5 else -1, "scale": scale,
            "tie_margin": tie_margin, "families": sorted({r["family"] for r in rows}),
            "examples": len(rows), "calibration_accuracy": max(signed_accuracy, 1-signed_accuracy)}


def score_difference(difference, calibration):
    if calibration["direction"] not in (-1, 1) or calibration["scale"] <= 0:
        raise ValueError("Invalid frozen calibration.")
    margin = float(difference * calibration["direction"] / calibration["scale"])
    if not np.isfinite(margin):
        raise ValueError("Non-finite relation margin.")
    label = None if abs(margin) <= calibration["tie_margin"] else ("A" if margin > 0 else "B")
    return {"margin": margin, "label": label, "abstain": label is None}


def joint_table(rows):
    table = {}
    for row in rows:
        d = "abstain" if row.get("decoder_label") is None else ("correct" if row["decoder_label"] == row["label"] else "wrong")
        a = "invalid" if row.get("answer") not in ("A", "B") else ("correct" if row["answer"] == row["label"] else "wrong")
        key = f"decoder_{d}__answer_{a}"
        table[key] = table.get(key, 0) + 1
    return {"n": len(rows), "counts": table}


def cluster_interval(rows, value_key, seed=0, draws=2000):
    """Equal-family mean and percentile cluster bootstrap, not question-level CI."""
    groups = {}
    for row in rows:
        groups.setdefault(row["family"], []).append(float(row[value_key]))
    if not groups:
        return {"mean": None, "ci95": None, "families": 0}
    means = np.array([np.mean(v) for v in groups.values()])
    if len(means) < 2:
        return {"mean": float(means.mean()), "ci95": None, "families": 1}
    rng = np.random.default_rng(seed)
    boot = rng.choice(means, size=(draws, len(means)), replace=True).mean(1)
    return {"mean": float(means.mean()), "ci95": np.quantile(boot, [0.025, 0.975]).tolist(), "families": len(means)}
