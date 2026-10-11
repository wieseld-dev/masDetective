from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from .schema import EvidenceObservation

EPS = 1e-6


def _safe_stats(xs: Sequence[float]) -> dict[str, float]:
    if not xs:
        return {"mean": 0.0, "median": 0.0, "max": 0.0, "min": 0.0, "std": 0.0, "last": 0.0, "trend": 0.0, "pos_frac": 0.0}
    a = np.asarray(xs, dtype=np.float64)
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "max": float(np.max(a)),
        "min": float(np.min(a)),
        "std": float(np.std(a)),
        "last": float(a[-1]),
        "trend": float(a[-1] - a[0]) if len(a) > 1 else 0.0,
        "pos_frac": float(np.mean(a > 0.0)),
    }


def robust_event_z(evidence: Mapping[str, float], roles: Sequence[str]) -> dict[str, float]:
    vals = np.asarray([float(evidence.get(r, 0.0)) for r in roles], dtype=np.float64)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    scale = max(1.4826 * mad, 0.25)  # floor prevents huge z from near-constant events
    return {r: float(np.clip((float(evidence.get(r, 0.0)) - med) / scale, -8.0, 8.0)) for r in roles}


def event_rank_percentile(evidence: Mapping[str, float], roles: Sequence[str]) -> dict[str, float]:
    order = sorted(roles, key=lambda r: float(evidence.get(r, 0.0)), reverse=True)
    n = max(len(order) - 1, 1)
    rank = {r: i for i, r in enumerate(order)}
    return {r: float(1.0 - rank[r] / n) for r in roles}


BASE_FEATURE_NAMES = (
    "n_history",
    "role_raw_mean", "role_raw_median", "role_raw_max", "role_raw_min", "role_raw_std", "role_raw_last", "role_raw_trend", "role_raw_pos_frac",
    "role_z_mean", "role_z_median", "role_z_max", "role_z_std", "role_z_last", "role_z_pos_frac",
    "role_rank_mean", "role_rank_max", "role_rank_last", "role_top3_frac",
    "target_count", "target_raw_mean", "target_raw_last", "target_z_mean", "target_z_last", "target_rank_mean",
    "cross_count", "cross_raw_mean", "cross_z_mean", "cross_rank_mean",
    "competitor_count", "competitor_raw_mean", "competitor_z_mean",
    "last_action_is_target", "last_action_is_competitor",
)


def role_history_features(history: Sequence[EvidenceObservation], role: str, roles: Sequence[str]) -> np.ndarray:
    if not history:
        return np.zeros(len(BASE_FEATURE_NAMES), dtype=np.float32)

    raw = [o.score(role) for o in history]
    z_maps = [robust_event_z(o.evidence, roles) for o in history]
    rank_maps = [event_rank_percentile(o.evidence, roles) for o in history]
    zs = [m[role] for m in z_maps]
    ranks = [m[role] for m in rank_maps]
    top3 = [1.0 if sorted(roles, key=lambda r: float(o.evidence.get(r, 0.0)), reverse=True).index(role) < 3 else 0.0 for o in history]

    t_idx = [i for i, o in enumerate(history) if o.action.target == role]
    c_idx = [i for i, o in enumerate(history) if o.action.competitor == role]
    x_idx = [i for i, o in enumerate(history) if o.action.target != role]

    def pick(vals, idx):
        return [vals[i] for i in idx]

    rs = _safe_stats(raw)
    zz = _safe_stats(zs)
    rr = _safe_stats(ranks)
    trs = _safe_stats(pick(raw, t_idx)); tzs = _safe_stats(pick(zs, t_idx)); trr = _safe_stats(pick(ranks, t_idx))
    crs = _safe_stats(pick(raw, c_idx)); czs = _safe_stats(pick(zs, c_idx))
    xrs = _safe_stats(pick(raw, x_idx)); xzs = _safe_stats(pick(zs, x_idx)); xrr = _safe_stats(pick(ranks, x_idx))

    vals = [
        float(len(history)),
        rs["mean"], rs["median"], rs["max"], rs["min"], rs["std"], rs["last"], rs["trend"], rs["pos_frac"],
        zz["mean"], zz["median"], zz["max"], zz["std"], zz["last"], zz["pos_frac"],
        rr["mean"], rr["max"], rr["last"], float(np.mean(top3)) if top3 else 0.0,
        float(len(t_idx)), trs["mean"], trs["last"], tzs["mean"], tzs["last"], trr["mean"],
        float(len(x_idx)), xrs["mean"], xzs["mean"], xrr["mean"],
        float(len(c_idx)), crs["mean"], czs["mean"],
        1.0 if history[-1].action.target == role else 0.0,
        1.0 if history[-1].action.competitor == role else 0.0,
    ]
    return np.asarray(vals, dtype=np.float32)


def all_role_feature_matrix(history: Sequence[EvidenceObservation], roles: Sequence[str]) -> np.ndarray:
    return np.vstack([role_history_features(history, r, roles) for r in roles]).astype(np.float32)


def binary_entropy(p: float) -> float:
    p = float(np.clip(p, EPS, 1.0 - EPS))
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))


def total_entropy(belief: Mapping[str, float], roles: Sequence[str]) -> float:
    return float(sum(binary_entropy(float(belief.get(r, 0.5))) for r in roles))
