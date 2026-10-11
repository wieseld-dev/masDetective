from __future__ import annotations

import itertools
import math
import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

KS = (3, 4, 5, 6)
EPS = 1e-12


def normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    s = float(w.sum())
    if not np.isfinite(s) or s <= 0:
        raise RuntimeError(f"invalid posterior normalization: sum={s}")
    return w / s


def exact_hypotheses(n_roles: int = 20):
    zs, ks, ws = [], [], []
    for k in KS:
        n = math.comb(n_roles, k)
        per = (1.0 / len(KS)) / n
        for c in itertools.combinations(range(n_roles), k):
            z = np.zeros(n_roles, dtype=np.float32)
            z[list(c)] = 1.0
            zs.append(z)
            ks.append(k)
            ws.append(per)
    return (
        np.stack(zs).astype(np.float32),
        np.asarray(ks, dtype=np.int16),
        normalize_weights(np.asarray(ws, dtype=np.float64)),
    )


def k_probs(weights: np.ndarray, klabels: np.ndarray) -> dict[int, float]:
    return {k: float(weights[klabels == k].sum()) for k in KS}


def role_probs(weights: np.ndarray, Z: np.ndarray, roles: Sequence[str]) -> dict[str, float]:
    p = np.asarray(weights, dtype=np.float64) @ np.asarray(Z, dtype=np.float64)
    return {str(r): float(p[i]) for i, r in enumerate(roles)}


def map_k(kp: Mapping[int | str, float]) -> int:
    return int(max(kp, key=lambda k: float(kp[k])))


def largest_gap_k(rp: Mapping[str, float]) -> tuple[int, float]:
    vals = sorted((float(v) for v in rp.values()), reverse=True)
    scored = []
    for k in KS:
        scored.append((vals[k - 1] - vals[k], -k, k))
    gap, _, k = max(scored)
    return int(k), float(gap)


def topk_roles(rp: Mapping[str, float], k: int) -> list[str]:
    return [
        r for r, _ in sorted(
            ((str(r), float(p)) for r, p in rp.items()),
            key=lambda x: (-x[1], x[0])
        )[:k]
    ]


def true_false_margin(rp: Mapping[str, float], truth: Sequence[str]) -> float:
    t = set(map(str, truth))
    pos = [float(rp[r]) for r in t]
    neg = [float(p) for r, p in rp.items() if r not in t]
    if not pos or not neg:
        return 0.0
    return float(min(pos) - max(neg))


def true_mask(Z: np.ndarray, roles: Sequence[str], truth: Sequence[str]) -> np.ndarray:
    t = set(map(str, truth))
    tv = np.asarray([1.0 if r in t else 0.0 for r in roles], dtype=np.float32)
    return np.all(Z == tv[None, :], axis=1)


def word_set(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(s).lower()))


def word_jaccard(a: str, b: str) -> float:
    x, y = word_set(a), word_set(b)
    if not x or not y:
        return 0.0
    return len(x & y) / max(1, len(x | y))


def repetition_penalty(query: str, previous_queries: Sequence[str]) -> float:
    return max((word_jaccard(query, q) for q in previous_queries), default=0.0)


def diagnosticity(
    p_e_given_zq: np.ndarray,
    weights: np.ndarray,
    Z: np.ndarray,
) -> np.ndarray:
    """
    Role-wise pre-execution discrimination:
      | E[P(E_i=1)|Z_i=1,q] - E[P(E_i=1)|Z_i=0,q] |
    under the current posterior.
    """
    p = np.asarray(p_e_given_zq, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    z = np.asarray(Z, dtype=np.float64)
    if p.shape != z.shape:
        raise ValueError(f"diagnosticity shape mismatch p={p.shape} Z={z.shape}")

    out = np.zeros(z.shape[1], dtype=np.float64)
    for i in range(z.shape[1]):
        m1 = z[:, i] > 0.5
        m0 = ~m1
        s1 = float(w[m1].sum())
        s0 = float(w[m0].sum())
        if s1 <= EPS or s0 <= EPS:
            continue
        a = float(np.sum(w[m1] * p[m1, i]) / s1)
        b = float(np.sum(w[m0] * p[m0, i]) / s0)
        out[i] = abs(a - b)
    return np.clip(out, 0.0, 1.0)


def coverage_bonus(
    coverage: np.ndarray,
    diag: np.ndarray,
    rp: Mapping[str, float],
    roles: Sequence[str],
) -> float:
    cov = np.asarray(coverage, dtype=np.float64)
    d = np.asarray(diag, dtype=np.float64)
    ps = np.asarray([float(rp[r]) for r in roles], dtype=np.float64)
    uncertainty = np.clip(4.0 * ps * (1.0 - ps), 0.0, 1.0)
    need = (1.0 - np.clip(cov, 0.0, 1.0)) * (0.25 + 0.75 * uncertainty)
    den = max(float(need.sum()), EPS)
    return float(np.clip(np.sum(need * d) / den, 0.0, 1.0))


def reward_visible_truth_gain(
    *,
    before_w: np.ndarray,
    after_w: np.ndarray,
    klabels: np.ndarray,
    Z: np.ndarray,
    roles: Sequence[str],
    truth_roles: Sequence[str],
    coverage: np.ndarray,
    diag: np.ndarray,
    query: str,
    previous_queries: Sequence[str],
) -> dict:
    """
    Training-only reward.
    Truth is used ONLY here for the scalar reward/metrics.
    The posterior update itself remains query + final visible output only.
    """
    truth_roles = tuple(map(str, truth_roles))
    kt = len(truth_roles)
    mask = true_mask(Z, roles, truth_roles)

    kp0 = k_probs(before_w, klabels)
    kp1 = k_probs(after_w, klabels)
    rp0 = role_probs(before_w, Z, roles)
    rp1 = role_probs(after_w, Z, roles)

    pz0 = max(float(before_w[mask].sum()), EPS)
    pz1 = max(float(after_w[mask].sum()), EPS)
    pk0 = max(float(kp0[kt]), EPS)
    pk1 = max(float(kp1[kt]), EPS)

    dlogz_raw = math.log(pz1) - math.log(pz0)
    dlogk_raw = math.log(pk1) - math.log(pk0)
    dm_raw = true_false_margin(rp1, truth_roles) - true_false_margin(rp0, truth_roles)

    z_gain = float(np.clip(dlogz_raw, -2.0, 2.0) / 2.0)
    k_gain = float(np.clip(dlogk_raw, -1.0, 1.0))
    margin_gain = float(np.clip(2.0 * dm_raw, -1.0, 1.0))
    cov_bonus = coverage_bonus(coverage, diag, rp0, roles)
    rep = repetition_penalty(query, previous_queries)

    total = (
        0.50 * z_gain
        + 0.25 * k_gain
        + 0.15 * margin_gain
        + 0.10 * cov_bonus
        - 0.20 * rep
    )

    return {
        "total": float(total),
        "delta_log_exact_raw": float(dlogz_raw),
        "delta_log_k_raw": float(dlogk_raw),
        "delta_margin_raw": float(dm_raw),
        "exact_gain_clipped": z_gain,
        "k_gain_clipped": k_gain,
        "margin_gain_clipped": margin_gain,
        "coverage_bonus": float(cov_bonus),
        "repetition_penalty": float(rep),
        "p_true_composition_before": pz0,
        "p_true_composition_after": pz1,
        "p_true_k_before": pk0,
        "p_true_k_after": pk1,
    }


def save_matrix_npz(path: Path, ids: Sequence[str], matrix: np.ndarray, field: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.asarray(list(ids)), **{field: np.asarray(matrix)})


def load_matrix_npz(path: Path, field: str) -> tuple[list[str], np.ndarray]:
    x = np.load(path, allow_pickle=False)
    return list(map(str, x["ids"].tolist())), np.asarray(x[field])
