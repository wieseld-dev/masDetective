from __future__ import annotations

import hashlib
import math
import os
import random
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from adaptive_v11_active.reward import normalized_word_jaccard
from adaptive_v11_curriculum.policy import focus_roles, h01
from v11.role_universe import ROLE_DESCRIPTIONS, TRAIN_ROLES, anchor_probe

FEATURE_DIM = 192
HASH_DIM = 48
RIDGE = 4.0
MIN_MODEL_SAMPLES = 96
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokens(text: str) -> list[str]:
    return [x.lower() for x in _TOKEN_RE.findall(str(text))]


def _set_tokens(text: str) -> set[str]:
    return set(_tokens(text))


def _jaccard_tokens(a: str, b: str) -> float:
    aa, bb = _set_tokens(a), _set_tokens(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / max(1, len(aa | bb))


def _signed_bucket(token: str, buckets: int) -> tuple[int, float]:
    d = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    n = int.from_bytes(d, "big", signed=False)
    return n % buckets, (1.0 if ((n >> 8) & 1) == 0 else -1.0)


def _hashed_text(text: str, buckets: int = HASH_DIM) -> np.ndarray:
    toks = _tokens(text)
    grams = list(toks)
    grams += [f"{a}::{b}" for a, b in zip(toks, toks[1:])]
    out = np.zeros(buckets, dtype=np.float64)
    for g in grams:
        idx, sign = _signed_bucket(g, buckets)
        out[idx] += sign
    norm = float(np.linalg.norm(out))
    if norm > 0:
        out /= norm
    return out


def _role_text(role: str) -> str:
    return f"{role} {ROLE_DESCRIPTIONS.get(role, role)} {anchor_probe(role)}"


def feature_vector(state, query: str, candidate_index: int) -> np.ndarray:
    """Public-only features. Never uses truth, MAS response, or hidden state."""
    roles = list(TRAIN_ROLES)
    probs = np.array([float(state.role_probs[r]) for r in roles], dtype=np.float64)
    ents = np.array([h01(float(state.role_probs[r])) / math.log(2.0) for r in roles], dtype=np.float64)

    focus = set(focus_roles(state, int(candidate_index), 6))
    fmask = np.array([1.0 if r in focus else 0.0 for r in roles], dtype=np.float64)

    q = str(query)
    lexical = np.array([_jaccard_tokens(q, _role_text(r)) for r in roles], dtype=np.float64)
    hq = _hashed_text(q, HASH_DIM)

    hist_queries = [str(h.query) for h in getattr(state, "history", [])]
    if hist_queries:
        sims = np.array([float(normalized_word_jaccard(q, x)) for x in hist_queries], dtype=np.float64)
        max_hist = float(np.max(sims))
        mean_hist = float(np.mean(sims))
    else:
        max_hist = 0.0
        mean_hist = 0.0

    qtoks = _tokens(q)
    uniq_ratio = len(set(qtoks)) / max(1, len(qtoks))
    ranked = sorted((float(v) for v in state.role_probs.values()), reverse=True)
    top_gap = ranked[0] - ranked[1] if len(ranked) >= 2 else (ranked[0] if ranked else 0.0)
    fidx = [i for i, r in enumerate(roles) if r in focus]
    fent = ents[fidx] if fidx else ents
    fprob = probs[fidx] if fidx else probs

    globals_ = np.array(
        [
            1.0,
            min(1.0, float(getattr(state, "round_index", 0)) / 10.0),
            min(1.0, len(qtoks) / 40.0),
            float(uniq_ratio),
            min(1.0, len(hist_queries) / 10.0),
            max_hist,
            mean_hist,
            float(np.mean(ents)),
            float(np.max(ents)),
            float(np.mean(probs)),
            float(np.max(probs)),
            float(top_gap),
            float(np.mean(fent)),
            float(np.mean(fprob)),
            float(np.max(fent)),
            float(np.min(fprob)),
        ],
        dtype=np.float64,
    )

    cand_onehot = np.zeros(8, dtype=np.float64)
    cand_onehot[int(candidate_index) % len(cand_onehot)] = 1.0

    x = np.concatenate(
        [
            probs,                 # 20
            ents,                  # 20
            fmask,                 # 20
            probs * fmask,         # 20
            ents * fmask,          # 20
            lexical,               # 20
            hq,                    # 48
            globals_,              # 16
            cand_onehot,           # 8
        ]
    )
    if x.shape != (FEATURE_DIM,):
        raise RuntimeError(f"selector feature shape mismatch: {x.shape} != {(FEATURE_DIM,)}")
    if not np.all(np.isfinite(x)):
        raise RuntimeError("selector features contain non-finite values")
    return x


def _empty_stats(ridge: float = RIDGE) -> dict[str, np.ndarray | float | int]:
    return {
        "A": np.eye(FEATURE_DIM, dtype=np.float64) * float(ridge),
        "b": np.zeros(FEATURE_DIM, dtype=np.float64),
        "w": np.zeros(FEATURE_DIM, dtype=np.float64),
        "count": 0,
        "sum_y": 0.0,
        "sum_y2": 0.0,
        "applied_ids": [],
    }


def load_stats(path: str | Path, ridge: float = RIDGE) -> dict[str, np.ndarray | float | int]:
    p = Path(path)
    if not p.exists():
        return _empty_stats(ridge)
    z = np.load(p, allow_pickle=False)
    A = np.asarray(z["A"], dtype=np.float64)
    b = np.asarray(z["b"], dtype=np.float64)
    w = np.asarray(z["w"], dtype=np.float64)
    if A.shape != (FEATURE_DIM, FEATURE_DIM) or b.shape != (FEATURE_DIM,) or w.shape != (FEATURE_DIM,):
        raise RuntimeError(f"invalid selector model dimensions in {p}")
    applied_ids = [str(x) for x in np.asarray(z["applied_ids"]).tolist()] if "applied_ids" in z.files else []
    return {
        "A": A,
        "b": b,
        "w": w,
        "count": int(np.asarray(z["count"]).item()),
        "sum_y": float(np.asarray(z["sum_y"]).item()),
        "sum_y2": float(np.asarray(z["sum_y2"]).item()),
        "applied_ids": applied_ids,
    }


def update_model(
    path: str | Path,
    features: Sequence[np.ndarray],
    targets: Sequence[float],
    *,
    ridge: float = RIDGE,
    update_id: str | None = None,
) -> dict[str, float | int | str]:
    if len(features) != len(targets):
        raise ValueError("features/targets length mismatch")
    if not features:
        stats = load_stats(path, ridge)
        return model_summary(stats)

    X = np.stack([np.asarray(x, dtype=np.float64) for x in features], axis=0)
    y = np.asarray(targets, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != FEATURE_DIM:
        raise ValueError(f"bad X shape {X.shape}")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("non-finite selector update data")

    stats = load_stats(path, ridge)
    applied_ids = list(stats.get("applied_ids", []))
    if update_id is not None and str(update_id) in applied_ids:
        out = model_summary(stats)
        out["status"] = "already_applied"
        out["update_id"] = str(update_id)
        return out
    A = np.asarray(stats["A"], dtype=np.float64) + X.T @ X
    b = np.asarray(stats["b"], dtype=np.float64) + X.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, b, rcond=None)[0]

    count = int(stats["count"]) + int(len(y))
    sum_y = float(stats["sum_y"]) + float(np.sum(y))
    sum_y2 = float(stats["sum_y2"]) + float(np.sum(y * y))

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if update_id is not None:
        applied_ids.append(str(update_id))
    tmp = Path(str(p) + ".tmp.npz")
    np.savez_compressed(
        tmp, A=A, b=b, w=w, count=count, sum_y=sum_y, sum_y2=sum_y2,
        applied_ids=np.asarray(applied_ids, dtype='U64'),
    )
    os.replace(tmp, p)

    out = model_summary({
        "A": A, "b": b, "w": w, "count": count, "sum_y": sum_y, "sum_y2": sum_y2,
        "applied_ids": applied_ids,
    })
    out["status"] = "updated"
    if update_id is not None:
        out["update_id"] = str(update_id)
    return out


def model_summary(stats: Mapping[str, object]) -> dict[str, float | int]:
    count = int(stats["count"])
    mean_y = float(stats["sum_y"]) / max(1, count)
    var_y = max(0.0, float(stats["sum_y2"]) / max(1, count) - mean_y * mean_y)
    w = np.asarray(stats["w"], dtype=np.float64)
    return {
        "count": count,
        "mean_target": mean_y,
        "target_sd": math.sqrt(var_y),
        "weight_l2": float(np.linalg.norm(w)),
    }


def heuristic_score(state, query: str, candidate_index: int) -> float:
    focus = focus_roles(state, int(candidate_index), 6)
    if not focus:
        return 0.0
    ent = np.mean([h01(float(state.role_probs[r])) / math.log(2.0) for r in focus])
    mean_p = np.mean([float(state.role_probs[r]) for r in focus])
    prev = [str(h.query) for h in getattr(state, "history", [])]
    novelty = 1.0 - max((float(normalized_word_jaccard(query, q)) for q in prev), default=0.0)
    # Prefer unresolved regions; mildly penalize repeatedly testing already-high probability regions.
    return float(0.70 * ent + 0.20 * novelty + 0.10 * (1.0 - mean_p))


def predict_utility(
    model_path: str | Path,
    state,
    query: str,
    candidate_index: int,
    *,
    min_samples: int = MIN_MODEL_SAMPLES,
) -> tuple[float, str, int]:
    stats = load_stats(model_path)
    count = int(stats["count"])
    if count < int(min_samples):
        return heuristic_score(state, query, candidate_index), "heuristic_warmup", count
    x = feature_vector(state, query, candidate_index)
    return float(x @ np.asarray(stats["w"], dtype=np.float64)), "ridge_utility", count


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & 0x7FFFFFFF


def choose_candidate(
    model_path: str | Path,
    state,
    candidates: Sequence[Mapping[str, object]],
    *,
    epsilon: float,
    seed_parts: Iterable[object],
    min_samples: int = MIN_MODEL_SAMPLES,
) -> tuple[int, list[dict[str, object]], str]:
    valid = [c for c in candidates if bool(c.get("valid"))]
    if not valid:
        return -1, [], "no_valid_candidate"

    scored: list[dict[str, object]] = []
    modes: list[str] = []
    for c in valid:
        idx = int(c["candidate_index"])
        pred, mode, count = predict_utility(model_path, state, str(c.get("query", "")), idx, min_samples=min_samples)
        scored.append({"candidate_index": idx, "predicted_utility": float(pred), "selector_model_samples": int(count)})
        modes.append(mode)

    rng = random.Random(_stable_seed(*seed_parts))
    eps = min(1.0, max(0.0, float(epsilon)))
    if eps > 0.0 and rng.random() < eps:
        chosen = int(rng.choice([int(x["candidate_index"]) for x in scored]))
        mode = f"epsilon_explore({eps:.3f})"
    else:
        chosen = int(max(scored, key=lambda x: (float(x["predicted_utility"]), -int(x["candidate_index"])))["candidate_index"])
        mode = modes[0] if len(set(modes)) == 1 else "mixed_selector"

    return chosen, scored, mode
