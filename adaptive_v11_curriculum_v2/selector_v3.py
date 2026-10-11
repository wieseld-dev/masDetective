from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from adaptive_v11_curriculum_v2.selector import (
    FEATURE_DIM,
    _stable_seed,
    feature_vector,
)


def build_model():
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(FEATURE_DIM, 128),
        nn.GELU(),
        nn.Dropout(0.10),

        nn.Linear(128, 64),
        nn.GELU(),
        nn.Dropout(0.05),

        nn.Linear(64, 1),
    )


@lru_cache(maxsize=4)
def _load_model_cached(path_str: str):
    import torch

    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(p)

    try:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(p, map_location="cpu")

    if int(ckpt.get("feature_dim", -1)) != FEATURE_DIM:
        raise RuntimeError(
            f"feature mismatch: checkpoint={ckpt.get('feature_dim')} "
            f"current={FEATURE_DIM}"
        )

    model = build_model()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    mu = np.asarray(ckpt["mu"], dtype=np.float32)
    sd = np.asarray(ckpt["sd"], dtype=np.float32)

    if mu.shape != (FEATURE_DIM,) or sd.shape != (FEATURE_DIM,):
        raise RuntimeError("V3 scaler shape mismatch")

    groups = int(ckpt.get("training_groups", 0))
    pairs = int(ckpt.get("pairwise_examples", 0))

    return model, mu, sd, groups, pairs


def predict_utility(
    model_path: str | Path,
    state,
    query: str,
    candidate_index: int,
):
    import torch

    model, mu, sd, groups, pairs = _load_model_cached(str(model_path))

    x = feature_vector(
        state,
        query,
        candidate_index,
    ).astype(np.float32)

    x = (x - mu) / sd

    with torch.no_grad():
        score = float(
            model(torch.from_numpy(x).unsqueeze(0))
            .squeeze()
            .item()
        )

    return score, "pairwise_mlp_v3", groups, pairs


def choose_candidate(
    model_path: str | Path,
    state,
    candidates: Sequence[Mapping[str, object]],
    *,
    epsilon: float,
    seed_parts: Iterable[object],
):
    valid = [c for c in candidates if bool(c.get("valid"))]

    if not valid:
        return -1, [], "no_valid_candidate"

    scored = []

    for c in valid:
        idx = int(c["candidate_index"])

        pred, _, groups, pairs = predict_utility(
            model_path,
            state,
            str(c.get("query", "")),
            idx,
        )

        scored.append({
            "candidate_index": idx,
            "predicted_utility": float(pred),
            "selector_model_samples": groups,
            "selector_pairwise_examples": pairs,
        })

    rng = random.Random(_stable_seed(*seed_parts))
    eps = min(1.0, max(0.0, float(epsilon)))

    if eps > 0.0 and rng.random() < eps:
        chosen = int(
            rng.choice([x["candidate_index"] for x in scored])
        )
        mode = f"v3_epsilon_explore({eps:.3f})"
    else:
        chosen = int(
            max(
                scored,
                key=lambda x: (
                    float(x["predicted_utility"]),
                    -int(x["candidate_index"]),
                ),
            )["candidate_index"]
        )
        mode = "pairwise_mlp_v3"

    return chosen, scored, mode
