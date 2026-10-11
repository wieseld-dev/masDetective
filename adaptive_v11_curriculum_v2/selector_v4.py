from __future__ import annotations

import math
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

# Existing 192 public-only features
# + 60 explicit query/belief interactions
# + 10 public K-belief features.
FEATURE_DIM_V4 = FEATURE_DIM + 60 + 10


def _mapping_prob(mapping, key: int) -> float:
    if not isinstance(mapping, Mapping):
        return 0.0

    if key in mapping:
        return float(mapping[key])

    sk = str(key)
    if sk in mapping:
        return float(mapping[sk])

    return 0.0


def _k_features(state) -> np.ndarray:
    raw = getattr(state, "k_probs", {}) or {}

    vals = np.asarray(
        [_mapping_prob(raw, k) for k in range(3, 9)],
        dtype=np.float32,
    )

    vals = np.clip(vals, 0.0, None)

    s = float(vals.sum())

    if s > 1e-12:
        vals = vals / s

        expected_k = float(
            sum(
                k * vals[k - 3]
                for k in range(3, 9)
            )
        )

        nz = vals[vals > 1e-12]

        ent = float(
            -np.sum(nz * np.log(nz))
            / math.log(6.0)
        )

        ranked = sorted(
            vals.tolist(),
            reverse=True,
        )

        top_gap = (
            ranked[0] - ranked[1]
            if len(ranked) >= 2
            else ranked[0]
        )

        max_p = float(vals.max())

    else:
        expected_k = 0.0
        ent = 0.0
        top_gap = 0.0
        max_p = 0.0

    return np.concatenate([
        vals,
        np.asarray(
            [
                expected_k / 8.0,
                ent,
                max_p,
                top_gap,
            ],
            dtype=np.float32,
        ),
    ])


def feature_vector_v4(
    state,
    query: str,
    candidate_index: int,
) -> np.ndarray:
    base = feature_vector(
        state,
        query,
        candidate_index,
    ).astype(np.float32)

    if base.shape != (FEATURE_DIM,):
        raise RuntimeError(
            f"bad base feature shape: {base.shape}"
        )

    # Layout inherited from selector.py:
    # 0:20    role probabilities
    # 20:40   role entropies
    # 40:60   focus mask
    # ...
    # 100:120 lexical similarity query <-> role
    probs = base[0:20]
    ents = base[20:40]
    fmask = base[40:60]
    lexical = base[100:120]

    # Explicit interactions are useful here because our training set
    # is modest and the exact interaction we care about is:
    #
    # "does this query target a role which is currently unresolved?"
    interactions = np.concatenate([
        lexical * ents,
        lexical * (1.0 - probs),
        lexical * fmask,
    ]).astype(np.float32)

    kfeat = _k_features(state)

    x = np.concatenate([
        base,
        interactions,
        kfeat,
    ]).astype(np.float32)

    if x.shape != (FEATURE_DIM_V4,):
        raise RuntimeError(
            f"V4 feature shape mismatch: "
            f"{x.shape} != {(FEATURE_DIM_V4,)}"
        )

    if not np.all(np.isfinite(x)):
        raise RuntimeError(
            "V4 features contain non-finite values"
        )

    return x


def build_model():
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(FEATURE_DIM_V4, 192),
        nn.GELU(),
        nn.Dropout(0.10),

        nn.Linear(192, 96),
        nn.GELU(),
        nn.Dropout(0.05),

        nn.Linear(96, 1),
    )


@lru_cache(maxsize=4)
def _load_model_cached(path_str: str):
    import torch

    p = Path(path_str)

    if not p.exists():
        raise FileNotFoundError(p)

    try:
        ckpt = torch.load(
            p,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        ckpt = torch.load(
            p,
            map_location="cpu",
        )

    if int(
        ckpt.get("feature_dim", -1)
    ) != FEATURE_DIM_V4:
        raise RuntimeError(
            "V4 feature dimension mismatch: "
            f"{ckpt.get('feature_dim')} "
            f"!= {FEATURE_DIM_V4}"
        )

    model = build_model()
    model.load_state_dict(
        ckpt["state_dict"]
    )
    model.eval()

    mu = np.asarray(
        ckpt["mu"],
        dtype=np.float32,
    )

    sd = np.asarray(
        ckpt["sd"],
        dtype=np.float32,
    )

    if (
        mu.shape != (FEATURE_DIM_V4,)
        or sd.shape != (FEATURE_DIM_V4,)
    ):
        raise RuntimeError(
            "V4 scaler shape mismatch"
        )

    groups = int(
        ckpt.get("training_groups", 0)
    )

    pairs = int(
        ckpt.get("pairwise_examples", 0)
    )

    return (
        model,
        mu,
        sd,
        groups,
        pairs,
    )


def predict_utility(
    model_path: str | Path,
    state,
    query: str,
    candidate_index: int,
):
    import torch

    (
        model,
        mu,
        sd,
        groups,
        pairs,
    ) = _load_model_cached(
        str(model_path)
    )

    x = feature_vector_v4(
        state,
        query,
        candidate_index,
    )

    x = (
        x - mu
    ) / sd

    with torch.no_grad():
        score = float(
            model(
                torch.from_numpy(
                    x
                ).unsqueeze(0)
            )
            .squeeze()
            .item()
        )

    return (
        score,
        "marginal_pairwise_v4",
        groups,
        pairs,
    )


def choose_candidate(
    model_path: str | Path,
    state,
    candidates: Sequence[
        Mapping[str, object]
    ],
    *,
    epsilon: float = 0.0,
    seed_parts: Iterable[object] = (),
):
    valid = [
        c
        for c in candidates
        if bool(c.get("valid"))
    ]

    if not valid:
        return (
            -1,
            [],
            "no_valid_candidate",
        )

    scored = []

    for c in valid:
        idx = int(
            c["candidate_index"]
        )

        (
            pred,
            _,
            groups,
            pairs,
        ) = predict_utility(
            model_path,
            state,
            str(
                c.get(
                    "query",
                    "",
                )
            ),
            idx,
        )

        scored.append({
            "candidate_index":
                idx,

            "predicted_utility":
                float(pred),

            "selector_model_samples":
                groups,

            "selector_pairwise_examples":
                pairs,
        })

    rng = random.Random(
        _stable_seed(
            *seed_parts
        )
    )

    eps = min(
        1.0,
        max(
            0.0,
            float(epsilon),
        ),
    )

    if (
        eps > 0.0
        and rng.random() < eps
    ):
        chosen = int(
            rng.choice([
                x["candidate_index"]
                for x in scored
            ])
        )

        mode = (
            f"v4_epsilon_explore"
            f"({eps:.3f})"
        )

    else:
        chosen = int(
            max(
                scored,
                key=lambda x: (
                    float(
                        x[
                            "predicted_utility"
                        ]
                    ),
                    -int(
                        x[
                            "candidate_index"
                        ]
                    ),
                ),
            )[
                "candidate_index"
            ]
        )

        mode = (
            "marginal_pairwise_v4"
        )

    return (
        chosen,
        scored,
        mode,
    )
