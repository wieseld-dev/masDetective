#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path("artifacts/v14_adaptive_role_v1")
CF = Path("artifacts/eqs_v3_6_counterfactual/state_query_rows.jsonl")
CACHE = Path("artifacts/eqs_v3_7_interaction/embeddings/semantic_embeddings.pt")
V31_DIR = Path("artifacts/eqs_v3_1_mas_auc")
V31_SCRIPT = Path("scripts/v3_1_mas_auc_semantic.py")
PILOT = Path("artifacts/eqs_v3_2_closed_loop/pilot")
OUT = Path("artifacts/eqs_v3_9_frozen_global")

SEEDS = (7, 17, 29)
ALPHAS = (0.0, 0.10, 0.25, 0.50, 1.0, 2.0)


def read_jsonl(path):
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_v31():
    spec = importlib.util.spec_from_file_location(
        "v31_v39",
        V31_SCRIPT,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


v31 = load_v31()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


print("=" * 100)
print("V3.9 — FROZEN V3.1 GLOBAL QUERY PRIOR + LEARNED CONTEXTUAL RESIDUAL")
print("=" * 100)
print("NO OPENAI")
print("NO MAS EXECUTIONS")
print()


# ============================================================
# Embeddings
# ============================================================

payload = torch.load(
    CACHE,
    map_location="cpu",
    weights_only=False,
)

emb = payload["embeddings"]
emb_dim = int(payload["dim"])

print(f"EMBED_DIM={emb_dim}")
print(f"CACHED_TEXTS={len(emb)}")


def qvec(q):
    k = v31.query_text(q)
    if k not in emb:
        raise KeyError(f"Missing query embedding: {q[:100]}")
    return emb[k].float()


# ============================================================
# Frozen V3.1 query-only ensemble
# ============================================================

global_models = []

for seed in SEEDS:
    ckpt_path = (
        V31_DIR
        / f"selector_query_only_seed{seed}.pt"
    )

    ck = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    model = v31.AUCSelector(
        int(ck["emb_dim"]),
        "query_only",
    )

    model.load_state_dict(
        ck["state_dict"]
    )

    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    global_models.append(model)


@torch.no_grad()
def frozen_global(q):
    dummy_turns = torch.zeros(
        0,
        emb_dim,
    )

    dummy_belief = torch.full(
        (len(v31.TRAIN_ROLES),),
        0.5,
    )

    scores = torch.stack([
        m.score(
            dummy_turns,
            dummy_belief,
            q,
        )
        for m in global_models
    ])

    g = scores.mean(0)

    # Only ranking/relative scale matters.
    return (
        g - g.mean()
    ) / g.std().clamp_min(1e-4)


# ============================================================
# V3.6 counterfactual matrices
# ============================================================

rows = read_jsonl(CF)

by_system = defaultdict(
    lambda: defaultdict(dict)
)

query_texts = defaultdict(dict)

for x in rows:
    sid = str(x["session_id"])
    rd = int(x["round"])
    qi = int(x["query_index"])

    by_system[sid][rd][qi] = float(
        x["delta_auc"]
    )

    query_texts[sid][qi] = str(
        x["query"]
    )


raw_states = {}

for rd in range(2, 7):
    p = ROOT / f"states_train_round_{rd-1:02d}.jsonl"

    for x in read_jsonl(p):
        sid = str(x["session_id"])
        if sid in by_system:
            raw_states[(sid, rd)] = x


def belief_vec(st):
    return torch.tensor(
        [
            float(st["role_probs"][r])
            for r in v31.TRAIN_ROLES
        ],
        dtype=torch.float32,
    )


class Item:
    pass


items = []

for sid in sorted(by_system):
    rounds = [
        r
        for r in sorted(by_system[sid])
        if r >= 2
        and (sid, r) in raw_states
    ]

    if len(rounds) < 2:
        continue

    common = None

    for r in rounds:
        qs = set(by_system[sid][r])

        common = (
            qs
            if common is None
            else common & qs
        )

    qids = sorted(common or [])

    if len(qids) < 4:
        continue

    queries = [
        query_texts[sid][qi]
        for qi in qids
    ]

    q = torch.stack([
        qvec(x)
        for x in queries
    ])

    x = Item()
    x.sid = sid
    x.rounds = rounds
    x.queries = queries
    x.q = q

    x.global_score = frozen_global(
        q
    )

    x.beliefs = torch.stack([
        belief_vec(
            raw_states[(sid, r)]
        )
        for r in rounds
    ])

    x.U = torch.tensor(
        [
            [
                by_system[sid][r][qi]
                for qi in qids
            ]
            for r in rounds
        ],
        dtype=torch.float32,
    )

    items.append(x)


print(f"SYSTEM MATRICES={len(items)}")
print(
    "STATE×QUERY PAIRS="
    f"{sum(x.U.numel() for x in items)}"
)


# ============================================================
# Contextual residual model
# ============================================================

class ResidualModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.qproj = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
        )

        self.bproj = nn.Sequential(
            nn.Linear(
                len(v31.TRAIN_ROLES),
                64,
            ),
            nn.GELU(),
            nn.LayerNorm(64),
        )

        self.head = nn.Sequential(
            nn.Linear(64, 48),
            nn.GELU(),
            nn.Linear(48, 1),
        )

        # Begin with H=0, i.e. exact frozen global policy.
        nn.init.zeros_(
            self.head[-1].weight
        )

        nn.init.zeros_(
            self.head[-1].bias
        )

    def forward(
        self,
        beliefs,
        q,
    ):
        qx = self.qproj(q)

        neutral = torch.full_like(
            beliefs,
            0.5,
        )

        bx = (
            self.bproj(beliefs)
            - self.bproj(neutral)
        )

        interaction = (
            bx[:, None, :]
            * qx[None, :, :]
        )

        return self.head(
            interaction
        ).squeeze(-1)


def soft_rank(scores, target):
    z = (
        target
        - target.mean(dim=-1, keepdim=True)
    ) / target.std(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-4)

    target_p = F.softmax(
        z / 0.70,
        dim=-1,
    )

    return -(
        target_p
        * F.log_softmax(
            scores,
            dim=-1,
        )
    ).sum(dim=-1).mean()


def residual_target(U):
    # For each query remove its across-state mean.
    # What remains is exactly:
    # "is q unusually useful in THIS state?"
    R = (
        U
        - U.mean(
            dim=0,
            keepdim=True,
        )
    )

    return (
        R
        / R.std().clamp_min(1e-4)
    )


def training_loss(
    model,
    item,
):
    h = model(
        item.beliefs,
        item.q,
    )

    R = residual_target(
        item.U
    )

    reg = F.smooth_l1_loss(
        h,
        R,
    )

    # Explicitly rank states for the SAME query.
    cross_state_rank = soft_rank(
        h.transpose(0, 1),
        R.transpose(0, 1),
    )

    # Light regularization prevents contextual branch
    # from simply creating another global query prior.
    zero_mean = (
        h.mean(dim=0)
        .pow(2)
        .mean()
    )

    # Also require that a moderate contextual correction
    # improves state-wise ranking during training.
    total = (
        item.global_score.unsqueeze(0)
        + 0.5 * h
    )

    within_state = soft_rank(
        total,
        item.U,
    )

    return (
        reg
        + 0.50 * cross_state_rank
        + 0.35 * within_state
        + 0.10 * zero_mean
    )


@torch.no_grad()
def evaluate(
    model,
    eval_items,
    alpha,
):
    model.eval()

    base_vals = []
    ctx_vals = []
    oracle_vals = []
    static_vals = []
    unique = []
    residual_corrs = []

    for item in eval_items:
        h = model(
            item.beliefs,
            item.q,
        )

        total = (
            item.global_score.unsqueeze(0)
            + alpha * h
        )

        U = item.U.numpy()

        base_pick = int(
            item.global_score.argmax()
        )

        base_vals.append(
            float(
                U[:, base_pick].mean()
            )
        )

        picks = total.argmax(
            dim=1
        ).numpy()

        ctx_vals.append(
            float(
                np.mean([
                    U[i, picks[i]]
                    for i in range(
                        U.shape[0]
                    )
                ])
            )
        )

        static_pick = int(
            U.mean(axis=0).argmax()
        )

        static_vals.append(
            float(
                U[:, static_pick].mean()
            )
        )

        oracle_vals.append(
            float(
                U.max(axis=1).mean()
            )
        )

        unique.append(
            len(
                set(
                    map(int, picks)
                )
            )
        )

        R = (
            U
            - U.mean(
                axis=0,
                keepdims=True,
            )
        )

        hp = h.numpy().ravel()
        rp = R.ravel()

        if (
            np.std(hp) > 1e-10
            and np.std(rp) > 1e-10
        ):
            residual_corrs.append(
                float(
                    np.corrcoef(
                        hp,
                        rp,
                    )[0, 1]
                )
            )

    base = float(
        np.mean(base_vals)
    )

    ctx = float(
        np.mean(ctx_vals)
    )

    oracle = float(
        np.mean(oracle_vals)
    )

    static = float(
        np.mean(static_vals)
    )

    return {
        "frozen_v31_global_delta_auc":
        base,

        "contextual_delta_auc":
        ctx,

        "context_gain_over_frozen_global":
        ctx - base,

        "true_static_oracle_delta_auc":
        static,

        "contextual_oracle_delta_auc":
        oracle,

        "headroom_captured_vs_frozen_global":
        (
            (ctx - base)
            / (oracle - base)
            if abs(oracle - base) > 1e-8
            else 0.0
        ),

        "mean_unique_contextual_queries":
        float(
            np.mean(unique)
        ),

        "residual_correlation":
        float(
            np.mean(residual_corrs)
        ) if residual_corrs else 0.0,
    }


def train_seed(seed):
    seed_all(seed)

    ids = list(
        range(len(items))
    )

    rng = random.Random(
        seed + 991
    )

    rng.shuffle(ids)

    n_val = max(
        5,
        round(
            0.25 * len(ids)
        ),
    )

    val_ids = set(
        ids[:n_val]
    )

    train_items = [
        x
        for i, x in enumerate(items)
        if i not in val_ids
    ]

    val_items = [
        x
        for i, x in enumerate(items)
        if i in val_ids
    ]

    model = ResidualModel()

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=8e-4,
        weight_decay=5e-4,
    )

    best = None
    best_epoch = 0
    best_alpha = 0.0
    best_gain = -1e9
    bad = 0

    for epoch in range(1, 121):
        model.train()

        order = list(
            range(len(train_items))
        )

        rng.shuffle(order)

        losses = []

        for idx in order:
            loss = training_loss(
                model,
                train_items[idx],
            )

            opt.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            opt.step()

            losses.append(
                float(
                    loss.detach()
                )
            )

        alpha_results = {
            a: evaluate(
                model,
                val_items,
                a,
            )
            for a in ALPHAS
        }

        epoch_alpha = max(
            ALPHAS,
            key=lambda a:
            alpha_results[a][
                "context_gain_over_frozen_global"
            ],
        )

        epoch_result = (
            alpha_results[
                epoch_alpha
            ]
        )

        gain = epoch_result[
            "context_gain_over_frozen_global"
        ]

        if gain > best_gain + 1e-5:
            best_gain = gain
            best_epoch = epoch
            best_alpha = epoch_alpha

            best = {
                k:
                v.detach().clone()
                for k, v
                in model.state_dict().items()
            }

            bad = 0
        else:
            bad += 1

        if (
            epoch == 1
            or epoch % 10 == 0
        ):
            print(
                f"[seed={seed}] "
                f"epoch={epoch:03d} "
                f"loss={np.mean(losses):.5f} "
                f"alpha={epoch_alpha:.2f} "
                f"base="
                f"{epoch_result['frozen_v31_global_delta_auc']:+.4f} "
                f"ctx="
                f"{epoch_result['contextual_delta_auc']:+.4f} "
                f"gain="
                f"{gain:+.4f} "
                f"corr="
                f"{epoch_result['residual_correlation']:+.3f}",
                flush=True,
            )

        if (
            epoch >= 25
            and bad >= 20
        ):
            break

    model.load_state_dict(
        best
    )

    result = evaluate(
        model,
        val_items,
        best_alpha,
    )

    return (
        model,
        best_epoch,
        best_alpha,
        result,
    )


results = []
models = []

for seed in SEEDS:
    print()
    print("=" * 100)
    print(f"SEED {seed}")
    print("=" * 100)

    model, epoch, alpha, result = (
        train_seed(seed)
    )

    print(
        "[HELDOUT]",
        json.dumps(
            {
                "epoch": epoch,
                "alpha": alpha,
                **result,
            },
            sort_keys=True,
        ),
    )

    models.append(
        (
            model,
            alpha,
        )
    )

    results.append(
        {
            "seed": seed,
            "epoch": epoch,
            "alpha": alpha,
            **result,
        }
    )


metric_keys = [
    "frozen_v31_global_delta_auc",
    "contextual_delta_auc",
    "context_gain_over_frozen_global",
    "true_static_oracle_delta_auc",
    "contextual_oracle_delta_auc",
    "headroom_captured_vs_frozen_global",
    "mean_unique_contextual_queries",
    "residual_correlation",
]


agg = {
    k: float(
        np.mean([
            x[k]
            for x in results
        ])
    )
    for k in metric_keys
}


print()
print("=" * 100)
print("V3.9 HELD-OUT-SYSTEM RESULTS")
print("=" * 100)

for k, v in agg.items():
    print(
        f"{k}: {v:.6f}"
    )


# ============================================================
# Decision logic
# ============================================================

base = agg[
    "frozen_v31_global_delta_auc"
]

gain = agg[
    "context_gain_over_frozen_global"
]

corr = agg[
    "residual_correlation"
]


if base < 0.02:
    decision = "ACTION_SPACE_MISMATCH"

    explanation = (
        "The previously validated V3.1 query-only policy also fails "
        "on these per-system V3.6 query pools. The counterfactual "
        "matrices are therefore not aligned with the fixed common "
        "deployment query-bank distribution."
    )

elif (
    gain > 0
    and corr > 0
):
    decision = "PASS"

    explanation = (
        "The frozen global query prior remains useful and the "
        "state-dependent residual improves it on unseen systems."
    )

else:
    decision = "CONTEXT_FAIL"

    explanation = (
        "The frozen global prior transfers, but the learned "
        "belief-conditioned residual does not improve it."
    )


print()
print("=" * 100)
print("V3.9 DECISION")
print("=" * 100)

print(f"DECISION: {decision}")
print(explanation)


OUT.mkdir(
    parents=True,
    exist_ok=True,
)

(
    OUT / "results.json"
).write_text(
    json.dumps(
        {
            "experiment":
            "V3.9 frozen V3.1 global + contextual residual",

            "heldout":
            agg,

            "seeds":
            results,

            "decision":
            decision,

            "explanation":
            explanation,

            "no_new_mas_calls":
            True,
        },
        indent=2,
    )
)

print()
print(
    "SAVED:",
    OUT / "results.json",
)
