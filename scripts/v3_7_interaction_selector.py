#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import itertools
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path("artifacts/v14_adaptive_role_v1")
CF = Path("artifacts/eqs_v3_6_counterfactual/state_query_rows.jsonl")
OUT = Path("artifacts/eqs_v3_7_interaction")
V31_PATH = Path("scripts/v3_1_mas_auc_semantic.py")
PILOT = Path("artifacts/eqs_v3_2_closed_loop/pilot")

SEEDS = (7, 17, 29)


def read_jsonl(path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_v31():
    spec = importlib.util.spec_from_file_location(
        "v31_interaction",
        V31_PATH,
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


v31 = load_v31()


# ============================================================
# LOAD COUNTERFACTUAL State x Query UTILITIES
# ============================================================

cf_rows = read_jsonl(CF)

by_system = defaultdict(
    lambda: defaultdict(dict)
)

query_text = defaultdict(dict)

for x in cf_rows:
    sid = str(x["session_id"])
    r = int(x["round"])
    qi = int(x["query_index"])

    by_system[sid][r][qi] = float(
        x["delta_auc"]
    )

    query_text[sid][qi] = str(
        x["query"]
    )

systems = sorted(by_system)

print("=" * 100)
print("V3.7 — COUNTERFACTUAL STATE × QUERY INTERACTION SELECTOR")
print("=" * 100)
print("NO OPENAI")
print("NO MAS EXECUTIONS")
print(f"TRAIN systems={len(systems)}")


# ============================================================
# LOAD RAW TRAIN STATES
# ============================================================

state_raw = {}

for r in range(1, 7):
    path = ROOT / f"states_train_round_{r-1:02d}.jsonl"

    for x in read_jsonl(path):
        sid = str(x["session_id"])

        if sid in by_system:
            state_raw[(sid, r)] = x


# ============================================================
# DEV STATES ONLY FOR FINAL STATE-SENSITIVITY AUDIT
# No DEV utility is used for training/tuning.
# ============================================================

dev_states = {}

for r in range(1, 7):
    path = (
        PILOT
        / "history_belief_query"
        / f"states_dev_round_{r-1:02d}.jsonl"
    )

    xs = read_jsonl(path)

    xs.sort(
        key=lambda z: str(z["session_id"])
    )

    dev_states[r] = xs


bank = json.loads(
    (PILOT / "query_bank.json").read_text()
)["queries"]


# ============================================================
# EMBEDDINGS
# ============================================================

all_groups = []

all_queries = sorted({
    str(x["query"])
    for x in cf_rows
} | set(bank))


# Include every TRAIN history.
for (sid, r), st in sorted(state_raw.items()):
    all_groups.append(
        SimpleNamespace(
            sid=sid,
            history=list(st["history"]),
            role_probs=dict(st["role_probs"]),
            candidates=[],
        )
    )


# Include every DEV history.
for r in range(1, 7):
    for st in dev_states[r]:
        all_groups.append(
            SimpleNamespace(
                sid=str(st["session_id"]),
                history=list(st["history"]),
                role_probs=dict(st["role_probs"]),
                candidates=[],
            )
        )


# Put every possible query into one group so the encoder
# definitely materializes every query embedding.
all_groups.append(
    SimpleNamespace(
        sid="__queries__",
        history=[],
        role_probs={
            role: 0.5
            for role in v31.TRAIN_ROLES
        },
        candidates=[
            SimpleNamespace(query=q)
            for q in all_queries
        ],
    )
)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

embed_dir = OUT / "embeddings"
embed_dir.mkdir(parents=True, exist_ok=True)

cache, emb_dim = v31.load_or_build_embeddings(
    ROOT,
    embed_dir,
    all_groups,
    device,
)

print(f"EMBED_DIM={emb_dim}")


def turns_for(st):
    hist = list(st["history"])

    if not hist:
        return torch.zeros(
            0,
            emb_dim,
            dtype=torch.float32,
        )

    return torch.stack([
        cache[
            v31.turn_text(h)
        ].float()
        for h in hist
    ])


def belief_for(st):
    return torch.tensor([
        float(st["role_probs"][role])
        for role in v31.TRAIN_ROLES
    ])


def qvec(q):
    return cache[
        v31.query_text(q)
    ].float()


# ============================================================
# MATRIX DATASET
#
# One item = one MAS:
#
# rows    = attacker states / rounds
# columns = SAME query pool
# U[s,q]  = actual counterfactual ΔAUC
#
# This is the crucial V3.7 difference.
# ============================================================

class MatrixItem:
    def __init__(
        self,
        sid,
        rounds,
        turns,
        beliefs,
        queries,
        qvecs,
        utility,
    ):
        self.sid = sid
        self.rounds = rounds
        self.turns = turns
        self.beliefs = beliefs
        self.queries = queries
        self.qvecs = qvecs
        self.utility = utility


items = []

for sid in systems:
    rounds = sorted(by_system[sid])

    # Round 1 has no previous evidence and cannot genuinely
    # perform system-specific adaptive selection.
    rounds = [
        r
        for r in rounds
        if r >= 2
        and (sid, r) in state_raw
    ]

    if len(rounds) < 2:
        continue

    common_q = None

    for r in rounds:
        qs = set(by_system[sid][r])
        common_q = (
            qs
            if common_q is None
            else common_q & qs
        )

    qids = sorted(common_q or [])

    if len(qids) < 4:
        continue

    queries = [
        query_text[sid][q]
        for q in qids
    ]

    U = torch.tensor(
        [
            [
                by_system[sid][r][q]
                for q in qids
            ]
            for r in rounds
        ],
        dtype=torch.float32,
    )

    item = MatrixItem(
        sid=sid,
        rounds=rounds,
        turns=[
            turns_for(
                state_raw[(sid, r)]
            )
            for r in rounds
        ],
        beliefs=[
            belief_for(
                state_raw[(sid, r)]
            )
            for r in rounds
        ],
        queries=queries,
        qvecs=torch.stack([
            qvec(q)
            for q in queries
        ]),
        utility=U,
    )

    items.append(item)


print(
    f"MATRIX systems={len(items)} "
    f"state-query pairs="
    f"{sum(x.utility.numel() for x in items)}"
)


# ============================================================
# INTERACTION-ONLY MODEL
# ============================================================

class InteractionSelector(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.turn_proj = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
        )

        self.gru = nn.GRU(
            256,
            192,
            batch_first=True,
        )

        self.belief_proj = nn.Sequential(
            nn.Linear(
                len(v31.TRAIN_ROLES),
                64,
            ),
            nn.GELU(),
        )

        self.state_proj = nn.Sequential(
            nn.Linear(
                192 + 64,
                160,
            ),
            nn.GELU(),
            nn.LayerNorm(160),
        )

        self.delta_proj = nn.Sequential(
            nn.Linear(160, 160),
            nn.GELU(),
            nn.LayerNorm(160),
        )

        self.qproj = nn.Sequential(
            nn.Linear(
                emb_dim,
                256,
            ),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 160),
            nn.GELU(),
            nn.LayerNorm(160),
        )

        # IMPORTANT:
        # head only receives multiplicative State × Query
        # interaction. No standalone q vector exists here.
        self.interaction_head = nn.Sequential(
            nn.Linear(160, 96),
            nn.GELU(),
            nn.Linear(96, 1),
        )

    def raw_state(
        self,
        turns,
        belief,
    ):
        b = self.belief_proj(
            belief
        )

        if turns.shape[0] == 0:
            h = torch.zeros(
                192,
                device=belief.device,
            )
        else:
            x = self.turn_proj(
                turns
            ).unsqueeze(0)

            _, hn = self.gru(x)

            h = hn[-1, 0]

        return self.state_proj(
            torch.cat(
                [h, b],
                dim=-1,
            )
        )

    def contextual_state(
        self,
        turns,
        belief,
    ):
        actual = self.raw_state(
            turns,
            belief,
        )

        # Same history length, but NO CONTENT
        # and completely neutral belief.
        neutral_turns = torch.zeros_like(
            turns
        )

        neutral_belief = torch.full_like(
            belief,
            0.5,
        )

        neutral = self.raw_state(
            neutral_turns,
            neutral_belief,
        )

        return self.delta_proj(
            actual - neutral
        )

    def score(
        self,
        turns,
        belief,
        queries,
    ):
        s = self.contextual_state(
            turns,
            belief,
        )

        q = self.qproj(
            queries
        )

        interaction = (
            q * s.unsqueeze(0)
        )

        return self.interaction_head(
            interaction
        ).squeeze(-1)


# ============================================================
# LOSS
# ============================================================

def row_rank_loss(
    scores,
    utility,
):
    uz = (
        utility
        - utility.mean()
    ) / utility.std().clamp_min(
        1e-4
    )

    target = F.softmax(
        uz / 0.70,
        dim=0,
    )

    return -(
        target
        * F.log_softmax(
            scores,
            dim=0,
        )
    ).sum()


def matrix_scores(
    model,
    item,
):
    rows = []

    for turns, belief in zip(
        item.turns,
        item.beliefs,
    ):
        rows.append(
            model.score(
                turns,
                belief,
                item.qvecs,
            )
        )

    return torch.stack(rows)


def double_center(x):
    return (
        x
        - x.mean(
            dim=1,
            keepdim=True,
        )
        - x.mean(
            dim=0,
            keepdim=True,
        )
        + x.mean()
    )


def matrix_loss(
    model,
    item,
):
    P = matrix_scores(
        model,
        item,
    )

    U = item.utility

    # 1. Within each state:
    # rank queries by actual ΔAUC.
    rank = torch.stack([
        row_rank_loss(
            P[i],
            U[i],
        )
        for i in range(
            U.shape[0]
        )
    ]).mean()

    # 2. Explicit State × Query interaction target.
    #
    # Double-centering removes:
    #   - global query quality
    #   - globally easy/hard states
    #
    # leaving the interaction:
    # "this query is unusually useful in THIS state".
    Ui = double_center(U)
    Pi = double_center(P)

    Ui = Ui / Ui.std().clamp_min(
        1e-4
    )

    Pi = Pi / Pi.std().clamp_min(
        1e-4
    )

    interaction = F.smooth_l1_loss(
        Pi,
        Ui,
    )

    return (
        rank
        + 0.75 * interaction
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    eval_items,
):
    model.eval()

    selected = []
    static = []
    oracle = []
    unique_counts = []
    rank_corrs = []

    for item in eval_items:
        P = matrix_scores(
            model,
            item,
        ).cpu().numpy()

        U = item.utility.cpu().numpy()

        picks = np.argmax(
            P,
            axis=1,
        )

        adaptive_u = np.mean([
            U[i, picks[i]]
            for i in range(
                U.shape[0]
            )
        ])

        static_idx = int(
            np.argmax(
                U.mean(axis=0)
            )
        )

        static_u = float(
            U[:, static_idx].mean()
        )

        oracle_u = float(
            U.max(axis=1).mean()
        )

        selected.append(
            adaptive_u
        )

        static.append(
            static_u
        )

        oracle.append(
            oracle_u
        )

        unique_counts.append(
            len(set(
                map(int, picks)
            ))
        )

        # Correlation between predicted and true
        # double-centered interactions.
        Pd = double_center(
            torch.tensor(P)
        ).numpy().ravel()

        Ud = double_center(
            torch.tensor(U)
        ).numpy().ravel()

        if (
            np.std(Pd) > 1e-10
            and np.std(Ud) > 1e-10
        ):
            rank_corrs.append(
                float(
                    np.corrcoef(
                        Pd,
                        Ud,
                    )[0, 1]
                )
            )

    selected = float(
        np.mean(selected)
    )

    static = float(
        np.mean(static)
    )

    oracle = float(
        np.mean(oracle)
    )

    denom = (
        oracle - static
    )

    capture = (
        (selected - static)
        / denom
        if abs(denom) > 1e-8
        else 0.0
    )

    return {
        "selected_delta_auc":
        selected,

        "global_static_delta_auc":
        static,

        "contextual_oracle_delta_auc":
        oracle,

        "gain_over_global_static":
        selected - static,

        "adaptive_headroom_captured":
        capture,

        "mean_unique_selected_queries":
        float(
            np.mean(
                unique_counts
            )
        ),

        "interaction_correlation":
        float(
            np.mean(
                rank_corrs
            )
        ) if rank_corrs else 0.0,
    }


# ============================================================
# TRAIN WITH SYSTEM-LEVEL INTERNAL VALIDATION
# ============================================================

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
        4,
        round(
            0.20 * len(ids)
        ),
    )

    val_idx = set(
        ids[:n_val]
    )

    train_items = [
        x
        for i, x in enumerate(items)
        if i not in val_idx
    ]

    val_items = [
        x
        for i, x in enumerate(items)
        if i in val_idx
    ]

    model = InteractionSelector(
        emb_dim
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=5e-4,
    )

    best_epoch = 1
    best_metric = -1e9
    best_state = None
    bad = 0

    for epoch in range(
        1,
        101,
    ):
        model.train()

        order = list(
            range(len(train_items))
        )

        rng.shuffle(order)

        losses = []

        for i in order:
            loss = matrix_loss(
                model,
                train_items[i],
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

        val = evaluate(
            model,
            val_items,
        )

        # Primary validation target:
        # beat the best static/global query.
        metric = (
            val[
                "gain_over_global_static"
            ]
        )

        if metric > best_metric + 1e-5:
            best_metric = metric
            best_epoch = epoch

            best_state = {
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
                f"val_gain_static="
                f"{val['gain_over_global_static']:+.5f} "
                f"capture="
                f"{val['adaptive_headroom_captured']:+.3f} "
                f"unique="
                f"{val['mean_unique_selected_queries']:.2f} "
                f"corr="
                f"{val['interaction_correlation']:.3f}",
                flush=True,
            )

        if (
            epoch >= 20
            and bad >= 20
        ):
            break

    model.load_state_dict(
        best_state
    )

    final_val = evaluate(
        model,
        val_items,
    )

    print(
        f"[BEST seed={seed}] "
        f"epoch={best_epoch} "
        f"{json.dumps(final_val, sort_keys=True)}",
        flush=True,
    )

    return (
        model,
        best_epoch,
        final_val,
    )


models = []
training = []

for seed in SEEDS:
    print()
    print("=" * 100)
    print(f"SEED {seed}")
    print("=" * 100)

    model, epoch, val = train_seed(
        seed
    )

    torch.save(
        {
            "state_dict":
            model.state_dict(),

            "emb_dim":
            emb_dim,

            "seed":
            seed,

            "selected_epoch":
            epoch,

            "architecture":
            "V3.7 interaction-only State×Query",

            "roles":
            list(
                v31.TRAIN_ROLES
            ),
        },
        OUT
        / f"selector_interaction_seed{seed}.pt",
    )

    models.append(model)

    training.append(
        {
            "seed": seed,
            "selected_epoch": epoch,
            "internal_validation": val,
        }
    )


# ============================================================
# GENERALIZATION AUDIT ON EXISTING 8 DEV STATES
#
# No DEV utility is used here.
# We only ask whether DIFFERENT observed systems now produce
# DIFFERENT query rankings.
# ============================================================

bank_vec = torch.stack([
    qvec(q)
    for q in bank
])


@torch.no_grad()
def ensemble_score(
    turns,
    belief,
):
    return torch.stack([
        m.score(
            turns,
            belief,
            bank_vec,
        )
        for m in models
    ]).mean(0)


def rankvec(x):
    order = np.argsort(-x)

    ranks = np.empty(
        len(x),
        dtype=float,
    )

    ranks[order] = np.arange(
        len(x),
        dtype=float,
    )

    return ranks


audit = []

for r in range(2, 7):
    states = dev_states[r]

    true_scores = []
    same_hist_swap = []
    same_belief_swap = []

    top1 = []

    prepared = []

    for st in states:
        prepared.append(
            (
                turns_for(st),
                belief_for(st),
            )
        )

    for i, st in enumerate(states):
        turns, belief = prepared[i]

        s = ensemble_score(
            turns,
            belief,
        )

        true_scores.append(
            s.numpy()
        )

        p = int(
            s.argmax()
        )

        top1.append(
            p
        )

        j = (
            i + 1
        ) % len(states)

        other_turns, other_belief = (
            prepared[j]
        )

        sh = ensemble_score(
            other_turns,
            belief,
        )

        sb = ensemble_score(
            turns,
            other_belief,
        )

        same_hist_swap.append(
            int(
                sh.argmax()
            ) == p
        )

        same_belief_swap.append(
            int(
                sb.argmax()
            ) == p
        )

    pair_corr = []

    for a, b in itertools.combinations(
        true_scores,
        2,
    ):
        ra = rankvec(a)
        rb = rankvec(b)

        pair_corr.append(
            float(
                np.corrcoef(
                    ra,
                    rb,
                )[0, 1]
            )
        )

    row = {
        "round": r,

        "unique_top1":
        len(set(top1)),

        "same_top1_history_swap":
        float(
            np.mean(
                same_hist_swap
            )
        ),

        "same_top1_belief_swap":
        float(
            np.mean(
                same_belief_swap
            )
        ),

        "cross_system_rank_corr":
        float(
            np.mean(
                pair_corr
            )
        ),
    }

    audit.append(row)

    print(
        f"[DEV AUDIT r={r}] "
        f"uniqueQ={row['unique_top1']} "
        f"historySwap="
        f"{row['same_top1_history_swap']:.3f} "
        f"beliefSwap="
        f"{row['same_top1_belief_swap']:.3f} "
        f"rankCorr="
        f"{row['cross_system_rank_corr']:.4f}",
        flush=True,
    )


aggregate_audit = {
    "mean_unique_top1":
    float(
        np.mean([
            x["unique_top1"]
            for x in audit
        ])
    ),

    "mean_same_top1_history_swap":
    float(
        np.mean([
            x["same_top1_history_swap"]
            for x in audit
        ])
    ),

    "mean_same_top1_belief_swap":
    float(
        np.mean([
            x["same_top1_belief_swap"]
            for x in audit
        ])
    ),

    "mean_cross_system_rank_corr":
    float(
        np.mean([
            x["cross_system_rank_corr"]
            for x in audit
        ])
    ),
}


print()
print("=" * 100)
print("V3.7 FINAL STATE-SENSITIVITY AUDIT")
print("=" * 100)

for k, v in aggregate_audit.items():
    print(
        f"{k}: {v:.6f}"
    )


internal = {
    k: float(
        np.mean([
            x[
                "internal_validation"
            ][k]
            for x in training
        ])
    )
    for k in [
        "selected_delta_auc",
        "global_static_delta_auc",
        "contextual_oracle_delta_auc",
        "gain_over_global_static",
        "adaptive_headroom_captured",
        "mean_unique_selected_queries",
        "interaction_correlation",
    ]
}


print()
print("=" * 100)
print("V3.7 INTERNAL HELD-OUT-SYSTEM RESULTS")
print("=" * 100)

for k, v in internal.items():
    print(
        f"{k}: {v:.6f}"
    )


result = {
    "experiment":
    "V3.7 Counterfactual State x Query Interaction Selector",

    "training_data":
    {
        "systems":
        len(items),

        "source":
        "V3.6 TRAIN-only counterfactual replay",

        "dev_utility_used":
        False,
    },

    "architecture":
    {
        "state":
        "history GRU + role belief",

        "state_representation":
        "real_state - neutral_same_length_state",

        "query":
        "Qwen2.5 semantic embedding",

        "scorer":
        "interaction-only elementwise State×Query MLP",

        "standalone_query_branch":
        False,
    },

    "loss":
    (
        "within-state query ranking + "
        "double-centered State×Query interaction loss"
    ),

    "training":
    training,

    "internal_heldout_system_mean":
    internal,

    "dev_state_sensitivity":
    {
        "rounds":
        audit,

        "aggregate":
        aggregate_audit,
    },

    "no_new_mas_calls":
    True,
}


(
    OUT / "results.json"
).write_text(
    json.dumps(
        result,
        indent=2,
    )
)

print()
print(
    "SAVED:",
    OUT / "results.json",
)
