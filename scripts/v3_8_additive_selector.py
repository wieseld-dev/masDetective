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
CACHE_PATH = Path(
    "artifacts/eqs_v3_7_interaction/embeddings/semantic_embeddings.pt"
)
PILOT = Path("artifacts/eqs_v3_2_closed_loop/pilot")
OUT = Path("artifacts/eqs_v3_8_additive")
V31_PATH = Path("scripts/v3_1_mas_auc_semantic.py")

SEEDS = (7, 17, 29)


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
        "v31_v38",
        V31_PATH,
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


print("=" * 100)
print("V3.8 — ADDITIVE GLOBAL + BELIEF×QUERY CONTEXTUAL SELECTOR")
print("=" * 100)
print("NO OPENAI")
print("NO MAS EXECUTIONS")
print()


# ------------------------------------------------------------------
# Load V3.6 counterfactual matrix.
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Load raw belief states.
# ------------------------------------------------------------------

raw_states = {}

for rd in range(2, 7):
    p = ROOT / f"states_train_round_{rd-1:02d}.jsonl"

    for x in read_jsonl(p):
        sid = str(x["session_id"])
        if sid in by_system:
            raw_states[(sid, rd)] = x


# ------------------------------------------------------------------
# Reuse exact Qwen embeddings already computed by V3.7.
# ------------------------------------------------------------------

if not CACHE_PATH.exists():
    raise RuntimeError(
        f"Missing embedding cache: {CACHE_PATH}"
    )

payload = torch.load(
    CACHE_PATH,
    map_location="cpu",
    weights_only=False,
)

cache = payload["embeddings"]
emb_dim = int(payload["dim"])

print(f"EMBED_DIM={emb_dim}")
print(f"CACHED_TEXTS={len(cache)}")


def qvec(text):
    key = v31.query_text(text)

    if key not in cache:
        raise KeyError(
            f"Query embedding missing from V3.7 cache: {text[:100]}"
        )

    return cache[key].float()


def belief_vec(state):
    return torch.tensor(
        [
            float(state["role_probs"][r])
            for r in v31.TRAIN_ROLES
        ],
        dtype=torch.float32,
    )


# ------------------------------------------------------------------
# One MatrixItem = one MAS.
#
# U[state, query] is known from V3.6 counterfactual replay.
# ------------------------------------------------------------------

class MatrixItem:
    pass


items = []

for sid in sorted(by_system):
    rounds = [
        rd
        for rd in sorted(by_system[sid])
        if rd >= 2
        and (sid, rd) in raw_states
    ]

    if len(rounds) < 2:
        continue

    common = None

    for rd in rounds:
        qset = set(by_system[sid][rd])
        common = (
            qset
            if common is None
            else common & qset
        )

    qids = sorted(common or [])

    if len(qids) < 4:
        continue

    item = MatrixItem()
    item.sid = sid
    item.rounds = rounds

    item.queries = [
        query_texts[sid][qi]
        for qi in qids
    ]

    item.q = torch.stack(
        [
            qvec(q)
            for q in item.queries
        ]
    )

    item.beliefs = torch.stack(
        [
            belief_vec(
                raw_states[(sid, rd)]
            )
            for rd in rounds
        ]
    )

    item.U = torch.tensor(
        [
            [
                by_system[sid][rd][qi]
                for qi in qids
            ]
            for rd in rounds
        ],
        dtype=torch.float32,
    )

    items.append(item)


print(f"SYSTEM MATRICES={len(items)}")
print(
    "STATE×QUERY PAIRS="
    f"{sum(int(x.U.numel()) for x in items)}"
)


# ------------------------------------------------------------------
# Additive decomposition:
#
# score(s,q) = G(q) + H(b,q)
#
# G learns static/global utility.
# H learns only residual:
#
# R(s,q) = U(s,q) - mean_s U(s,q)
# ------------------------------------------------------------------

class AdditiveSelector(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        # Completely independent global branch.
        self.global_q = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
        )

        self.global_head = nn.Linear(
            64,
            1,
        )

        # Separate contextual query representation.
        self.ctx_q = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 48),
            nn.LayerNorm(48),
        )

        # Current belief = location in hypothesis space.
        self.ctx_b = nn.Sequential(
            nn.Linear(
                len(v31.TRAIN_ROLES),
                48,
            ),
            nn.GELU(),
            nn.LayerNorm(48),
        )

        # Interaction only.
        self.ctx_head = nn.Sequential(
            nn.Linear(48, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # Start contextual correction at zero.
        nn.init.zeros_(
            self.ctx_head[-1].weight
        )

        nn.init.zeros_(
            self.ctx_head[-1].bias
        )

    def global_score(self, q):
        return self.global_head(
            self.global_q(q)
        ).squeeze(-1)

    def contextual_score(
        self,
        beliefs,
        q,
    ):
        qx = self.ctx_q(q)

        neutral = torch.full_like(
            beliefs,
            0.5,
        )

        # Difference from neutral belief.
        bx = (
            self.ctx_b(beliefs)
            - self.ctx_b(neutral)
        )

        # S × Q interaction.
        interaction = (
            bx[:, None, :]
            * qx[None, :, :]
        )

        return self.ctx_head(
            interaction
        ).squeeze(-1)

    def components(
        self,
        beliefs,
        q,
    ):
        g = self.global_score(q)
        h = self.contextual_score(
            beliefs,
            q,
        )

        total = (
            g.unsqueeze(0)
            + h
        )

        return g, h, total


def soft_rank_loss(
    score,
    target,
):
    z = (
        target
        - target.mean()
    ) / target.std().clamp_min(
        1e-4
    )

    p = F.softmax(
        z / 0.70,
        dim=-1,
    )

    return -(
        p
        * F.log_softmax(
            score,
            dim=-1,
        )
    ).sum(dim=-1).mean()


def item_loss(
    model,
    item,
    device,
):
    q = item.q.to(device)
    b = item.beliefs.to(device)
    U = item.U.to(device)

    g, h, total = model.components(
        b,
        q,
    )

    # True static value of each query.
    G_target = U.mean(
        dim=0
    )

    # Pure state-dependent correction.
    R_target = (
        U
        - G_target.unsqueeze(0)
    )

    # Learn static query quality.
    global_rank = soft_rank_loss(
        g,
        G_target,
    )

    global_reg = F.smooth_l1_loss(
        g,
        G_target,
    )

    # Learn contextual residual.
    residual_reg = F.smooth_l1_loss(
        h,
        R_target,
    )

    # Final policy must rank correctly in each state.
    contextual_rank = soft_rank_loss(
        total,
        U,
    )

    loss = (
        contextual_rank
        + 0.40 * global_rank
        + 0.50 * global_reg
        + 1.00 * residual_reg
    )

    return loss


@torch.no_grad()
def evaluate(
    model,
    eval_items,
    device,
):
    model.eval()

    learned_global = []
    learned_context = []
    true_static = []
    oracle = []

    contextual_unique = []
    residual_corr = []

    for item in eval_items:
        q = item.q.to(device)
        b = item.beliefs.to(device)
        U = item.U.to(device)

        g, h, total = model.components(
            b,
            q,
        )

        U_np = U.cpu().numpy()
        g_np = g.cpu().numpy()
        h_np = h.cpu().numpy()
        t_np = total.cpu().numpy()

        global_pick = int(
            np.argmax(g_np)
        )

        ctx_picks = np.argmax(
            t_np,
            axis=1,
        )

        learned_global.append(
            float(
                U_np[:, global_pick].mean()
            )
        )

        learned_context.append(
            float(
                np.mean([
                    U_np[i, ctx_picks[i]]
                    for i in range(
                        U_np.shape[0]
                    )
                ])
            )
        )

        true_static_pick = int(
            np.argmax(
                U_np.mean(axis=0)
            )
        )

        true_static.append(
            float(
                U_np[
                    :,
                    true_static_pick
                ].mean()
            )
        )

        oracle.append(
            float(
                U_np.max(
                    axis=1
                ).mean()
            )
        )

        contextual_unique.append(
            len(
                set(
                    map(
                        int,
                        ctx_picks,
                    )
                )
            )
        )

        residual_true = (
            U_np
            - U_np.mean(
                axis=0,
                keepdims=True,
            )
        )

        a = h_np.ravel()
        btrue = residual_true.ravel()

        if (
            np.std(a) > 1e-10
            and np.std(btrue) > 1e-10
        ):
            residual_corr.append(
                float(
                    np.corrcoef(
                        a,
                        btrue,
                    )[0, 1]
                )
            )

    lg = float(
        np.mean(learned_global)
    )

    lc = float(
        np.mean(learned_context)
    )

    ts = float(
        np.mean(true_static)
    )

    oc = float(
        np.mean(oracle)
    )

    return {
        "learned_global_delta_auc":
        lg,

        "learned_contextual_delta_auc":
        lc,

        "true_static_oracle_delta_auc":
        ts,

        "contextual_oracle_delta_auc":
        oc,

        "context_gain_over_learned_global":
        lc - lg,

        "learned_global_gap_to_true_static":
        lg - ts,

        "remaining_oracle_gap":
        oc - lc,

        "adaptive_headroom_captured_vs_learned_global":
        (
            (lc - lg)
            / (oc - lg)
            if abs(oc - lg) > 1e-8
            else 0.0
        ),

        "mean_unique_contextual_queries":
        float(
            np.mean(
                contextual_unique
            )
        ),

        "residual_correlation":
        float(
            np.mean(
                residual_corr
            )
        ) if residual_corr else 0.0,
    }


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"DEVICE={DEVICE}")


def fit_model(
    train_items,
    val_items,
    seed,
):
    seed_all(seed)

    model = AdditiveSelector(
        emb_dim
    ).to(DEVICE)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=8e-4,
        weight_decay=5e-4,
    )

    best = None
    best_epoch = 0
    best_metric = -1e9
    bad = 0

    rng = random.Random(seed)

    for epoch in range(
        1,
        121,
    ):
        model.train()

        order = list(
            range(len(train_items))
        )

        rng.shuffle(order)

        losses = []

        for idx in order:
            loss = item_loss(
                model,
                train_items[idx],
                DEVICE,
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
                    loss.detach().cpu()
                )
            )

        val = evaluate(
            model,
            val_items,
            DEVICE,
        )

        # Primary scientific target:
        # contextual policy must beat its OWN
        # learned global policy.
        metric = val[
            "context_gain_over_learned_global"
        ]

        if metric > best_metric + 1e-5:
            best_metric = metric
            best_epoch = epoch

            best = {
                k:
                v.detach().cpu().clone()
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
                f"global={val['learned_global_delta_auc']:+.4f} "
                f"context={val['learned_contextual_delta_auc']:+.4f} "
                f"gain={val['context_gain_over_learned_global']:+.4f} "
                f"resCorr={val['residual_correlation']:+.3f}",
                flush=True,
            )

        if (
            epoch >= 25
            and bad >= 20
        ):
            break

    model.load_state_dict(best)
    model.to(DEVICE)

    return (
        model,
        best_epoch,
        evaluate(
            model,
            val_items,
            DEVICE,
        ),
    )


def refit_all(
    seed,
    epochs,
):
    seed_all(
        seed + 10000
    )

    model = AdditiveSelector(
        emb_dim
    ).to(DEVICE)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=8e-4,
        weight_decay=5e-4,
    )

    rng = random.Random(
        seed + 10000
    )

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        order = list(
            range(len(items))
        )

        rng.shuffle(order)

        for idx in order:
            loss = item_loss(
                model,
                items[idx],
                DEVICE,
            )

            opt.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            opt.step()

    return model.eval()


training = []
final_models = []


for seed in SEEDS:
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
            0.25
            * len(ids)
        ),
    )

    val_ids = set(
        ids[:n_val]
    )

    tr = [
        x
        for i, x in enumerate(items)
        if i not in val_ids
    ]

    va = [
        x
        for i, x in enumerate(items)
        if i in val_ids
    ]

    print()
    print("=" * 100)
    print(f"SEED {seed}")
    print(
        f"TRAIN SYSTEMS={len(tr)} "
        f"HELDOUT SYSTEMS={len(va)}"
    )
    print("=" * 100)

    _, epoch, val = fit_model(
        tr,
        va,
        seed,
    )

    print(
        "[HELDOUT]",
        json.dumps(
            val,
            sort_keys=True,
        ),
        flush=True,
    )

    final_model = refit_all(
        seed,
        epoch,
    )

    torch.save(
        {
            "state_dict":
            {
                k: v.detach().cpu()
                for k, v
                in final_model.state_dict().items()
            },

            "emb_dim":
            emb_dim,

            "seed":
            seed,

            "selected_epoch":
            epoch,

            "architecture":
            "G(q) + H(belief,q)",

            "roles":
            list(
                v31.TRAIN_ROLES
            ),
        },
        OUT / f"selector_seed{seed}.pt",
    )

    training.append(
        {
            "seed": seed,
            "selected_epoch": epoch,
            "heldout": val,
        }
    )

    final_models.append(
        final_model
    )


# ------------------------------------------------------------------
# Aggregate system-held-out results.
# ------------------------------------------------------------------

keys = list(
    training[0]["heldout"].keys()
)

aggregate = {
    k: float(
        np.mean([
            x["heldout"][k]
            for x in training
        ])
    )
    for k in keys
}


print()
print("=" * 100)
print("V3.8 HELD-OUT-SYSTEM RESULTS")
print("=" * 100)

for k, v in aggregate.items():
    print(
        f"{k}: {v:.6f}"
    )


# ------------------------------------------------------------------
# Existing DEV state-sensitivity audit.
#
# No DEV utility labels.
# ------------------------------------------------------------------

bank = json.loads(
    (PILOT / "query_bank.json").read_text()
)["queries"]

bank_q = torch.stack(
    [
        qvec(q)
        for q in bank
    ]
).to(DEVICE)


@torch.no_grad()
def ensemble_components(
    belief,
):
    gs = []
    hs = []
    ts = []

    for model in final_models:
        g, h, t = model.components(
            belief.unsqueeze(0),
            bank_q,
        )

        gs.append(g)
        hs.append(h[0])
        ts.append(t[0])

    return (
        torch.stack(gs).mean(0),
        torch.stack(hs).mean(0),
        torch.stack(ts).mean(0),
    )


def rank_corr(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    ra = np.empty(
        len(a),
        dtype=float,
    )

    rb = np.empty(
        len(b),
        dtype=float,
    )

    ra[
        np.argsort(-a)
    ] = np.arange(len(a))

    rb[
        np.argsort(-b)
    ] = np.arange(len(b))

    return float(
        np.corrcoef(
            ra,
            rb,
        )[0, 1]
    )


audit = []

for rd in range(2, 7):
    path = (
        PILOT
        / "history_belief_query"
        / f"states_dev_round_{rd-1:02d}.jsonl"
    )

    states = read_jsonl(path)

    states.sort(
        key=lambda x:
        str(x["session_id"])
    )

    total_scores = []
    residual_scores = []

    total_top = []
    residual_top = []

    beliefs = [
        belief_vec(x).to(DEVICE)
        for x in states
    ]

    for b in beliefs:
        _, h, total = ensemble_components(
            b
        )

        total_scores.append(
            total.cpu().numpy()
        )

        residual_scores.append(
            h.cpu().numpy()
        )

        total_top.append(
            int(total.argmax())
        )

        residual_top.append(
            int(h.argmax())
        )

    total_corr = []
    residual_corr = []

    for i, j in itertools.combinations(
        range(len(states)),
        2,
    ):
        total_corr.append(
            rank_corr(
                total_scores[i],
                total_scores[j],
            )
        )

        residual_corr.append(
            rank_corr(
                residual_scores[i],
                residual_scores[j],
            )
        )

    # Swap belief between systems.
    swap_total_same = []
    swap_residual_same = []

    for i, b in enumerate(beliefs):
        j = (
            i + 1
        ) % len(beliefs)

        _, h2, t2 = ensemble_components(
            beliefs[j]
        )

        swap_total_same.append(
            int(t2.argmax())
            == total_top[i]
        )

        swap_residual_same.append(
            int(h2.argmax())
            == residual_top[i]
        )

    row = {
        "round": rd,

        "unique_total_top1":
        len(set(total_top)),

        "unique_residual_top1":
        len(set(residual_top)),

        "belief_swap_same_total_top1":
        float(
            np.mean(
                swap_total_same
            )
        ),

        "belief_swap_same_residual_top1":
        float(
            np.mean(
                swap_residual_same
            )
        ),

        "cross_system_total_rank_corr":
        float(
            np.mean(total_corr)
        ),

        "cross_system_residual_rank_corr":
        float(
            np.mean(residual_corr)
        ),
    }

    audit.append(row)

    print(
        f"[DEV AUDIT r={rd}] "
        f"uniqueTotal={row['unique_total_top1']} "
        f"uniqueResidual={row['unique_residual_top1']} "
        f"swapTotal={row['belief_swap_same_total_top1']:.3f} "
        f"swapResidual={row['belief_swap_same_residual_top1']:.3f} "
        f"corrTotal={row['cross_system_total_rank_corr']:.4f} "
        f"corrResidual={row['cross_system_residual_rank_corr']:.4f}"
    )


audit_agg = {
    "mean_unique_total_top1":
    float(
        np.mean([
            x["unique_total_top1"]
            for x in audit
        ])
    ),

    "mean_unique_residual_top1":
    float(
        np.mean([
            x["unique_residual_top1"]
            for x in audit
        ])
    ),

    "mean_belief_swap_same_total_top1":
    float(
        np.mean([
            x["belief_swap_same_total_top1"]
            for x in audit
        ])
    ),

    "mean_belief_swap_same_residual_top1":
    float(
        np.mean([
            x["belief_swap_same_residual_top1"]
            for x in audit
        ])
    ),

    "mean_cross_system_total_rank_corr":
    float(
        np.mean([
            x["cross_system_total_rank_corr"]
            for x in audit
        ])
    ),

    "mean_cross_system_residual_rank_corr":
    float(
        np.mean([
            x["cross_system_residual_rank_corr"]
            for x in audit
        ])
    ),
}


print()
print("=" * 100)
print("V3.8 DEV STATE-SENSITIVITY AUDIT")
print("=" * 100)

for k, v in audit_agg.items():
    print(
        f"{k}: {v:.6f}"
    )


decision = (
    "PASS"
    if (
        aggregate[
            "context_gain_over_learned_global"
        ] > 0
        and aggregate[
            "residual_correlation"
        ] > 0
    )
    else "FAIL"
)


print()
print("=" * 100)
print("V3.8 DECISION")
print("=" * 100)
print(f"DECISION: {decision}")

if decision == "PASS":
    print(
        "Contextual belief×query correction generalizes "
        "beyond the learned global query prior."
    )
else:
    print(
        "State-dependent opportunity exists, but the current "
        "belief×query representation does not generalize "
        "across held-out systems."
    )


result = {
    "experiment":
    "V3.8 additive global + belief-query contextual selector",

    "heldout_system_results":
    aggregate,

    "training":
    training,

    "dev_state_sensitivity":
    {
        "rounds": audit,
        "aggregate": audit_agg,
    },

    "decision":
    decision,

    "no_dev_utility_used":
    True,

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
