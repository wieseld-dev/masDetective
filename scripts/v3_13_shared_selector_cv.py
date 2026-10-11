from __future__ import annotations

import copy
import gc
import itertools
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, wilcoxon
from transformers import AutoModel, AutoTokenizer

from adaptive_v11_active.io import read_jsonl
from adaptive_v11_active.schema import PublicBeliefState
from v11.role_universe import TRAIN_ROLES


ROOT = Path("artifacts/eqs_v3_11e_final_shared_panel")
V14 = Path("artifacts/v14_adaptive_role_v1")
OUT = Path("artifacts/eqs_v3_13_shared_selector_cv")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

SEEDS = [7, 17, 29]
MODES = [
    "query_only",
    "belief_query",
    "history_belief_query",
]

N_FOLDS = 4
MAX_EPOCHS = 80
PATIENCE = 15


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def zscore_np(x):
    x = np.asarray(x, dtype=float)
    s = x.std()
    if s < 1e-8:
        return np.zeros_like(x)
    return (x - x.mean()) / s


def turn_text(h):
    if isinstance(h, dict):
        q = str(h.get("query", ""))
        a = str(h.get("answer", ""))
    else:
        q = str(getattr(h, "query", ""))
        a = str(getattr(h, "answer", ""))

    return (
        "[USER QUERY]\n"
        + q
        + "\n\n[VISIBLE MAS RESPONSE]\n"
        + a
    )


# ============================================================
# LOAD V3.12 LABEL MATRIX
# ============================================================

rows = read_jsonl(ROOT / "state_query_rows.jsonl")

assert len(rows) == 2880

by_group = defaultdict(list)

for r in rows:
    sid = str(r["session_id"])
    sr = int(r["state_round"])
    by_group[(sid, sr)].append(r)

assert len(by_group) == 120
assert {len(v) for v in by_group.values()} == {24}

systems = sorted({sid for sid, _ in by_group})

assert len(systems) == 24

query_text = {}

for r in rows:
    qidx = int(r["query_index"])
    q = str(r["query"])

    if qidx in query_text:
        assert query_text[qidx] == q
    else:
        query_text[qidx] = q

assert sorted(query_text) == list(range(24))


# ============================================================
# LOAD REAL ATTACKER STATES / HISTORIES
# ============================================================

states = {}

for state_round in range(1, 6):
    p = V14 / f"states_train_round_{state_round:02d}.jsonl"

    xs = read_jsonl(p)
    assert len(xs) == 24

    for x in xs:
        sid = str(x["session_id"])
        states[(sid, state_round)] = \
            PublicBeliefState.from_public_dict(x)

assert len(states) == 120


# ============================================================
# TEXT EMBEDDINGS
# ============================================================

OUT.mkdir(parents=True, exist_ok=True)

cache_path = OUT / "semantic_embeddings.pt"

texts = set(query_text.values())

for st in states.values():
    for h in st.history:
        texts.add(turn_text(h))

texts = sorted(texts)

print("=" * 100)
print("V3.13 SHARED-ACTION STRICT HELD-OUT-SYSTEM SELECTOR")
print("=" * 100)
print("systems:", len(systems))
print("states:", len(by_group))
print("queries/state:", 24)
print("labels:", len(rows))
print("unique semantic texts:", len(texts))
print("DEV SYSTEMS USED: NO")
print("NEW MAS CALLS: NO")
print()


def build_embeddings():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("EMBED DEVICE:", device)
    print("SEMANTIC MODEL:", MODEL_NAME)

    tok = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
        use_fast=True,
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
        torch_dtype=(
            torch.float16
            if device.type == "cuda"
            else torch.float32
        ),
    ).to(device)

    model.eval()

    cache = {}
    batch_size = 8

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_text = texts[start:start + batch_size]

            enc = tok(
                batch_text,
                padding=True,
                truncation=True,
                max_length=768,
                return_tensors="pt",
            )

            enc = {
                k: v.to(device)
                for k, v in enc.items()
            }

            out = model(**enc)

            h = out.last_hidden_state.float()
            mask = enc["attention_mask"].unsqueeze(-1).float()

            emb = (
                (h * mask).sum(dim=1)
                / mask.sum(dim=1).clamp_min(1.0)
            )

            emb = F.normalize(
                emb,
                p=2,
                dim=-1,
            ).cpu()

            for t, e in zip(batch_text, emb):
                cache[t] = e

            print(
                f"[embed] "
                f"{min(start + batch_size, len(texts))}"
                f"/{len(texts)}",
                flush=True,
            )

    payload = {
        "model": MODEL_NAME,
        "dim": int(next(iter(cache.values())).numel()),
        "embeddings": cache,
    }

    torch.save(payload, cache_path)

    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return payload


if cache_path.exists():
    payload = torch.load(
        cache_path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        payload.get("model") == MODEL_NAME
        and all(t in payload["embeddings"] for t in texts)
    ):
        print("EMBED CACHE: HIT")
    else:
        print("EMBED CACHE: STALE -> REBUILD")
        payload = build_embeddings()
else:
    payload = build_embeddings()

emb_cache = payload["embeddings"]
EMB_DIM = int(payload["dim"])

print("EMBED DIM:", EMB_DIM)


# ============================================================
# SELECTOR ITEMS
# ============================================================

@dataclass
class Item:
    sid: str
    state_round: int
    turns: torch.Tensor
    belief: torch.Tensor
    queries: torch.Tensor
    utility: torch.Tensor
    before_auc: float


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


shared_query_tensor = torch.stack([
    emb_cache[query_text[q]]
    for q in range(24)
]).to(device)


items = []

for (sid, sr), rs in sorted(by_group.items()):
    rs = sorted(
        rs,
        key=lambda x: int(x["query_index"])
    )

    st = states[(sid, sr)]

    hist = [
        emb_cache[turn_text(h)]
        for h in st.history
    ]

    if hist:
        turns = torch.stack(hist).to(device)
    else:
        turns = torch.empty(
            (0, EMB_DIM),
            dtype=torch.float32,
            device=device,
        )

    belief = torch.tensor(
        [
            float(st.role_probs[r])
            for r in TRAIN_ROLES
        ],
        dtype=torch.float32,
        device=device,
    )

    utility = torch.tensor(
        [
            float(x["delta_auc"])
            for x in rs
        ],
        dtype=torch.float32,
        device=device,
    )

    items.append(
        Item(
            sid=sid,
            state_round=sr,
            turns=turns,
            belief=belief,
            queries=shared_query_tensor,
            utility=utility,
            before_auc=float(rs[0]["before_auc"]),
        )
    )

assert len(items) == 120


# ============================================================
# SAME V3.1 SELECTOR ARCHITECTURE
# ============================================================

class AUCSelector(nn.Module):
    def __init__(self, emb_dim, mode):
        super().__init__()

        assert mode in {
            "query_only",
            "belief_query",
            "history_belief_query",
        }

        self.mode = mode

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
            nn.Linear(len(TRAIN_ROLES), 64),
            nn.GELU(),
        )

        self.state_proj = nn.Sequential(
            nn.Linear(192 + 64, 160),
            nn.GELU(),
            nn.LayerNorm(160),
        )

        self.belief_state_proj = nn.Sequential(
            nn.Linear(64, 160),
            nn.GELU(),
            nn.LayerNorm(160),
        )

        self.qproj = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),

            nn.Linear(256, 160),
            nn.GELU(),
            nn.LayerNorm(160),
        )

        self.head = nn.Sequential(
            nn.Linear(160 * 4, 192),
            nn.GELU(),
            nn.Dropout(0.10),

            nn.Linear(192, 64),
            nn.GELU(),

            nn.Linear(64, 1),
        )

        self.query_only_head = nn.Sequential(
            nn.Linear(160, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 1),
        )

    def state(self, turns, belief):
        b = self.belief_proj(
            belief.unsqueeze(0)
        ).squeeze(0)

        if self.mode == "belief_query":
            return self.belief_state_proj(b)

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

    def score(self, turns, belief, queries):
        q = self.qproj(queries)

        if self.mode == "query_only":
            return self.query_only_head(
                q
            ).squeeze(-1)

        s = self.state(
            turns,
            belief,
        )

        ss = s.unsqueeze(0).expand(
            q.shape[0],
            -1,
        )

        feat = torch.cat(
            [
                ss,
                q,
                ss * q,
                (ss - q).abs(),
            ],
            dim=-1,
        )

        return self.head(
            feat
        ).squeeze(-1)


def group_loss(scores, utility):
    uz = (
        utility - utility.mean()
    ) / utility.std().clamp_min(1e-4)

    target = F.softmax(
        uz / 0.70,
        dim=0,
    )

    rank_loss = -(
        target
        * F.log_softmax(scores, dim=0)
    ).sum()

    sz = (
        scores - scores.mean()
    ) / scores.std().clamp_min(1e-4)

    reg = F.smooth_l1_loss(
        sz,
        uz,
    )

    return rank_loss + 0.20 * reg


@torch.no_grad()
def validation_score(model, xs):
    vals = []

    model.eval()

    for x in xs:
        s = model.score(
            x.turns,
            x.belief,
            x.queries,
        )

        pick = int(s.argmax())

        u = x.utility

        rand = float(u.mean())
        oracle = float(u.max())
        learned = float(u[pick])

        d = oracle - rand

        if abs(d) > 1e-8:
            vals.append(
                (learned - rand) / d
            )

    return float(
        np.mean(vals)
    )


def train_final(
    train_items,
    mode,
    seed,
    fold,
):
    train_sids = sorted({
        x.sid for x in train_items
    })

    rng = random.Random(
        seed + fold * 1000 + 991
    )

    rng.shuffle(train_sids)

    n_val = 3

    val_sids = set(
        train_sids[:n_val]
    )

    fit_items = [
        x for x in train_items
        if x.sid not in val_sids
    ]

    val_items = [
        x for x in train_items
        if x.sid in val_sids
    ]

    seed_all(seed)

    model = AUCSelector(
        EMB_DIM,
        mode,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1.2e-3,
        weight_decay=4e-4,
    )

    best_epoch = 1
    best_val = -1e9
    bad = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        order = list(
            range(len(fit_items))
        )

        random.shuffle(order)

        model.train()

        total = 0.0

        for i in order:
            x = fit_items[i]

            scores = model.score(
                x.turns,
                x.belief,
                x.queries,
            )

            loss = group_loss(
                scores,
                x.utility,
            )

            opt.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            opt.step()

            total += float(
                loss.detach()
            )

        sc = validation_score(
            model,
            val_items,
        )

        if sc > best_val + 1e-5:
            best_val = sc
            best_epoch = epoch
            bad = 0
        else:
            bad += 1

        if (
            epoch == 1
            or epoch % 10 == 0
        ):
            print(
                f"[train] fold={fold} "
                f"mode={mode} "
                f"seed={seed} "
                f"epoch={epoch:03d} "
                f"loss={total / len(fit_items):.5f} "
                f"val={sc:.4f} "
                f"best_epoch={best_epoch}",
                flush=True,
            )

        if (
            epoch >= 15
            and bad >= PATIENCE
        ):
            break

    # Refit from scratch on ALL 18 training systems.
    seed_all(seed + 10000)

    final = AUCSelector(
        EMB_DIM,
        mode,
    ).to(device)

    opt = torch.optim.AdamW(
        final.parameters(),
        lr=1.2e-3,
        weight_decay=4e-4,
    )

    for epoch in range(best_epoch):
        order = list(
            range(len(train_items))
        )

        random.shuffle(order)

        final.train()

        for i in order:
            x = train_items[i]

            scores = final.score(
                x.turns,
                x.belief,
                x.queries,
            )

            loss = group_loss(
                scores,
                x.utility,
            )

            opt.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(
                final.parameters(),
                1.0,
            )

            opt.step()

    final.eval()

    print(
        f"[selected] fold={fold} "
        f"mode={mode} "
        f"seed={seed} "
        f"epoch={best_epoch} "
        f"internal_val={best_val:.4f}",
        flush=True,
    )

    return final, best_epoch, best_val


# ============================================================
# 4-FOLD STRICT SYSTEM SPLIT
# ============================================================

rng = random.Random(20261007)

shuffled = systems[:]
rng.shuffle(shuffled)

fold_systems = [
    shuffled[i::N_FOLDS]
    for i in range(N_FOLDS)
]

assert sorted(
    sum(fold_systems, [])
) == systems

assert {
    len(x)
    for x in fold_systems
} == {6}

print()
print("FOLDS:")

for i, fs in enumerate(fold_systems):
    print(
        f"fold={i} heldout={fs}"
    )


oof = []
training_meta = []


for fold in range(N_FOLDS):

    test_sids = set(
        fold_systems[fold]
    )

    train_items = [
        x for x in items
        if x.sid not in test_sids
    ]

    test_items = [
        x for x in items
        if x.sid in test_sids
    ]

    assert len(train_items) == 90
    assert len(test_items) == 30

    print()
    print("=" * 100)
    print(
        f"FOLD {fold}: "
        f"train_systems=18 "
        f"heldout_systems=6"
    )
    print("=" * 100)

    # Deployable static baseline learned only on training systems.
    train_u = torch.stack([
        x.utility.detach().cpu()
        for x in train_items
    ])

    train_query_mean = (
        train_u.mean(dim=0)
        .numpy()
    )

    best_global_q = int(
        np.argmax(
            train_query_mean
        )
    )

    models = {}

    for mode in MODES:
        models[mode] = []

        for seed in SEEDS:
            m, ep, val = train_final(
                train_items,
                mode,
                seed,
                fold,
            )

            models[mode].append(m)

            training_meta.append({
                "fold": fold,
                "mode": mode,
                "seed": seed,
                "selected_epoch": ep,
                "internal_val": val,
                "heldout_systems": sorted(test_sids),
            })

            ckpt = (
                OUT
                / f"fold{fold}_{mode}_seed{seed}.pt"
            )

            torch.save(
                {
                    "state_dict":
                        m.state_dict(),

                    "emb_dim":
                        EMB_DIM,

                    "mode":
                        mode,

                    "seed":
                        seed,

                    "fold":
                        fold,

                    "train_systems":
                        sorted(
                            set(systems)
                            - test_sids
                        ),

                    "heldout_systems":
                        sorted(test_sids),

                    "roles":
                        list(TRAIN_ROLES),
                },
                ckpt,
            )

    with torch.no_grad():
        for x in test_items:

            rec = {
                "fold": fold,
                "session_id": x.sid,
                "state_round": x.state_round,
                "before_auc": x.before_auc,
                "utility": (
                    x.utility
                    .detach()
                    .cpu()
                    .tolist()
                ),
                "train_query_mean": (
                    train_query_mean.tolist()
                ),
                "global_static_query":
                    best_global_q,
                "random_expected_delta":
                    float(x.utility.mean()),
                "global_static_delta":
                    float(
                        x.utility[
                            best_global_q
                        ]
                    ),
                "contextual_oracle_delta":
                    float(x.utility.max()),
                "contextual_oracle_query":
                    int(
                        x.utility.argmax()
                    ),
                "models": {},
            }

            for mode in MODES:
                seed_scores = []

                for model in models[mode]:
                    s = model.score(
                        x.turns,
                        x.belief,
                        x.queries,
                    )

                    s = (
                        s.detach()
                        .cpu()
                        .numpy()
                    )

                    seed_scores.append(
                        zscore_np(s)
                    )

                ens = np.mean(
                    seed_scores,
                    axis=0,
                )

                pick = int(
                    np.argmax(ens)
                )

                rec["models"][mode] = {
                    "scores":
                        ens.tolist(),

                    "pick":
                        pick,

                    "selected_delta":
                        float(
                            x.utility[pick]
                        ),
                }

            oof.append(rec)

    del models
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


assert len(oof) == 120

assert len({
    x["session_id"]
    for x in oof
}) == 24


# ============================================================
# PRIVILEGED PER-SYSTEM STATIC ORACLE
# ============================================================

oof_by_sid = defaultdict(list)

for r in oof:
    oof_by_sid[
        r["session_id"]
    ].append(r)

for sid, rs in oof_by_sid.items():
    mean_q = np.mean(
        [
            np.asarray(
                r["utility"],
                dtype=float,
            )
            for r in rs
        ],
        axis=0,
    )

    q = int(
        np.argmax(mean_q)
    )

    for r in rs:
        r[
            "per_system_static_oracle_query"
        ] = q

        r[
            "per_system_static_oracle_delta"
        ] = float(
            r["utility"][q]
        )


# ============================================================
# METRICS
# ============================================================

def mean(xs):
    return float(
        np.mean(
            np.asarray(xs, dtype=float)
        )
    )


baseline = {
    "random_expected_delta_auc":
        mean([
            r["random_expected_delta"]
            for r in oof
        ]),

    "learned_global_static_delta_auc":
        mean([
            r["global_static_delta"]
            for r in oof
        ]),

    "per_system_static_oracle_delta_auc":
        mean([
            r[
                "per_system_static_oracle_delta"
            ]
            for r in oof
        ]),

    "contextual_oracle_delta_auc":
        mean([
            r[
                "contextual_oracle_delta"
            ]
            for r in oof
        ]),
}


metrics = {}

for mode in MODES:

    selected = np.asarray([
        r["models"][mode][
            "selected_delta"
        ]
        for r in oof
    ])

    exact = np.asarray([
        int(
            r["models"][mode]["pick"]
            ==
            r["contextual_oracle_query"]
        )
        for r in oof
    ])

    regrets = np.asarray([
        r["contextual_oracle_delta"]
        -
        r["models"][mode][
            "selected_delta"
        ]
        for r in oof
    ])

    corrs = []

    for r in oof:
        rho = spearmanr(
            r["models"][mode]["scores"],
            r["utility"],
        ).statistic

        if np.isfinite(rho):
            corrs.append(float(rho))

    picks_by_sid = defaultdict(list)
    scores_by_sid = defaultdict(list)

    for r in oof:
        picks_by_sid[
            r["session_id"]
        ].append(
            int(
                r["models"][mode]["pick"]
            )
        )

        scores_by_sid[
            r["session_id"]
        ].append(
            np.asarray(
                r["models"][mode]["scores"],
                dtype=float,
            )
        )

    unique_q = [
        len(set(v))
        for v in picks_by_sid.values()
    ]

    cross_state_corrs = []

    for sid, arrs in scores_by_sid.items():
        for a, b in itertools.combinations(
            arrs,
            2,
        ):
            rho = spearmanr(
                a,
                b,
            ).statistic

            if np.isfinite(rho):
                cross_state_corrs.append(
                    float(rho)
                )

    metrics[mode] = {
        "mean_selected_delta_auc":
            float(selected.mean()),

        "positive_delta_rate":
            float(
                np.mean(
                    selected > 0
                )
            ),

        "contextual_oracle_exact_hit_rate":
            float(exact.mean()),

        "mean_contextual_oracle_regret":
            float(regrets.mean()),

        "mean_state_query_spearman":
            mean(corrs),

        "mean_unique_selected_queries_per_system":
            mean(unique_q),

        "mean_predicted_cross_state_rank_corr":
            mean(cross_state_corrs),
    }


# Contextual residual diagnostic:
# Does HBQ learn state-specific departures from the
# train-fold average utility of each query?
true_residual = []
pred_residual = []

for r in oof:
    u = np.asarray(
        r["utility"],
        dtype=float,
    )

    train_q = np.asarray(
        r["train_query_mean"],
        dtype=float,
    )

    hbq = np.asarray(
        r["models"][
            "history_belief_query"
        ]["scores"],
        dtype=float,
    )

    qo = np.asarray(
        r["models"][
            "query_only"
        ]["scores"],
        dtype=float,
    )

    true_residual.extend(
        (u - train_q).tolist()
    )

    pred_residual.extend(
        (
            zscore_np(hbq)
            -
            zscore_np(qo)
        ).tolist()
    )

residual_rho = float(
    spearmanr(
        pred_residual,
        true_residual,
    ).statistic
)


# ============================================================
# SYSTEM-LEVEL PAIRED COMPARISONS
# ============================================================

def system_means(field_fn):
    out = {}

    for sid, rs in oof_by_sid.items():
        out[sid] = mean([
            field_fn(r)
            for r in rs
        ])

    return out


hbq_sys = system_means(
    lambda r:
        r["models"][
            "history_belief_query"
        ]["selected_delta"]
)

qo_sys = system_means(
    lambda r:
        r["models"][
            "query_only"
        ]["selected_delta"]
)

bq_sys = system_means(
    lambda r:
        r["models"][
            "belief_query"
        ]["selected_delta"]
)

gs_sys = system_means(
    lambda r:
        r["global_static_delta"]
)


def paired_report(a, b):
    ids = sorted(a)

    d = np.asarray(
        [
            a[s] - b[s]
            for s in ids
        ],
        dtype=float,
    )

    rng = np.random.default_rng(
        20261007
    )

    boots = []

    for _ in range(20000):
        idx = rng.integers(
            0,
            len(d),
            len(d),
        )

        boots.append(
            float(
                d[idx].mean()
            )
        )

    lo, hi = np.quantile(
        boots,
        [0.025, 0.975],
    )

    try:
        p = float(
            wilcoxon(
                d,
                alternative="greater",
                zero_method="wilcox",
            ).pvalue
        )
    except ValueError:
        p = 1.0

    return {
        "mean_difference":
            float(d.mean()),

        "bootstrap95_ci":
            [
                float(lo),
                float(hi),
            ],

        "wins":
            int(np.sum(d > 0)),

        "ties":
            int(np.sum(d == 0)),

        "losses":
            int(np.sum(d < 0)),

        "wilcoxon_one_sided_p":
            p,
    }


comparisons = {
    "HBQ_minus_query_only":
        paired_report(
            hbq_sys,
            qo_sys,
        ),

    "HBQ_minus_belief_query":
        paired_report(
            hbq_sys,
            bq_sys,
        ),

    "HBQ_minus_global_static":
        paired_report(
            hbq_sys,
            gs_sys,
        ),
}


summary = {
    "experiment":
        "V3.13 strict held-out-system shared-action selector CV",

    "data": {
        "systems": 24,
        "states": 120,
        "queries_per_state": 24,
        "labels": 2880,
        "folds": 4,
        "heldout_systems_per_fold": 6,
        "dev_systems_used": False,
        "new_mas_calls": 0,
        "seeds": SEEDS,
    },

    "fold_systems":
        fold_systems,

    "baselines":
        baseline,

    "models":
        metrics,

    "contextual_residual_spearman_HBQ_minus_QOnly":
        residual_rho,

    "paired_system_level":
        comparisons,
}


(OUT / "summary.json").write_text(
    json.dumps(
        summary,
        indent=2,
    )
)

with (
    OUT / "oof_predictions.jsonl"
).open("w") as f:
    for r in oof:
        f.write(
            json.dumps(r)
            + "\n"
        )

(OUT / "training_meta.json").write_text(
    json.dumps(
        training_meta,
        indent=2,
    )
)


print()
print("=" * 100)
print("V3.13 FINAL OOF RESULTS")
print("=" * 100)

print()
print("BASELINES")

for k, v in baseline.items():
    print(
        f"{k:48s}: {v:+.6f}"
    )

print()
print("MODELS")

for mode in MODES:
    print()
    print(mode.upper())

    for k, v in metrics[mode].items():
        print(
            f"  {k:46s}: {v:.6f}"
        )

print()
print(
    "contextual_residual_spearman_HBQ_minus_QOnly:",
    f"{residual_rho:.6f}",
)

print()
print("PAIRED SYSTEM-LEVEL")

for name, x in comparisons.items():
    print()
    print(name)

    for k, v in x.items():
        print(
            f"  {k}: {v}"
        )

print()
print("SAVED:", OUT / "summary.json")
print("OOF:", OUT / "oof_predictions.jsonl")
