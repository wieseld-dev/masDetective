#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch


V14 = Path("artifacts/v14_adaptive_role_v1")
V31 = Path("artifacts/eqs_v3_1_mas_auc")
V31_SCRIPT = Path("scripts/v3_1_mas_auc_semantic.py")
V37_EMB = Path(
    "artifacts/eqs_v3_7_interaction/"
    "embeddings/semantic_embeddings.pt"
)
BANK_PATH = Path(
    "artifacts/eqs_v3_2_closed_loop/"
    "pilot/query_bank.json"
)
OUT = Path("artifacts/eqs_v3_11_shared_panel")

SEEDS = (7, 17, 29)
PANEL_SIZE = 24
QUALITY_POOL = 256


def load_v31():
    spec = importlib.util.spec_from_file_location(
        "v31_v311",
        V31_SCRIPT,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path):
    rows = []

    if not path.exists():
        return rows

    with path.open() as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


v31 = load_v31()

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD FIXED 512-QUERY DEPLOYMENT BANK
# ============================================================

bank_obj = json.loads(
    BANK_PATH.read_text()
)

bank = [
    str(q)
    for q in bank_obj["queries"]
]

assert len(bank) == 512, len(bank)


# ============================================================
# LOAD EXISTING QWEN EMBEDDINGS
# ============================================================

payload = torch.load(
    V37_EMB,
    map_location="cpu",
    weights_only=False,
)

cache = payload["embeddings"]
emb_dim = int(payload["dim"])


def qvec(q):
    key = v31.query_text(q)

    if key not in cache:
        raise RuntimeError(
            f"Missing cached embedding for: {q}"
        )

    x = cache[key].float()

    return (
        x
        / x.norm().clamp_min(1e-8)
    )


Q = torch.stack([
    qvec(q)
    for q in bank
])


# ============================================================
# LOAD FROZEN V3.1 QUERY-ONLY ENSEMBLE
# ============================================================

models = []

for seed in SEEDS:
    ck = torch.load(
        V31 / f"selector_query_only_seed{seed}.pt",
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

    models.append(model)


dummy_turns = torch.zeros(
    0,
    emb_dim,
)

dummy_belief = torch.full(
    (
        len(v31.TRAIN_ROLES),
    ),
    0.5,
)


with torch.no_grad():
    raw_quality = torch.stack([
        m.score(
            dummy_turns,
            dummy_belief,
            Q,
        )
        for m in models
    ]).mean(0).numpy()


# Convert to percentile-like 0..1 quality rank.
order = np.argsort(
    raw_quality
)

quality_rank = np.empty(
    len(bank),
    dtype=float,
)

quality_rank[order] = np.linspace(
    0.0,
    1.0,
    len(bank),
)


# ============================================================
# KEEP GOOD HALF OF BANK, THEN BALANCE QUALITY + DIVERSITY
# ============================================================

quality_candidates = list(
    np.argsort(
        -raw_quality
    )[:QUALITY_POOL]
)

selected = [
    quality_candidates[0]
]


while len(selected) < PANEL_SIZE:
    best_idx = None
    best_score = -1e9
    best_novelty = None

    S = Q[selected]

    for idx in quality_candidates:
        if idx in selected:
            continue

        sims = (
            S
            @ Q[idx]
        )

        max_sim = float(
            sims.max()
        )

        novelty = (
            1.0
            - max_sim
        )

        # Quality remains important, but diversity slightly dominates
        # after selecting the first high-quality anchor.
        combined = (
            0.45
            * float(
                quality_rank[idx]
            )
            +
            0.55
            * novelty
        )

        if combined > best_score:
            best_score = combined
            best_idx = idx
            best_novelty = novelty

    selected.append(
        int(best_idx)
    )


# ============================================================
# PANEL METADATA
# ============================================================

panel = []

for rank, idx in enumerate(
    selected,
    start=1,
):
    if rank == 1:
        max_sim_prev = None
    else:
        prev = Q[
            selected[:rank-1]
        ]

        max_sim_prev = float(
            (
                prev
                @ Q[idx]
            ).max()
        )

    panel.append(
        {
            "panel_index":
            rank - 1,

            "bank_index":
            int(idx),

            "query":
            bank[idx],

            "raw_v31_query_only_score":
            float(
                raw_quality[idx]
            ),

            "quality_percentile":
            float(
                quality_rank[idx]
            ),

            "max_cosine_similarity_to_previous":
            max_sim_prev,
        }
    )


# ============================================================
# TRAIN SYSTEMS
# ============================================================

manifest = json.loads(
    (
        V14
        / "manifest_privileged.json"
    ).read_text()
)

train_systems = [
    x
    for x in manifest["systems"]
    if x["split"] == "train"
]

train_ids = [
    str(x["session_id"])
    for x in train_systems
]

assert len(train_ids) == 24, len(train_ids)


# ============================================================
# CREATE EXACT SAME ACTION SET FOR EVERY TRAIN MAS
# ============================================================

candidate_path = (
    OUT
    / "candidates_train_round_01.jsonl"
)

with candidate_path.open("w") as f:
    for sid in train_ids:
        for p in panel:
            row = {
                "session_id":
                sid,

                "candidate_index":
                int(
                    p["panel_index"]
                ),

                "query":
                p["query"],

                "valid":
                True,

                "group_id":
                f"shared24::{sid}",

                "source":
                "V3.11_shared_panel",

                "bank_index":
                int(
                    p["bank_index"]
                ),
            }

            f.write(
                json.dumps(row)
                + "\n"
            )


# ============================================================
# COPY OLD TARGET CACHE INTO ISOLATED NEW ROOT
#
# The executor will reuse matching (system, query) pairs.
# ============================================================

for name in [
    "target_public_cache.jsonl",
    "target_truth_cache_training_only.jsonl",
]:
    src = V14 / name
    dst = OUT / name

    if src.exists():
        if not dst.exists():
            shutil.copy2(
                src,
                dst,
            )


# ============================================================
# ESTIMATE EXACT CACHE HITS BEFORE RUNNING
# ============================================================

public_rows = read_jsonl(
    OUT
    / "target_public_cache.jsonl"
)

truth_rows = read_jsonl(
    OUT
    / "target_truth_cache_training_only.jsonl"
)

truth_keys = {
    str(x.get("key", ""))
    for x in truth_rows
}

cached_pairs = set()

for x in public_rows:
    key = str(
        x.get("key", "")
    )

    if (
        key
        and key in truth_keys
    ):
        cached_pairs.add(
            (
                str(
                    x.get(
                        "session_id",
                        ""
                    )
                ),
                str(
                    x.get(
                        "query",
                        ""
                    )
                ),
            )
        )


total_pairs = (
    len(train_ids)
    * len(panel)
)

hits = sum(
    1
    for sid in train_ids
    for p in panel
    if (
        sid,
        p["query"],
    ) in cached_pairs
)

new_calls = (
    total_pairs
    - hits
)


# ============================================================
# SEMANTIC PANEL STATISTICS
# ============================================================

P = Q[selected].numpy()

pair_sims = []

for i in range(
    len(P)
):
    for j in range(
        i + 1,
        len(P),
    ):
        pair_sims.append(
            float(
                np.dot(
                    P[i],
                    P[j],
                )
            )
        )


summary = {
    "panel_size":
    len(panel),

    "train_systems":
    len(train_ids),

    "total_system_query_pairs":
    total_pairs,

    "expected_exact_cache_hits":
    hits,

    "expected_new_target_calls":
    new_calls,

    "cache_reuse_fraction":
    hits / total_pairs,

    "mean_panel_pairwise_cosine":
    float(
        np.mean(
            pair_sims
        )
    ),

    "max_panel_pairwise_cosine":
    float(
        np.max(
            pair_sims
        )
    ),

    "mean_quality_percentile":
    float(
        np.mean([
            x[
                "quality_percentile"
            ]
            for x in panel
        ])
    ),
}


(
    OUT
    / "panel.json"
).write_text(
    json.dumps(
        {
            "summary":
            summary,

            "queries":
            panel,
        },
        indent=2,
    )
)


print("=" * 100)
print("V3.11 — SHARED 24-PROBE PANEL")
print("=" * 100)

for k, v in summary.items():
    if isinstance(v, float):
        print(
            f"{k}: {v:.6f}"
        )
    else:
        print(
            f"{k}: {v}"
        )


print()
print("=" * 100)
print("SELECTED PROBES")
print("=" * 100)

for p in panel:
    print(
        f"{p['panel_index']:02d} | "
        f"bank={p['bank_index']:03d} | "
        f"quality={p['quality_percentile']:.3f} | "
        f"{p['query']}"
    )


print()
print("=" * 100)
print("NEXT-STEP COST ESTIMATE")
print("=" * 100)

print(
    f"TOTAL PAIRS : {total_pairs}"
)

print(
    f"CACHE HITS  : {hits}"
)

print(
    f"NEW CALLS   : {new_calls}"
)

print()
print(
    "No target calls were made by V3.11 build."
)

print(
    "Inspect this panel before executing it."
)

print()
print(
    "SAVED:",
    OUT / "panel.json",
)

print(
    "CANDIDATES:",
    candidate_path,
)
