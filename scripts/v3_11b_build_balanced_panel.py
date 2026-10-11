#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


V14 = Path("artifacts/v14_adaptive_role_v1")
V31 = Path("artifacts/eqs_v3_1_mas_auc")
V31_SCRIPT = Path("scripts/v3_1_mas_auc_semantic.py")
BANK_PATH = Path(
    "artifacts/eqs_v3_2_closed_loop/pilot/query_bank.json"
)
BANK_EMB_PATH = Path(
    "artifacts/eqs_v3_7_interaction/"
    "embeddings/semantic_embeddings.pt"
)
OUT = Path("artifacts/eqs_v3_11b_balanced_panel")

SEEDS = (7, 17, 29)
GENERAL_COUNT = 4


ROLE_DESCRIPTIONS = {
    "code":
        "software programming, coding, implementation, debugging, algorithms and source code",

    "writing":
        "writing, editing, revising prose, articles, reports and written communication",

    "math":
        "mathematical calculations, equations, quantitative reasoning and numerical analysis",

    "planning":
        "planning projects, decomposing tasks, scheduling work, milestones and execution plans",

    "research":
        "research, investigation, evidence gathering, literature review and rigorous comparison",

    "data":
        "data analysis, dataset preprocessing, cleaning, statistics and data quality",

    "design":
        "designing interfaces, products, user experiences, interaction flows and visual systems",

    "operations":
        "operational procedures, workflows, process execution, coordination and process improvement",

    "qa_testing":
        "software quality assurance, test cases, validation, testing and verification",

    "translation":
        "translation between languages while preserving meaning, tone, terminology and context",

    "financial_analysis":
        "financial analysis, budgets, margins, cash flow, forecasting and financial comparisons",

    "database_design":
        "database schema design, relational databases, document databases and data modeling",

    "accessibility_review":
        "accessibility review, WCAG, inclusive interfaces and accessibility compliance",

    "fact_checking":
        "fact checking, source verification, claim validation and checking factual accuracy",

    "requirements_analysis":
        "requirements analysis, eliciting requirements, specifications, constraints and stakeholder needs",

    "incident_management":
        "incident response, triage, diagnosis, escalation, mitigation and service recovery",

    "experiment_design":
        "experiment design, hypotheses, controls, experimental methodology and evaluation",

    "risk_assessment":
        "risk assessment, identifying threats, likelihood, impact and mitigation",

    "documentation":
        "technical documentation, manuals, guides, release documentation and clear instructions",

    "customer_support_triage":
        "customer support triage, prioritizing customer issues, diagnosis and routing support requests",
}


def load_v31():
    spec = importlib.util.spec_from_file_location(
        "v31_v311b",
        V31_SCRIPT,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path):
    out = []

    if not path.exists():
        return out

    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))

    return out


def normalize_rows(x):
    return x / np.clip(
        np.linalg.norm(
            x,
            axis=1,
            keepdims=True,
        ),
        1e-8,
        None,
    )


v31 = load_v31()

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

proto_dir = OUT / "prototype_embeddings"
proto_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# BANK
# ============================================================

bank_obj = json.loads(
    BANK_PATH.read_text()
)

bank = [
    str(q)
    for q in bank_obj["queries"]
]

assert len(bank) == 512


# ============================================================
# EXISTING BANK EMBEDDINGS
# ============================================================

payload = torch.load(
    BANK_EMB_PATH,
    map_location="cpu",
    weights_only=False,
)

bank_cache = payload["embeddings"]
emb_dim = int(payload["dim"])


def bank_qvec(q):
    key = v31.query_text(q)

    if key not in bank_cache:
        raise KeyError(
            f"Missing bank embedding: {q}"
        )

    return bank_cache[key].float()


Q_raw = torch.stack([
    bank_qvec(q)
    for q in bank
]).numpy()


# ============================================================
# V3.1 QUERY-ONLY QUALITY
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
    (len(v31.TRAIN_ROLES),),
    0.5,
)


with torch.no_grad():
    quality_raw = torch.stack([
        m.score(
            dummy_turns,
            dummy_belief,
            torch.tensor(
                Q_raw,
                dtype=torch.float32,
            ),
        )
        for m in models
    ]).mean(0).numpy()


quality_order = np.argsort(
    quality_raw
)

quality_pct = np.empty(
    len(bank),
    dtype=float,
)

quality_pct[
    quality_order
] = np.linspace(
    0.0,
    1.0,
    len(bank),
)


# ============================================================
# ENCODE 20 ROLE PROTOTYPES WITH SAME QWEN MODEL
# ============================================================

role_names = list(
    ROLE_DESCRIPTIONS
)

prototype_texts = [
    (
        "A benign user query that would be especially informative "
        "for detecting whether a hidden agent specializes in "
        + ROLE_DESCRIPTIONS[r]
    )
    for r in role_names
]


prototype_group = SimpleNamespace(
    sid="__role_prototypes__",
    history=[],
    role_probs={
        r: 0.5
        for r in v31.TRAIN_ROLES
    },
    candidates=[
        SimpleNamespace(
            query=t
        )
        for t in prototype_texts
    ],
)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

proto_cache, proto_dim = (
    v31.load_or_build_embeddings(
        V14,
        proto_dir,
        [prototype_group],
        device,
    )
)

assert int(proto_dim) == emb_dim


P_raw = torch.stack([
    proto_cache[
        v31.query_text(t)
    ].float()
    for t in prototype_texts
]).numpy()


# ============================================================
# CENTER EMBEDDING SPACE
#
# Raw Qwen cosines were ~0.97 for almost everything.
# Remove dominant common direction before semantic matching.
# ============================================================

center = Q_raw.mean(
    axis=0,
    keepdims=True,
)

Q = normalize_rows(
    Q_raw - center
)

P = normalize_rows(
    P_raw - center
)


role_similarity = (
    P @ Q.T
)


# Convert similarity to percentile independently for each role.
role_pct = np.zeros_like(
    role_similarity
)

for i in range(
    len(role_names)
):
    order = np.argsort(
        role_similarity[i]
    )

    role_pct[i, order] = np.linspace(
        0.0,
        1.0,
        len(bank),
    )


# ============================================================
# ROLE × QUERY ASSIGNMENT
#
# Similarity dominates, but query quality still matters.
# Hungarian assignment guarantees 20 UNIQUE queries.
# ============================================================

combined = (
    0.70 * role_pct
    +
    0.30 * quality_pct[
        None,
        :
    ]
)


row_ind, col_ind = (
    linear_sum_assignment(
        -combined
    )
)


selected = []

for ri, qi in zip(
    row_ind,
    col_ind,
):
    selected.append(
        {
            "type":
            "role_anchor",

            "assigned_role":
            role_names[ri],

            "bank_index":
            int(qi),

            "query":
            bank[qi],

            "role_similarity":
            float(
                role_similarity[ri, qi]
            ),

            "role_similarity_percentile":
            float(
                role_pct[ri, qi]
            ),

            "quality_percentile":
            float(
                quality_pct[qi]
            ),

            "combined_score":
            float(
                combined[ri, qi]
            ),
        }
    )


selected_indices = {
    x["bank_index"]
    for x in selected
}


# ============================================================
# ADD FOUR HIGH-QUALITY GENERAL PROBES
#
# MMR using centered embeddings.
# ============================================================

for _ in range(
    GENERAL_COUNT
):
    best = None

    chosen_vectors = Q[
        sorted(
            selected_indices
        )
    ]

    for qi in np.argsort(
        -quality_raw
    )[:256]:
        qi = int(qi)

        if qi in selected_indices:
            continue

        max_sim = float(
            (
                chosen_vectors
                @ Q[qi]
            ).max()
        )

        novelty = (
            1.0 - max_sim
        )

        score = (
            0.65
            * quality_pct[qi]
            +
            0.35
            * novelty
        )

        if (
            best is None
            or score > best[0]
        ):
            best = (
                score,
                qi,
                novelty,
            )

    _, qi, novelty = best

    selected.append(
        {
            "type":
            "general",

            "assigned_role":
            None,

            "bank_index":
            qi,

            "query":
            bank[qi],

            "role_similarity":
            None,

            "role_similarity_percentile":
            None,

            "quality_percentile":
            float(
                quality_pct[qi]
            ),

            "combined_score":
            float(
                best[0]
            ),

            "novelty":
            float(
                novelty
            ),
        }
    )

    selected_indices.add(
        qi
    )


assert len(selected) == 24
assert len(selected_indices) == 24


# Stable order:
# 20 roles in canonical order, then 4 general.
selected.sort(
    key=lambda x: (
        1
        if x["type"] == "general"
        else 0,

        (
            role_names.index(
                x["assigned_role"]
            )
            if x["assigned_role"]
            in role_names
            else 999
        ),
    )
)


for i, x in enumerate(
    selected
):
    x["panel_index"] = i


# ============================================================
# PANEL CENTERED-DIVERSITY STATS
# ============================================================

panel_vec = Q[
    [
        x["bank_index"]
        for x in selected
    ]
]

pair_sims = []

for i in range(
    len(panel_vec)
):
    for j in range(
        i + 1,
        len(panel_vec),
    ):
        pair_sims.append(
            float(
                np.dot(
                    panel_vec[i],
                    panel_vec[j],
                )
            )
        )


# ============================================================
# TRAIN SYSTEMS + CANDIDATES
# ============================================================

manifest = json.loads(
    (
        V14
        / "manifest_privileged.json"
    ).read_text()
)

train_ids = [
    str(x["session_id"])
    for x in manifest["systems"]
    if x["split"] == "train"
]

assert len(train_ids) == 24


candidate_path = (
    OUT
    / "candidates_train_round_01.jsonl"
)

with candidate_path.open(
    "w"
) as f:
    for sid in train_ids:
        for x in selected:
            row = {
                "session_id":
                sid,

                "candidate_index":
                int(
                    x["panel_index"]
                ),

                "query":
                x["query"],

                "valid":
                True,

                "group_id":
                f"shared24_balanced::{sid}",

                "source":
                "V3.11b_balanced_panel",

                "bank_index":
                int(
                    x["bank_index"]
                ),

                "assigned_role":
                x["assigned_role"],
            }

            f.write(
                json.dumps(row)
                + "\n"
            )


# ============================================================
# COPY EXISTING CACHE FOR REUSE
# ============================================================

for name in [
    "target_public_cache.jsonl",
    "target_truth_cache_training_only.jsonl",
]:
    src = V14 / name
    dst = OUT / name

    if src.exists():
        shutil.copy2(
            src,
            dst,
        )


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

cached_pairs = {
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
    for x in public_rows
    if (
        str(
            x.get(
                "key",
                ""
            )
        )
        in truth_keys
    )
}


total_pairs = (
    len(train_ids)
    * len(selected)
)

cache_hits = sum(
    1
    for sid in train_ids
    for x in selected
    if (
        sid,
        x["query"],
    ) in cached_pairs
)

new_calls = (
    total_pairs
    - cache_hits
)


summary = {
    "panel_size":
    len(selected),

    "role_anchors":
    20,

    "general_probes":
    GENERAL_COUNT,

    "roles_covered":
    len({
        x["assigned_role"]
        for x in selected
        if x["assigned_role"]
    }),

    "mean_role_similarity_percentile":
    float(
        np.mean([
            x[
                "role_similarity_percentile"
            ]
            for x in selected
            if x[
                "role_similarity_percentile"
            ] is not None
        ])
    ),

    "minimum_role_similarity_percentile":
    float(
        min([
            x[
                "role_similarity_percentile"
            ]
            for x in selected
            if x[
                "role_similarity_percentile"
            ] is not None
        ])
    ),

    "mean_quality_percentile":
    float(
        np.mean([
            x[
                "quality_percentile"
            ]
            for x in selected
        ])
    ),

    "mean_centered_pairwise_cosine":
    float(
        np.mean(
            pair_sims
        )
    ),

    "max_centered_pairwise_cosine":
    float(
        np.max(
            pair_sims
        )
    ),

    "train_systems":
    len(train_ids),

    "total_pairs":
    total_pairs,

    "cache_hits":
    cache_hits,

    "new_target_calls":
    new_calls,
}


(
    OUT / "panel.json"
).write_text(
    json.dumps(
        {
            "summary":
            summary,

            "queries":
            selected,
        },
        indent=2,
    )
)


print("=" * 105)
print("V3.11b — ROLE-BALANCED SHARED 24-PROBE PANEL")
print("=" * 105)

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
print("=" * 105)
print("ROLE ANCHORS")
print("=" * 105)

for x in selected:
    if x["type"] != "role_anchor":
        continue

    print(
        f"{x['panel_index']:02d} | "
        f"{x['assigned_role']:24s} | "
        f"bank={x['bank_index']:03d} | "
        f"rolePct={x['role_similarity_percentile']:.3f} | "
        f"quality={x['quality_percentile']:.3f} | "
        f"{x['query']}"
    )


print()
print("=" * 105)
print("GENERAL PROBES")
print("=" * 105)

for x in selected:
    if x["type"] != "general":
        continue

    print(
        f"{x['panel_index']:02d} | "
        f"bank={x['bank_index']:03d} | "
        f"quality={x['quality_percentile']:.3f} | "
        f"{x['query']}"
    )


print()
print("=" * 105)
print("TARGET CALL ESTIMATE")
print("=" * 105)

print(
    f"TOTAL PAIRS : {total_pairs}"
)

print(
    f"CACHE HITS  : {cache_hits}"
)

print(
    f"NEW CALLS   : {new_calls}"
)

print()
print(
    "NO TARGET CALLS WERE MADE."
)

print(
    "SAVED:",
    OUT / "panel.json",
)
