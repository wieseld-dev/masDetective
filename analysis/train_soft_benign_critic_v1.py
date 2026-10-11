#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

from sentence_transformers import SentenceTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


OUT = Path("artifacts/soft_benign_critic_v1")
OUT.mkdir(parents=True, exist_ok=True)

JUDGMENTS = Path(
    "artifacts/v5_heldout_multiseed/audit/query_judgments.jsonl"
)

PREFERENCES = Path(
    "artifacts/v6_pref_final_training/preferences_weighted.jsonl"
)

POSITIVES = Path(
    "artifacts/v4_training_ready/positives.jsonl"
)

EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 20260927


def read_jsonl(path):
    path = Path(path)

    if not path.exists():
        print("MISSING:", path)
        return []

    return [
        json.loads(line)
        for line in path.open(
            encoding="utf-8"
        )
        if line.strip()
    ]


def normalize(q):
    return re.sub(
        r"\s+",
        " ",
        str(q or "").strip(),
    ).lower()


examples = []


def add(q, y, source):
    q = str(q or "").strip()

    if not q:
        return

    examples.append({
        "query": q,
        "label": int(y),
        "source": source,
    })


# ============================================================
# 1. GPT / behavioral judgments
# ============================================================

for x in read_jsonl(JUDGMENTS):
    q = x.get("query", "")

    clean = (
        x.get("meta_label") == "NON_META"
        and x.get("user_like") == "USER_LIKE"
        and x.get("direction") == "USER_TO_SYSTEM"
        and x.get("naturalness") == "NATURAL"
    )

    clearly_bad = (
        x.get("meta_label") == "META"
        or x.get("user_like") == "ASSISTANT_LIKE"
        or x.get("direction") == "SYSTEM_TO_USER"
    )

    # Ambiguous cases are NOT forced into either class.
    if clean:
        add(q, 1, "heldout_judge_clean")

    elif clearly_bad:
        add(q, 0, "heldout_judge_bad")


# ============================================================
# 2. Manually approved Gold positives
# ============================================================

for x in read_jsonl(POSITIVES):
    q = (
        x.get("query")
        or x.get("completion")
        or x.get("chosen_query")
        or ""
    )

    add(
        q,
        1,
        "gold_positive",
    )


# ============================================================
# 3. Preference pairs
#
# Chosen = desired natural/user-like behavior.
# Rejected = deliberately inferior META / direction behavior.
# ============================================================

for x in read_jsonl(PREFERENCES):
    chosen = (
        x.get("chosen_query")
        or x.get("chosen")
        or ""
    )

    rejected = (
        x.get("rejected_query")
        or x.get("rejected")
        or ""
    )

    add(
        chosen,
        1,
        "preference_chosen",
    )

    add(
        rejected,
        0,
        "preference_rejected",
    )


# ============================================================
# Deduplicate.
#
# If the exact same normalized query has conflicting labels,
# REMOVE it rather than inventing truth.
# ============================================================

by_query = defaultdict(
    lambda: {
        "labels": set(),
        "texts": [],
        "sources": set(),
    }
)

for x in examples:
    k = normalize(
        x["query"]
    )

    by_query[k]["labels"].add(
        x["label"]
    )

    by_query[k]["texts"].append(
        x["query"]
    )

    by_query[k]["sources"].add(
        x["source"]
    )


clean_examples = []
conflicts = 0

for k, d in by_query.items():
    if len(d["labels"]) != 1:
        conflicts += 1
        continue

    clean_examples.append({
        "query":
            d["texts"][0],

        "label":
            next(
                iter(
                    d["labels"]
                )
            ),

        "sources":
            sorted(
                d["sources"]
            ),
    })


texts = [
    x["query"]
    for x in clean_examples
]

y = np.asarray(
    [
        x["label"]
        for x in clean_examples
    ],
    dtype=int,
)


print("=" * 90)
print("SOFT BENIGN CRITIC V1")
print("=" * 90)

print(
    "examples:",
    len(y),
)

print(
    "positive:",
    int(
        y.sum()
    ),
)

print(
    "negative:",
    int(
        (1 - y).sum()
    ),
)

print(
    "conflicting exact duplicates removed:",
    conflicts,
)


if len(np.unique(y)) != 2:
    raise RuntimeError(
        "Need both positive and negative examples"
    )


print()
print("Loading embedder...")

embedder = SentenceTransformer(
    EMBEDDER,
    device="cpu",
)


X = np.asarray(
    embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ),
    dtype=np.float32,
)


# ============================================================
# Out-of-fold evaluation.
# No example is scored by a model that trained on that example.
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)


oof = np.zeros(
    len(y),
    dtype=float,
)


for fold, (
    tr,
    va,
) in enumerate(
    cv.split(
        X,
        y,
    ),
    1,
):
    model = LogisticRegression(
        C=1.0,
        max_iter=5000,
        class_weight="balanced",
        random_state=
            SEED + fold,
    )

    model.fit(
        X[tr],
        y[tr],
    )

    oof[va] = (
        model.predict_proba(
            X[va]
        )[:, 1]
    )


auc = float(
    roc_auc_score(
        y,
        oof,
    )
)

ap = float(
    average_precision_score(
        y,
        oof,
    )
)

brier = float(
    brier_score_loss(
        y,
        oof,
    )
)

acc = float(
    accuracy_score(
        y,
        oof >= 0.5,
    )
)


pos_scores = oof[
    y == 1
]

neg_scores = oof[
    y == 0
]


summary = {
    "examples":
        int(
            len(y)
        ),

    "positive":
        int(
            y.sum()
        ),

    "negative":
        int(
            (1 - y).sum()
        ),

    "conflicting_exact_duplicates_removed":
        int(
            conflicts
        ),

    "oof_auc":
        auc,

    "oof_average_precision":
        ap,

    "oof_brier":
        brier,

    "oof_accuracy_at_0_5":
        acc,

    "positive_score_quantiles": {
        "q10":
            float(
                np.quantile(
                    pos_scores,
                    0.10,
                )
            ),

        "median":
            float(
                np.median(
                    pos_scores
                )
            ),

        "q90":
            float(
                np.quantile(
                    pos_scores,
                    0.90,
                )
            ),
    },

    "negative_score_quantiles": {
        "q10":
            float(
                np.quantile(
                    neg_scores,
                    0.10,
                )
            ),

        "median":
            float(
                np.median(
                    neg_scores
                )
            ),

        "q90":
            float(
                np.quantile(
                    neg_scores,
                    0.90,
                )
            ),
    },

    "score_semantics":
        (
            "continuous preference for "
            "ordinary benign user-like query; "
            "NOT a hard validity gate"
        ),
}


print()
print(
    json.dumps(
        summary,
        indent=2,
    )
)


# ============================================================
# Fit production critic on all labeled examples.
# ============================================================

final_model = LogisticRegression(
    C=1.0,
    max_iter=5000,
    class_weight="balanced",
    random_state=SEED,
)

final_model.fit(
    X,
    y,
)


joblib.dump(
    {
        "model":
            final_model,

        "embedder_name":
            EMBEDDER,

        "score_semantics":
            (
                "P(ordinary benign user-like query)"
            ),
    },
    OUT / "critic.joblib",
)


with (
    OUT
    / "training_examples.jsonl"
).open(
    "w",
    encoding="utf-8",
) as f:

    for x, p in zip(
        clean_examples,
        oof,
    ):
        row = dict(
            x
        )

        row[
            "oof_benign_score"
        ] = float(
            p
        )

        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


(
    OUT
    / "summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


print()
print(
    "WROTE:",
    OUT
    / "critic.joblib",
)

print(
    "WROTE:",
    OUT
    / "summary.json",
)
