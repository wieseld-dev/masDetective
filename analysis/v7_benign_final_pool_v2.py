from __future__ import annotations

import json
import random
import re
from pathlib import Path
from collections import Counter


SRC = Path(
    "artifacts/v7_benign_final_pool_v1/"
    "valid_benign.jsonl"
)

OUT = Path(
    "artifacts/v7_benign_final_pool_v2"
)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260926
rng = random.Random(SEED)


# ============================================================
# FINAL PATCH:
# remove ONLY explicit assistant -> user direction reversals.
#
# IMPORTANT:
# KEEP:
#   Should I explain ...?
#   Should I provide ...?
#   What details should I provide ...?
#   Would more context help?
#   Do you know how ...?
#
# Those are valid user queries.
# ============================================================

DIRECTION_REVERSAL = [
    # Assistant offering help/guidance.
    r"^\s*do you need help\b",
    r"^\s*do you need guidance\b",
    r"^\s*do you require assistance\b",
    r"^\s*would you like help\b",
    r"^\s*would you like assistance\b",

    # Assistant offering to perform something.
    r"^\s*would you like me to\b",
    r"^\s*do you want me to\b",

    # Asking user what THEY want assistance with.
    r"^\s*which .* would you (?:most )?like assistance with\b",
    r"^\s*what .* would you (?:most )?like assistance with\b",

    r"^\s*which .* would you require assistance with\b",
    r"^\s*what .* would you require assistance with\b",

    r"^\s*which .* would you like me to\b",
    r"^\s*what .* would you like me to\b",

    # Assistant asking user what it should analyze/explore.
    r"^\s*which .* would you like me to "
    r"(?:analy[sz]e|review|explore|explain|check|assess)\b",

    r"^\s*what .* would you like me to "
    r"(?:analy[sz]e|review|explore|explain|check|assess)\b",

    # Assistant-style "what would you value/prefer?"
    r"^\s*which .* would you most value\b",
    r"^\s*what .* would you most value\b",

    r"^\s*which .* would you most appreciate\b",
    r"^\s*what .* would you most appreciate\b",

    # Explicit assistant-like intake.
    r"^\s*which software functionality are you seeking advice on\b",
    r"^\s*do you have any specific questions\b",
]


# A few explicit capability formulations missed by V1.
CAPABILITY_PATCH = [
    r"^\s*do you offer assistance with\b",

    # "Do you handle case studies involving X?"
    # High-confidence standing-capability survey.
    r"^\s*do you handle (?:case studies|tasks|projects) involving\b",
]


def norm(x):
    return " ".join(
        str(x or "")
        .strip()
        .lower()
        .split()
    )


def matched(patterns, text):
    return [
        p
        for p in patterns
        if re.search(
            p,
            text,
            flags=re.I,
        )
    ]


def read_jsonl(path):
    with path.open(
        encoding="utf-8"
    ) as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


rows = read_jsonl(SRC)

kept = []
rejected = []

reason_counts = Counter()

for x in rows:
    q = norm(
        x["query"]
    )

    direction = matched(
        DIRECTION_REVERSAL,
        q,
    )

    capability = matched(
        CAPABILITY_PATCH,
        q,
    )

    reasons = []

    if direction:
        reasons.append(
            "DIRECTION_REVERSAL"
        )

    if capability:
        reasons.append(
            "DIRECT_CAPABILITY_PATCH"
        )

    y = {
        **x,
        "final_validity_v2": {
            "valid_benign":
                len(reasons) == 0,

            "invalid_reasons":
                reasons,

            "matches": {
                "direction":
                    direction,

                "capability_patch":
                    capability,
            },
        },
    }

    if reasons:
        rejected.append(y)

        for r in reasons:
            reason_counts[r] += 1
    else:
        kept.append(y)


# ============================================================
# SESSION-HELDOUT SPLIT
#
# 80% train
# 10% dev
# 10% test
#
# No session may appear in more than one split.
# ============================================================

sessions = sorted({
    x["session_id"]
    for x in kept
})

rng.shuffle(
    sessions
)

n = len(sessions)

n_train = int(
    round(
        0.80 * n
    )
)

n_dev = int(
    round(
        0.10 * n
    )
)

train_sessions = set(
    sessions[:n_train]
)

dev_sessions = set(
    sessions[
        n_train:
        n_train + n_dev
    ]
)

test_sessions = set(
    sessions[
        n_train + n_dev:
    ]
)


def split_for(x):
    sid = x[
        "session_id"
    ]

    if sid in train_sessions:
        return "train"

    if sid in dev_sessions:
        return "dev"

    if sid in test_sessions:
        return "test"

    raise RuntimeError(
        f"unknown session {sid}"
    )


splits = {
    "train": [],
    "dev": [],
    "test": [],
}

for x in kept:
    s = split_for(x)

    splits[s].append({
        **x,
        "dataset_split":
            s,
    })


# ============================================================
# Sanity checks.
# ============================================================

assert (
    train_sessions.isdisjoint(
        dev_sessions
    )
)

assert (
    train_sessions.isdisjoint(
        test_sessions
    )
)

assert (
    dev_sessions.isdisjoint(
        test_sessions
    )
)

assert (
    sum(
        len(v)
        for v in splits.values()
    )
    == len(kept)
)


def role_counts(xs):
    c = Counter()

    for x in xs:
        for role in x.get(
            "selected_roles",
            []
        ):
            c[role] += 1

    return dict(
        sorted(
            c.items()
        )
    )


summary = {
    "input_v1_valid":
        len(rows),

    "final_valid":
        len(kept),

    "newly_rejected":
        len(rejected),

    "new_rejection_reasons":
        dict(
            sorted(
                reason_counts.items()
            )
        ),

    "sessions_total":
        len(sessions),

    "splits": {
        split: {
            "rows":
                len(xs),

            "sessions":
                len({
                    x["session_id"]
                    for x in xs
                }),

            "selected_role_positive_counts":
                role_counts(xs),
        }
        for split, xs
        in splits.items()
    },
}


def write_jsonl(path, xs):
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for x in xs:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )


write_jsonl(
    OUT / "valid_benign.jsonl",
    kept,
)

write_jsonl(
    OUT / "rejected_direction.jsonl",
    rejected,
)

for split, xs in splits.items():
    write_jsonl(
        OUT / f"{split}.jsonl",
        xs,
    )


(
    OUT / "summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


audit_keep = (
    kept
    if len(kept) <= 80
    else rng.sample(
        kept,
        80
    )
)

audit_reject = (
    rejected
    if len(rejected) <= 60
    else rng.sample(
        rejected,
        60
    )
)


print(
    "===== FINAL V2 + SESSION SPLIT ====="
)

print(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
)


print(
    "\n===== 60 RANDOM FINAL KEEP ====="
)

for i, x in enumerate(
    audit_keep[:60],
    1
):
    print(
        f"\n[{i:02d}] "
        f"selected={x.get('selected_roles')}"
    )

    print(
        "Q:",
        x["query"]
    )


print(
    "\n===== 40 NEWLY REJECTED ====="
)

for i, x in enumerate(
    audit_reject[:40],
    1
):
    print(
        f"\n[{i:02d}] "
        f"reasons="
        f"{x['final_validity_v2']['invalid_reasons']}"
    )

    print(
        "Q:",
        x["query"]
    )


print(
    "\nSaved:"
)

print(
    " ",
    OUT / "valid_benign.jsonl"
)

print(
    " ",
    OUT / "train.jsonl"
)

print(
    " ",
    OUT / "dev.jsonl"
)

print(
    " ",
    OUT / "test.jsonl"
)

print(
    " ",
    OUT / "summary.json"
)
