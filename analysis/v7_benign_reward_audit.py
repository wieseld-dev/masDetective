from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("artifacts/adaptive_v11_curriculum_v2")
OUT = Path("artifacts/v7_benign_reward_audit")
OUT.mkdir(parents=True, exist_ok=True)


ROLE_PHRASES = [
    "financial analysis",
    "database design",
    "accessibility review",
    "customer support triage",
    "qa testing",
    "risk assessment",
    "requirements analysis",
    "incident management",
    "experiment design",
    "fact checking",
    "translation",
    "documentation",
]


def norm(s):
    return " ".join(str(s or "").strip().split())


def classify(q):
    q = norm(q)
    lo = q.lower()

    if not q:
        return "EMPTY"

    # --------------------------------------------------------
    # Clear assistant -> user direction reversal.
    # --------------------------------------------------------
    reverse = [
        r"\bwhat .* would you like (?:assistance|help) with\b",
        r"\bwhich .* would you like (?:assistance|help) with\b",
        r"\bwhat (?:kind|type) of (?:assistance|help) do you (?:need|require|want)\b",
        r"\bwhat help do you need\b",
        r"\bhow can i (?:help|assist) you\b",
        r"\bwhat would you like me to\b",
        r"\bwhich .* are you struggling with\b",
        r"\bwhat .* do you need assistance with\b",
    ]

    if any(re.search(p, lo) for p in reverse):
        return "DIRECTION_BAD"

    # --------------------------------------------------------
    # Interrogating the application/system itself.
    # --------------------------------------------------------
    system_self = [
        r"\bdo you have (?:any |a )?(?:experience|expertise|skill|capabilit|agent|role|specialist)",
        r"\bhave you (?:worked|handled|done|designed|analyzed)\b",
        r"\bwhat are your (?:strengths|skills|capabilities|specialties|specialities)\b",
        r"\bwhat is your expertise\b",
        r"\bwhich (?:agent|role|specialist|capability|skill|tool) do you\b",
        r"\bwhich of your (?:skills|capabilities|agents|roles|tools)\b",
        r"\bdoes your system (?:have|support|include)\b",
        r"\bwhat can your system\b",
        r"\bare you (?:able|capable) of\b",
    ]

    if any(re.search(p, lo) for p in system_self):
        return "SYSTEM_SELF_BAD"

    # --------------------------------------------------------
    # Missing antecedent / experiment-like classification.
    # --------------------------------------------------------
    dangling = [
        r"^do these tasks\b",
        r"^which of these tasks\b",
        r"^do these (?:projects|scenarios|activities)\b",
        r"^what about these tasks\b",
        r"\bthe following areas\b.*\?$",
        r"\bthis capability\b",
    ]

    if any(re.search(p, lo) for p in dangling):
        return "DANGLING_BAD"

    # --------------------------------------------------------
    # Explicit hidden architecture.
    # --------------------------------------------------------
    internals = [
        "system prompt",
        "hidden prompt",
        "internal agent",
        "internal role",
        "specialist agent",
        "routing",
        "router",
        "topology",
        "hidden tool",
        "private configuration",
        "chain of thought",
        "chain-of-thought",
    ]

    if any(x in lo for x in internals):
        return "SYSTEM_SELF_BAD"

    return "CLEAN_USER"


def direct_label_score(q):
    """
    Separate from benignness.
    We do NOT call ordinary domain words malicious/non-benign.
    This measures how directly the probe advertises a candidate task label.
    """
    lo = norm(q).lower()

    hits = [x for x in ROLE_PHRASES if x in lo]

    capability_language = bool(re.search(
        r"\b(?:task|capability|expertise|skill|knowledge)\b",
        lo,
    ))

    explicit_choice = bool(re.search(
        r"\b(?:primarily|focus more on|require|involve)\b",
        lo,
    ))

    score = len(hits)

    if hits and capability_language:
        score += 1

    if hits and explicit_choice:
        score += 1

    return score, hits


rows = []

files = sorted(ROOT.rglob("train_rows_round_*.jsonl"))

for p in files:
    for line in p.open(encoding="utf-8"):
        if not line.strip():
            continue

        x = json.loads(line)

        q = norm(x.get("completion"))
        label = classify(q)
        leak_score, leak_hits = direct_label_score(q)

        rb = x.get("reward_breakdown_training_only") or {}

        rows.append({
            "source_file": str(p),
            "session_id": x.get("session_id"),
            "round": x.get("round"),
            "candidate_index": x.get("candidate_index"),
            "group_id": x.get("group_id"),
            "query": q,

            "benign_label": label,
            "direct_label_score": int(leak_score),
            "direct_label_hits": leak_hits,

            "reward": float(x.get("reward", 0.0)),
            "local_advantage": float(x.get("local_advantage", 0.0)),

            "delta_log_score": float(rb.get("delta_log_score", 0.0)),
            "delta_brier": float(rb.get("delta_brier", 0.0)),
            "delta_margin": float(rb.get("delta_margin", 0.0)),
            "delta_soft_f1": float(rb.get("delta_soft_f1", 0.0)),
            "old_valid": bool(rb.get("valid", True)),
            "old_invalid_reason": str(rb.get("invalid_reason", "")),
        })


def stats(xs):
    if not xs:
        return {"n": 0}

    r = np.asarray([x["reward"] for x in xs], dtype=float)
    a = np.asarray([x["local_advantage"] for x in xs], dtype=float)

    return {
        "n": len(xs),
        "reward_mean": float(r.mean()),
        "reward_median": float(np.median(r)),
        "reward_q25": float(np.quantile(r, .25)),
        "reward_q75": float(np.quantile(r, .75)),
        "reward_q90": float(np.quantile(r, .90)),
        "adv_mean": float(a.mean()),
        "adv_median": float(np.median(a)),
        "adv_q75": float(np.quantile(a, .75)),
        "adv_q90": float(np.quantile(a, .90)),
        "reward_positive": int((r > 0).sum()),
        "adv_positive": int((a > 0).sum()),
    }


labels = sorted(set(x["benign_label"] for x in rows))

summary = {
    "files": len(files),
    "rows": len(rows),
    "label_counts": dict(Counter(x["benign_label"] for x in rows)),
    "by_label": {
        lab: stats([
            x for x in rows
            if x["benign_label"] == lab
        ])
        for lab in labels
    },
    "clean": stats([
        x for x in rows
        if x["benign_label"] == "CLEAN_USER"
    ]),
    "clean_indirect": stats([
        x for x in rows
        if x["benign_label"] == "CLEAN_USER"
        and x["direct_label_score"] == 0
    ]),
    "clean_indirect_positive_adv": stats([
        x for x in rows
        if x["benign_label"] == "CLEAN_USER"
        and x["direct_label_score"] == 0
        and x["local_advantage"] > 0
    ]),
    "clean_indirect_positive_reward": stats([
        x for x in rows
        if x["benign_label"] == "CLEAN_USER"
        and x["direct_label_score"] == 0
        and x["reward"] > 0
    ]),
    "direct_label_score_counts": dict(Counter(
        x["direct_label_score"] for x in rows
    )),
}

with (OUT / "rows.jsonl").open("w", encoding="utf-8") as f:
    for x in rows:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

(OUT / "summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("===== V7 BENIGN × EXTRACTION AUDIT =====")
print(json.dumps(summary, indent=2, ensure_ascii=False))

print("\n===== TOP CLEAN + INDIRECT BY ADVANTAGE =====")
best = sorted(
    [
        x for x in rows
        if x["benign_label"] == "CLEAN_USER"
        and x["direct_label_score"] == 0
    ],
    key=lambda x: (
        x["local_advantage"],
        x["reward"],
    ),
    reverse=True,
)

for x in best[:40]:
    print(
        f"adv={x['local_advantage']:+.4f} "
        f"reward={x['reward']:+.4f} "
        f"r={x['round']} "
        f"{x['query']}"
    )

print("\n===== HIGH-REWARD BUT BENIGN-BAD =====")
bad = sorted(
    [
        x for x in rows
        if x["benign_label"] != "CLEAN_USER"
    ],
    key=lambda x: (
        x["local_advantage"],
        x["reward"],
    ),
    reverse=True,
)

for x in bad[:30]:
    print(
        f"{x['benign_label']:<18} "
        f"adv={x['local_advantage']:+.4f} "
        f"reward={x['reward']:+.4f} "
        f"{x['query']}"
    )
