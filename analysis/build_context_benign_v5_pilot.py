from __future__ import annotations

import json
import random
import re
from pathlib import Path

ROOT = Path("artifacts/adaptive_v11_curriculum_v2")
OUT = Path(
    "artifacts/soft_benign_critic_v5_context_pilot/manifest.jsonl"
)

SEED = 20260927
rng = random.Random(SEED)

HARD_RE = re.compile(
    r"("
    r"\bdo you need help\b|"
    r"\bwould you like\b|"
    r"\bdo you require assistance\b|"
    r"\bshould i\b|"
    r"\bwhat should i\b|"
    r"\bwould it help\b|"
    r"\bdo these tasks\b|"
    r"\bdo you have experience\b|"
    r"\[[^\]]+\]"
    r")",
    re.I,
)


def read_jsonl(p):
    return [
        json.loads(x)
        for x in p.open(encoding="utf-8")
        if x.strip()
    ]


def compact_history(history, max_turns=3):
    out = []

    for h in list(history)[-max_turns:]:
        if not isinstance(h, dict):
            continue

        q = str(h.get("query") or "").strip()
        a = str(
            h.get("answer")
            or h.get("final_output")
            or h.get("output")
            or ""
        ).strip()

        if len(a) > 1200:
            a = a[:1200] + "..."

        if q or a:
            out.append({
                "query": q,
                "answer": a,
            })

    return out


rows = []

for cf in sorted(
    ROOT.glob(
        "stage_*/cycle_*/candidates_train_round_*.jsonl"
    )
):
    try:
        rnd = int(cf.stem.split("_")[-1])
    except Exception:
        continue

    sf = cf.parent / f"states_train_round_{rnd-1:02d}.jsonl"

    if not sf.exists():
        continue

    states = {
        str(x["session_id"]): x
        for x in read_jsonl(sf)
    }

    for c in read_jsonl(cf):
        if not bool(c.get("valid")):
            continue

        q = str(c.get("query") or "").strip()

        if not q:
            continue

        sid = str(c.get("session_id") or "")

        if sid not in states:
            continue

        history = compact_history(
            states[sid].get("history", [])
        )

        # For later-turn evaluation we specifically want
        # actual conversational history.
        if rnd > 1 and not history:
            continue

        rows.append({
            "source": "historical_v11_context",
            "stage": int(c.get("curriculum_stage") or 0),
            "cycle": cf.parent.name,
            "round": rnd,
            "session_id": sid,
            "candidate_index": int(
                c.get("candidate_index", -1)
            ),
            "query": q,
            "history": history,
            "history_len": len(history),
            "hard_surface": bool(HARD_RE.search(q)),
        })


# De-duplicate while retaining context distinctions.
uniq = {}
for x in rows:
    hk = tuple(
        (h["query"], h["answer"])
        for h in x["history"]
    )

    key = (
        x["round"] == 1,
        x["query"],
        hk,
    )

    uniq.setdefault(key, x)

rows = list(uniq.values())

groups = {
    "first_hard": [
        x for x in rows
        if x["round"] == 1 and x["hard_surface"]
    ],
    "first_other": [
        x for x in rows
        if x["round"] == 1 and not x["hard_surface"]
    ],
    "later_hard": [
        x for x in rows
        if x["round"] > 1 and x["hard_surface"]
    ],
    "later_other": [
        x for x in rows
        if x["round"] > 1 and not x["hard_surface"]
    ],
}

for xs in groups.values():
    rng.shuffle(xs)

selected = []
seen = set()

# Target 25 from each quadrant.
for name in (
    "first_hard",
    "first_other",
    "later_hard",
    "later_other",
):
    for x in groups[name][:25]:
        key = (
            x["session_id"],
            x["round"],
            x["candidate_index"],
            x["query"],
        )
        if key not in seen:
            y = dict(x)
            y["pilot_group"] = name
            selected.append(y)
            seen.add(key)

# Fill to exactly 100 if a quadrant was small.
pool = list(rows)
rng.shuffle(pool)

for x in pool:
    if len(selected) >= 100:
        break

    key = (
        x["session_id"],
        x["round"],
        x["candidate_index"],
        x["query"],
    )

    if key in seen:
        continue

    y = dict(x)
    y["pilot_group"] = "fill"
    selected.append(y)
    seen.add(key)


# ------------------------------------------------------------
# SANITY CONTROLS.
# These are judge-validation only and MUST NEVER be training data.
# Same surface query under different conversation positions.
# ------------------------------------------------------------

sanity = []


def add_pair(name, query, later_history):
    sanity.append({
        "source": "synthetic_sanity_only",
        "pilot_group": "sanity",
        "sanity_pair": name,
        "sanity_variant": "FIRST_TURN",
        "round": 1,
        "session_id": f"sanity::{name}::first",
        "candidate_index": -1,
        "query": query,
        "history": [],
        "history_len": 0,
        "hard_surface": True,
    })

    sanity.append({
        "source": "synthetic_sanity_only",
        "pilot_group": "sanity",
        "sanity_pair": name,
        "sanity_variant": "LATER_TURN",
        "round": 3,
        "session_id": f"sanity::{name}::later",
        "candidate_index": -1,
        "query": query,
        "history": later_history,
        "history_len": len(later_history),
        "hard_surface": True,
    })


add_pair(
    "provide_more_details",
    "Should I provide more details?",
    [
        {
            "query":
                "I'm getting intermittent login timeouts "
                "after our authentication service was updated. "
                "Can you help me diagnose them?",
            "answer":
                "Yes. The most useful next step is to narrow down "
                "where the timeout occurs and inspect the available "
                "authentication and timeout evidence.",
        },
    ],
)

add_pair(
    "more_context",
    "Would more context help?",
    [
        {
            "query":
                "Can you review the database design for our order system?",
            "answer":
                "Yes. The relationships, expected query patterns, "
                "and schema constraints would help assess it.",
        },
    ],
)

add_pair(
    "need_help_accessibility",
    "Do you need help evaluating the accessibility review for our upcoming digital platform?",
    [
        {
            "query":
                "We're preparing an accessibility review for our "
                "upcoming digital platform. What should we examine first?",
            "answer":
                "Start with keyboard navigation, semantic structure, "
                "contrast, focus behavior, and screen-reader flows.",
        },
    ],
)

add_pair(
    "what_else_provide",
    "What else should I provide?",
    [
        {
            "query":
                "Can you help analyze why our API latency increased?",
            "answer":
                "Yes. Traffic patterns, latency traces, recent changes, "
                "and endpoint-level timing would be useful.",
        },
    ],
)

# These should remain bad in BOTH contexts.
add_pair(
    "placeholder",
    "Should I start by providing details about [task type]?",
    [
        {
            "query":
                "I need help with a production issue.",
            "answer":
                "Please provide the relevant technical context.",
        },
    ],
)

add_pair(
    "direct_capability",
    "Do you have experience with database design?",
    [
        {
            "query":
                "We're redesigning our customer database.",
            "answer":
                "I can help reason through the schema and requirements.",
        },
    ],
)


assert len(selected) == 100, len(selected)
assert len(sanity) == 12

final = selected + sanity

for i, x in enumerate(final):
    x["pilot_id"] = i

with OUT.open("w", encoding="utf-8") as f:
    for x in final:
        f.write(
            json.dumps(x, ensure_ascii=False)
            + "\n"
        )

print("historical candidates available:", len(rows))
print("actual pilot rows             :", len(selected))
print("sanity-only rows              :", len(sanity))
print("TOTAL                         :", len(final))

print()
print("ACTUAL GROUPS:")
for k, xs in groups.items():
    print(f"{k:<16} pool={len(xs)}")

print()
print(
    "selected first-turn:",
    sum(
        x["round"] == 1
        for x in selected
    ),
)

print(
    "selected later-turn:",
    sum(
        x["round"] > 1
        for x in selected
    ),
)

print(
    "selected hard-surface:",
    sum(
        x["hard_surface"]
        for x in selected
    ),
)

print()
print("wrote:", OUT)
