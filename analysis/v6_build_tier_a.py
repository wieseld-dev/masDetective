import json
import re
from pathlib import Path
from collections import Counter, defaultdict

SRC = Path("artifacts/v6_pref_strict_v2/preferences.jsonl")
OUT = Path("artifacts/v6_pref_tierA_v4")
OUT.mkdir(parents=True, exist_ok=True)

FINAL = OUT / "preferences.jsonl"
SUMMARY = OUT / "summary.json"
REVIEW = OUT / "review.txt"

GOLDS = [2,6,8,9,10,16,23,24,26,27,32,33,35,37,39,43,44]

CAP = {
    "DIRECTION": 6,
    "META": 6,
}

DIRECTION_PATTERNS = [
    r"\bhow can i (?:help|assist|support)\b",
    r"\bhow may i (?:help|assist|support)\b",

    r"\bwould you like (?:me )?(?:to )?"
    r"(?:help|clarify|explain|discuss|explore|address)",

    r"\bwhat\b.*\bdo you need\b",
    r"\bwhich\b.*\bdo you need\b",

    r"\bwhat\b.*\bwould you like\b",
    r"\bwhich\b.*\bwould you like\b",

    r"\bwhat\b.*\bdo you want\b",
    r"\bwhich\b.*\bdo you want\b",

    r"\bwhat\b.*\bare you (?:struggling|facing|unsure|uncertain|confused)\b",
    r"\bwhich\b.*\bare you (?:struggling|facing|unsure|uncertain|confused)\b",

    r"\bwhat\b.*\bis unclear to you\b",
    r"\bwhich\b.*\bis unclear to you\b",

    r"\bwhat\b.*\bdo you find (?:difficult|challenging|confusing|unclear)\b",

    r"\bwhat information do you require\b",
    r"\bwhat details do you require\b",

    r"\bwhat\b.*\bshould i (?:help|clarify|explain|address|focus)\b",

    r"\bwhat\b.*\bcan i (?:help|clarify|explain|address)\b",

    r"\bwhat\b.*\bare you interested in\b",
]

META_PATTERNS = [
    # Explicit system/application self-reference.
    r"\byour system\b",
    r"\byour application\b",
    r"\byour platform\b",
    r"\byour model\b",

    # Explicit method/process belonging to the addressee.
    r"\byour approach\b",
    r"\byour method\b",
    r"\byour process\b",
    r"\byour procedure\b",
    r"\byour protocol\b",
    r"\byour strategy\b",
    r"\byour techniques?\b",
    r"\byour tactics?\b",

    # Addressee behavior/capability.
    r"^\s*how do you\b",
    r"^\s*how would you\b",
    r"^\s*how can you\b",
    r"^\s*how does your\b",

    r"^\s*what do you\b",
    r"^\s*what does your\b",

    r"\bdo you have (?:experience|methods?|a way|an approach|a process|tools?)\b",

    r"\bdoes your (?:system|application|platform|model)\b",

    r"\bcan you describe how you\b",
    r"\bcan you explain how you\b",
    r"\bcan you describe your\b",
    r"\bcan you explain your\b",

    r"\bhow do you typically\b",
    r"\bhow do you usually\b",
]

# These are strong signs that a META candidate is actually
# a normal user/domain question rather than system-self inquiry.
META_USER_CENTRIC = [
    r"\bwhat should i\b",
    r"\bwhat do i\b",
    r"\bhow should i\b",
    r"\bhow can i\b",
    r"\bmy .* (?:planning|project|work|workflow|process|task)\b",
]


def load(path):
    return [
        json.loads(line)
        for line in path.open(encoding="utf-8")
        if line.strip()
    ]


def norm(s):
    return " ".join(str(s).lower().split())


def matches_any(text, patterns):
    return [
        p for p in patterns
        if re.search(p, text, flags=re.I)
    ]


def tokens(s):
    return set(re.findall(r"[a-z0-9_]+", s.lower()))


def jac(a, b):
    A, B = tokens(a), tokens(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def near_dup(a, b):
    return jac(a, b) >= 0.82


rows = load(SRC)

rows = [
    x for x in rows
    if x["family"] in {"DIRECTION", "META"}
]

assert len(rows) == 495, len(rows)

accepted = []
rejected_reason = Counter()

for x in rows:
    text = norm(x["rejected"])
    fam = x["family"]

    if fam == "DIRECTION":
        cues = matches_any(text, DIRECTION_PATTERNS)

        # Direction should not primarily interrogate system internals.
        meta_cues = matches_any(text, META_PATTERNS)

        if not cues:
            rejected_reason["DIRECTION:no_explicit_assistant_cue"] += 1
            continue

        if meta_cues and not re.search(
            r"\b(?:need|want|would you like|unsure|uncertain|"
            r"struggling|facing|unclear to you)\b",
            text,
        ):
            rejected_reason["DIRECTION:looks_meta"] += 1
            continue

    elif fam == "META":
        cues = matches_any(text, META_PATTERNS)

        if not cues:
            rejected_reason["META:no_explicit_self_cue"] += 1
            continue

        if matches_any(text, META_USER_CENTRIC):
            rejected_reason["META:user_centric"] += 1
            continue

    else:
        continue

    y = dict(x)
    y["tier_a_cues"] = cues
    y["lexical_overlap"] = jac(
        y["chosen"],
        y["rejected"],
    )

    accepted.append(y)


print("=" * 80)
print("V6 TIER-A HIGH PRECISION BUILD")
print("=" * 80)
print("input DIRECTION+META :", len(rows))
print("Tier-A eligible      :", len(accepted))
print("rejection reasons    :", dict(rejected_reason))


groups = defaultdict(list)

for x in accepted:
    groups[
        (
            int(x["gold_index"]),
            x["family"],
        )
    ].append(x)


selected = []

for gi in GOLDS:
    for fam in ["DIRECTION", "META"]:
        xs = list(groups[(gi, fam)])

        # Prefer closest same-topic negatives.
        xs.sort(
            key=lambda z: z["lexical_overlap"],
            reverse=True,
        )

        chosen = []

        for x in xs:
            if any(
                near_dup(
                    x["rejected"],
                    y["rejected"],
                )
                for y in chosen
            ):
                continue

            chosen.append(x)

            if len(chosen) >= CAP[fam]:
                break

        selected.extend(chosen)


by_family = Counter(x["family"] for x in selected)
by_gold = Counter(int(x["gold_index"]) for x in selected)
by_cell = Counter(
    (int(x["gold_index"]), x["family"])
    for x in selected
)


print("\n===== CELL COUNTS =====")

for gi in GOLDS:
    print(
        f"G{gi:02d}",
        f"DIR={by_cell[(gi,'DIRECTION')]:2d}",
        f"META={by_cell[(gi,'META')]:2d}",
        f"TOTAL={by_gold[gi]:2d}",
    )


final = []

for x in selected:
    final.append({
        "gold_index": int(x["gold_index"]),
        "family": x["family"],
        "prompt": x["prompt"],
        "chosen": x["chosen"],
        "rejected": x["rejected"],
        "tier_a_cues": x["tier_a_cues"],
        "lexical_overlap": x["lexical_overlap"],
    })


with FINAL.open("w", encoding="utf-8") as f:
    for x in final:
        f.write(
            json.dumps(x, ensure_ascii=False)
            + "\n"
        )


summary = {
    "source_direction_meta": len(rows),
    "tier_a_eligible": len(accepted),
    "selected": len(final),
    "selected_by_family": dict(by_family),
    "selected_by_gold": dict(by_gold),
    "cells": {
        f"G{g:02d}:{fam}":
            by_cell[(g, fam)]
        for g in GOLDS
        for fam in ["DIRECTION", "META"]
    },
    "zero_cells": [
        f"G{g:02d}:{fam}"
        for g in GOLDS
        for fam in ["DIRECTION", "META"]
        if by_cell[(g, fam)] == 0
    ],
    "rejection_reasons": dict(rejected_reason),
    "cap_per_cell": CAP,
}

SUMMARY.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)


# Print EVERY selected candidate in a compact review.
# Dataset should now be small enough to inspect manually.
lines = []

for gi in GOLDS:
    for fam in ["DIRECTION", "META"]:
        xs = [
            x for x in final
            if x["gold_index"] == gi
            and x["family"] == fam
        ]

        if not xs:
            continue

        lines += [
            "",
            "=" * 100,
            f"G{gi:02d} / {fam} / n={len(xs)}",
            "=" * 100,
            "GOLD:",
            xs[0]["chosen"],
        ]

        for x in xs:
            lines += [
                "",
                f"lex={x['lexical_overlap']:.3f}",
                "NEG:",
                x["rejected"],
                "CUE:",
                " | ".join(x["tier_a_cues"]),
            ]


REVIEW.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print("\n===== SUMMARY =====")
print(json.dumps(summary, indent=2))
print()
print("preferences:", FINAL)
print("review     :", REVIEW)
