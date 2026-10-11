import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from difflib import SequenceMatcher

SRC = Path("artifacts/v6_pref_tierA_v4/preferences.jsonl")
OUT = Path("artifacts/v6_pref_final_hp_v5")
OUT.mkdir(parents=True, exist_ok=True)

FINAL = OUT / "preferences.jsonl"
SUMMARY = OUT / "summary.json"
REVIEW = OUT / "review.txt"

GOLDS = [
    2,6,8,9,10,16,23,24,26,
    27,32,33,35,37,39,43,44
]

CAP = 4


def load(p):
    return [
        json.loads(x)
        for x in p.open(encoding="utf-8")
        if x.strip()
    ]


def norm(s):
    return " ".join(str(s).lower().split())


def words(s):
    return set(re.findall(r"[a-z0-9_]+", norm(s)))


def jac(a,b):
    A,B = words(a), words(b)
    if not A or not B:
        return 0.0
    return len(A&B)/len(A|B)


def near_dup(a,b):
    na, nb = norm(a), norm(b)

    if jac(na,nb) >= 0.78:
        return True

    if SequenceMatcher(None,na,nb).ratio() >= 0.88:
        return True

    return False


# ------------------------------------------------------------
# Strong META cues only.
# ------------------------------------------------------------

EXPLICIT_SYSTEM = re.compile(
    r"\byour (?:system|application|platform|model)\b",
    re.I,
)

SELF_STRUCTURE = re.compile(
    r"\byour (?:approach|method|process|procedure|protocol|"
    r"strategy|workflow|technique|techniques|tactic|tactics)\b",
    re.I,
)

BEHAVIOR_VERBS = (
    r"handle|manage|approach|assess|evaluate|ensure|address|"
    r"respond|perform|conduct|identify|detect|prioritize|"
    r"utilize|use|verify|process|support|facilitate|navigate|"
    r"deal|confront|execute|maintain|confirm|implement|resolve|"
    r"integrate|operate|apply|recognize|check|improve|assist|"
    r"help|preserve"
)

DIRECT_BEHAVIOR = re.compile(
    rf"^\s*(?:"
    rf"how do you|"
    rf"how would you|"
    rf"can you describe how you|"
    rf"can you explain how you"
    rf")\s+(?:(?:typically|usually)\s+)?"
    rf"(?:{BEHAVIOR_VERBS})\b",
    re.I,
)

WHAT_DO_YOU_DO = re.compile(
    r"^\s*what do you (?:typically )?do\b",
    re.I,
)

# Explicitly ambiguous/generic opinion formulations.
META_DENY = [
    re.compile(r"^\s*what do you believe\b", re.I),
    re.compile(r"^\s*what do you think\b", re.I),
    re.compile(r"^\s*what do you consider\b", re.I),

    # Usually asks for a domain explanation rather than revealing
    # system behavior.
    re.compile(
        r"^\s*how would you (?:typically )?explain\b",
        re.I,
    ),
]


def strong_meta(text):
    t = norm(text)

    if any(p.search(t) for p in META_DENY):
        return False, []

    cues = []

    if EXPLICIT_SYSTEM.search(t):
        cues.append("EXPLICIT_SYSTEM")

    if SELF_STRUCTURE.search(t):
        cues.append("SELF_STRUCTURE")

    if DIRECT_BEHAVIOR.search(t):
        cues.append("DIRECT_BEHAVIOR")

    if WHAT_DO_YOU_DO.search(t):
        cues.append("WHAT_DO_YOU_DO")

    return bool(cues), cues


rows = load(SRC)

assert len(rows) == 175, len(rows)

eligible = []
removed = Counter()

for x in rows:
    y = dict(x)

    if y["family"] == "DIRECTION":
        # Already survived strict Tier-A direction rules.
        y["final_cues"] = ["TIER_A_DIRECTION"]

    elif y["family"] == "META":
        ok, cues = strong_meta(y["rejected"])

        if not ok:
            removed["META:not_strong_self_reference"] += 1
            continue

        y["final_cues"] = cues

    else:
        continue

    y["lexical_overlap"] = jac(
        y["chosen"],
        y["rejected"],
    )

    eligible.append(y)


groups = defaultdict(list)

for x in eligible:
    groups[
        (
            int(x["gold_index"]),
            x["family"],
        )
    ].append(x)


selected = []

for gi in GOLDS:
    for fam in ["DIRECTION","META"]:
        xs = list(groups[(gi,fam)])

        # Prefer same-topic hard negatives.
        xs.sort(
            key=lambda x: x["lexical_overlap"],
            reverse=True,
        )

        chosen = []

        for x in xs:
            if any(
                near_dup(
                    x["rejected"],
                    z["rejected"],
                )
                for z in chosen
            ):
                continue

            chosen.append(x)

            if len(chosen) >= CAP:
                break

        selected.extend(chosen)


by_family = Counter(
    x["family"]
    for x in selected
)

by_gold = Counter(
    int(x["gold_index"])
    for x in selected
)

by_cell = Counter(
    (
        int(x["gold_index"]),
        x["family"],
    )
    for x in selected
)


final=[]

for x in selected:
    final.append({
        "gold_index":
            int(x["gold_index"]),
        "family":
            x["family"],
        "prompt":
            x["prompt"],
        "chosen":
            x["chosen"],
        "rejected":
            x["rejected"],
        "final_cues":
            x["final_cues"],
        "lexical_overlap":
            x["lexical_overlap"],
    })


with FINAL.open("w",encoding="utf-8") as f:
    for x in final:
        f.write(
            json.dumps(x,ensure_ascii=False)
            + "\n"
        )


summary = {
    "tier_a_input": len(rows),
    "final_eligible": len(eligible),
    "selected": len(final),
    "selected_by_family":
        dict(by_family),
    "selected_by_gold":
        dict(by_gold),
    "cells": {
        f"G{g:02d}:{fam}":
            by_cell[(g,fam)]
        for g in GOLDS
        for fam in ["DIRECTION","META"]
    },
    "zero_cells": [
        f"G{g:02d}:{fam}"
        for g in GOLDS
        for fam in ["DIRECTION","META"]
        if by_cell[(g,fam)] == 0
    ],
    "removed":
        dict(removed),
    "cap_per_cell":
        CAP,
}

SUMMARY.write_text(
    json.dumps(summary,indent=2),
    encoding="utf-8",
)


# Print every final pair: this should now be small enough
# for final human verification.
lines=[]

for gi in GOLDS:
    for fam in ["DIRECTION","META"]:
        xs=[
            x for x in final
            if x["gold_index"] == gi
            and x["family"] == fam
        ]

        if not xs:
            continue

        lines += [
            "",
            "="*100,
            f"G{gi:02d} / {fam} / n={len(xs)}",
            "="*100,
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
                " | ".join(x["final_cues"]),
            ]


REVIEW.write_text(
    "\n".join(lines)+"\n",
    encoding="utf-8",
)

print("="*80)
print("V6 FINAL HIGH-PRECISION DATASET")
print("="*80)
print(json.dumps(summary,indent=2))
print()
print("preferences:",FINAL)
print("review     :",REVIEW)
