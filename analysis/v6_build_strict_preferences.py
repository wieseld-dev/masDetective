import json, re, statistics
from pathlib import Path
from collections import Counter, defaultdict

SRC = Path("artifacts/v6_pref_filtered_v1/judged.jsonl")
OUT = Path("artifacts/v6_pref_strict_v2")
OUT.mkdir(parents=True, exist_ok=True)

KEEP = [
    2,6,8,9,10,16,23,24,26,
    27,32,33,35,37,39,43,44
]

CAPS = {
    "DIRECTION": 15,
    "META": 15,
    "DANGLING": 5,
}


def load(p):
    return [
        json.loads(x)
        for x in p.open()
        if x.strip()
    ]


def toks(s):
    return set(
        re.findall(
            r"[a-zA-Z0-9_]+",
            str(s).lower()
        )
    )


def jac(a,b):
    A,B=toks(a),toks(b)
    if not A or not B:
        return 0.0
    return len(A&B)/len(A|B)


def too_close(a,b):
    return jac(a,b) >= 0.88


rows=load(SRC)
assert len(rows)==6120

# --------------------------------------------------
# 1. Strict eligibility
# --------------------------------------------------

eligible=[]

for x in rows:
    if (
        int(x["topic_preservation"]) == 3
        and int(x["family_strength"]) == 3
        and int(x["isolation"]) == 3
        and int(x["plausibility"]) == 3
        and int(x["hardness"]) == 3
    ):
        y=dict(x)
        y["lexical_overlap"]=jac(
            y["gold"],
            y["rejected"]
        )
        eligible.append(y)

print("="*80)
print("V6 STRICT PREFERENCE BUILD")
print("="*80)
print("input           :",len(rows))
print("strict eligible :",len(eligible))

# --------------------------------------------------
# 2. Cell coverage BEFORE selection
# --------------------------------------------------

cells=defaultdict(list)

for x in eligible:
    cells[
        (
            int(x["gold_index"]),
            x["family"]
        )
    ].append(x)

print("\n===== STRICT CELL COVERAGE =====")

for gi in KEEP:
    print(
        f"G{gi:02d}",
        " ".join(
            f"{f[:3]}={len(cells[(gi,f)]):3d}"
            for f in ["DIRECTION","META","DANGLING"]
        )
    )

# --------------------------------------------------
# 3. Select close + diverse negatives
# --------------------------------------------------

selected=[]

for gi in KEEP:
    for fam in [
        "DIRECTION",
        "META",
        "DANGLING"
    ]:
        xs=list(cells[(gi,fam)])

        # Prefer semantically close negatives.
        xs.sort(
            key=lambda x: x["lexical_overlap"],
            reverse=True
        )

        chosen=[]

        for x in xs:
            # Don't select near duplicates of already chosen negatives.
            if any(
                too_close(
                    x["rejected"],
                    y["rejected"]
                )
                for y in chosen
            ):
                continue

            chosen.append(x)

            if len(chosen) >= CAPS[fam]:
                break

        selected.extend(chosen)

# --------------------------------------------------
# 4. Integrity
# --------------------------------------------------

by_g=Counter(
    int(x["gold_index"])
    for x in selected
)

by_f=Counter(
    x["family"]
    for x in selected
)

by_cell=Counter(
    (
        int(x["gold_index"]),
        x["family"]
    )
    for x in selected
)

print("\n===== SELECTED =====")
print("total:",len(selected))
print("family:",dict(by_f))

for gi in KEEP:
    print(
        f"G{gi:02d}",
        " ".join(
            f"{f[:3]}={by_cell[(gi,f)]:2d}"
            for f in ["DIRECTION","META","DANGLING"]
        ),
        f"TOTAL={by_g[gi]:2d}"
    )

assert all(
    x["family"] in CAPS
    for x in selected
)

assert all(
    by_cell[(gi,f)] <= CAPS[f]
    for gi in KEEP
    for f in CAPS
)

assert len({
    (
        int(x["gold_index"]),
        x["family"],
        x["rejected"].lower().strip()
    )
    for x in selected
}) == len(selected)

# --------------------------------------------------
# 5. DPO rows
# --------------------------------------------------

prefs=[]

for x in selected:
    prefs.append({
        "gold_index":
            int(x["gold_index"]),
        "family":
            x["family"],
        "prompt":
            x["prompt"],
        "chosen":
            x["gold"],
        "rejected":
            x["rejected"],

        # Important for V6 training:
        # weights are normalized later per Gold/family.
        "judge": {
            "family_strength":
                int(x["family_strength"]),
            "hardness":
                int(x["hardness"]),
            "lexical_overlap":
                float(x["lexical_overlap"]),
        }
    })

with (
    OUT/"preferences.jsonl"
).open("w") as f:
    for x in prefs:
        f.write(json.dumps(x)+"\n")

# --------------------------------------------------
# 6. Human review: weakest selected examples
# --------------------------------------------------

review=[]

for gi in KEEP:
    for fam in [
        "DIRECTION",
        "META",
        "DANGLING"
    ]:
        xs=[
            x for x in selected
            if int(x["gold_index"])==gi
            and x["family"]==fam
        ]

        if not xs:
            continue

        # Low lexical overlap = more likely topic drift,
        # so inspect weakest few.
        xs=sorted(
            xs,
            key=lambda x:x["lexical_overlap"]
        )

        review += [
            "",
            "="*100,
            f"G{gi:02d} / {fam} / n={len(xs)}",
            "="*100,
            "GOLD:",
            xs[0]["gold"],
        ]

        for x in xs[:3]:
            review += [
                "",
                f"lex={x['lexical_overlap']:.3f}",
                "NEG:",
                x["rejected"],
            ]

(OUT/"review.txt").write_text(
    "\n".join(review)+"\n"
)

summary={
    "input":len(rows),
    "strict_eligible":len(eligible),
    "selected":len(selected),
    "selected_by_family":dict(by_f),
    "selected_by_gold":dict(by_g),
    "caps":CAPS,
    "zero_strict_cells":[
        {
            "gold_index":gi,
            "family":f
        }
        for gi in KEEP
        for f in CAPS
        if len(cells[(gi,f)])==0
    ],
}

(OUT/"summary.json").write_text(
    json.dumps(summary,indent=2)
)

print("\n===== SUMMARY =====")
print(json.dumps(summary,indent=2))
print()
print("wrote:",OUT/"preferences.jsonl")
print("wrote:",OUT/"review.txt")
