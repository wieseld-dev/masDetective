import json
import math
import re
import statistics
from pathlib import Path
from collections import defaultdict

ROOT = Path(
    "artifacts/HIGHEST_CLEAN_AUC/"
    "experiments/15_LONG_6CLEAN_3HYBRID_6CLEAN_10HYBRID_V1/"
    "data/run"
)

ROUNDS = range(1, 26)


def load(path):
    if not path.exists():
        return []
    return [
        json.loads(x)
        for x in path.read_text().splitlines()
        if x.strip()
    ]


def sid(x):
    return str(
        x.get("session_id", x.get("system_id", ""))
    )


def extract_target(row):
    p = str(row.get("prompt", ""))

    m = re.search(
        r"TARGET CANDIDATE CAPABILITY:\s*([^\n]+)",
        p
    )

    return m.group(1).strip() if m else None


def get_probs(x):
    possible = [
        x.get("role_probs"),
        x.get("role_probabilities"),
        x.get("roles"),
    ]

    for k in [
        "belief",
        "public_belief",
        "state",
    ]:
        d = x.get(k)

        if isinstance(d, dict):
            possible.extend([
                d.get("role_probs"),
                d.get("role_probabilities"),
            ])

    for p in possible:
        if isinstance(p, dict):
            try:
                return {
                    str(k): float(v)
                    for k, v in p.items()
                }
            except Exception:
                pass

    raise RuntimeError(
        f"Cannot find role probs. Keys={sorted(x.keys())}"
    )


def logit(p):
    p = min(
        max(float(p), 1e-6),
        1.0 - 1e-6,
    )

    return math.log(
        p / (1.0 - p)
    )


def roc_auc(labels, scores):
    pos = [
        s for y, s in zip(labels, scores)
        if y == 1
    ]

    neg = [
        s for y, s in zip(labels, scores)
        if y == 0
    ]

    if not pos or not neg:
        return float("nan")

    wins = 0.0

    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5

    return wins / (
        len(pos) * len(neg)
    )


manifest = json.loads(
    (ROOT / "manifest_privileged.json").read_text()
)

truth = {
    str(x["session_id"]): set(map(str, x["roles"]))
    for x in manifest["systems"]
    if x.get("split") == "dev"
}


records = []

for rnd in ROUNDS:

    before = {
        sid(x): x
        for x in load(
            ROOT / f"states_dev_round_{rnd-1:02d}.jsonl"
        )
    }

    after = {
        sid(x): x
        for x in load(
            ROOT / f"states_dev_round_{rnd:02d}.jsonl"
        )
    }

    selected = {
        sid(x): x
        for x in load(
            ROOT / f"candidates_dev_round_{rnd:02d}.jsonl"
        )
    }

    scored = {
        sid(x): x
        for x in load(
            ROOT / f"scored_dev_round_{rnd:02d}.jsonl"
        )
    }

    responses = {
        sid(x): x
        for x in load(
            ROOT / f"responses_dev_round_{rnd:02d}.jsonl"
        )
    }

    for s in sorted(selected):

        if s not in before or s not in after:
            continue

        c = selected[s]

        target = extract_target(c)

        if not target:
            continue

        pb = get_probs(before[s])
        pa = get_probs(after[s])

        if target not in pb or target not in pa:
            continue

        present = int(
            target in truth[s]
        )

        p_before = float(
            pb[target]
        )

        p_after = float(
            pa[target]
        )

        delta_p = (
            p_after - p_before
        )

        delta_logit = (
            logit(p_after) -
            logit(p_before)
        )

        sc = scored.get(s, {})

        ev = sc.get(
            "evidence",
            {}
        )

        sensor_target = None

        if (
            isinstance(ev, dict)
            and target in ev
        ):
            sensor_target = float(
                ev[target]
            )

        activated = set(
            map(
                str,
                responses.get(
                    s,
                    {}
                ).get(
                    "selected_roles_training_only",
                    []
                )
            )
        )

        records.append({
            "round": rnd,
            "system": s,
            "target": target,
            "present": present,
            "p_before": p_before,
            "p_after": p_after,
            "delta_p": delta_p,
            "delta_logit": delta_logit,
            "sensor_target": sensor_target,
            "probe_style": c.get(
                "probe_style",
                "?"
            ),
            "target_activated": (
                target in activated
            ),
            "activated": sorted(
                activated
            ),
            "query": c.get(
                "query",
                ""
            ),
        })


present = [
    x for x in records
    if x["present"] == 1
]

absent = [
    x for x in records
    if x["present"] == 0
]


def mean(xs):
    return (
        statistics.mean(xs)
        if xs else float("nan")
    )


def median(xs):
    return (
        statistics.median(xs)
        if xs else float("nan")
    )


print("=" * 120)
print("TARGET-PRESENCE DISCRIMINATION AUDIT")
print("=" * 120)

print()
print("TOTAL =", len(records))
print(
    "PRESENT TARGET =",
    len(present)
)
print(
    "ABSENT TARGET  =",
    len(absent)
)


print()
print("=" * 120)
print("1. TARGET BELIEF MOVEMENT")
print("=" * 120)

for name, xs in [
    ("PRESENT", present),
    ("ABSENT", absent),
]:
    ds = [
        x["delta_p"]
        for x in xs
    ]

    dls = [
        x["delta_logit"]
        for x in xs
    ]

    up = sum(
        d > 0
        for d in ds
    )

    down = sum(
        d < 0
        for d in ds
    )

    zero = len(ds) - up - down

    print()
    print(name)
    print(
        f"  mean ΔP       = {mean(ds):+.6f}"
    )
    print(
        f"  median ΔP     = {median(ds):+.6f}"
    )
    print(
        f"  mean Δlogit   = {mean(dls):+.6f}"
    )
    print(
        f"  median Δlogit = {median(dls):+.6f}"
    )
    print(
        f"  up / zero / down = "
        f"{up}/{zero}/{down}"
    )


labels = [
    x["present"]
    for x in records
]

delta_scores = [
    x["delta_logit"]
    for x in records
]

print()
print(
    "AUC using TARGET Δlogit to distinguish "
    "PRESENT vs ABSENT =",
    f"{roc_auc(labels, delta_scores):.6f}"
)


print()
print("=" * 120)
print("2. RAW SENSOR TARGET SCORE")
print("=" * 120)

sensor_records = [
    x for x in records
    if x["sensor_target"] is not None
]

sensor_present = [
    x["sensor_target"]
    for x in sensor_records
    if x["present"] == 1
]

sensor_absent = [
    x["sensor_target"]
    for x in sensor_records
    if x["present"] == 0
]

print(
    f"PRESENT mean  = {mean(sensor_present):+.6f}"
)
print(
    f"PRESENT median= {median(sensor_present):+.6f}"
)
print(
    f"ABSENT mean   = {mean(sensor_absent):+.6f}"
)
print(
    f"ABSENT median = {median(sensor_absent):+.6f}"
)

sensor_auc = roc_auc(
    [x["present"] for x in sensor_records],
    [x["sensor_target"] for x in sensor_records],
)

print(
    "AUC using RAW TARGET SENSOR evidence =",
    f"{sensor_auc:.6f}"
)


print()
print("=" * 120)
print("3. DESIRED SIGN ACCURACY")
print("=" * 120)

correct = 0

for x in records:

    # Desired:
    # present -> target belief rises
    # absent  -> target belief falls
    good = (
        x["delta_logit"] > 0
        if x["present"]
        else x["delta_logit"] < 0
    )

    correct += int(good)

print(
    f"Correct direction = "
    f"{correct}/{len(records)} "
    f"({correct/len(records):.1%})"
)

if present:
    pgood = sum(
        x["delta_logit"] > 0
        for x in present
    )

    print(
        "PRESENT rises = "
        f"{pgood}/{len(present)} "
        f"({pgood/len(present):.1%})"
    )

if absent:
    agood = sum(
        x["delta_logit"] < 0
        for x in absent
    )

    print(
        "ABSENT falls  = "
        f"{agood}/{len(absent)} "
        f"({agood/len(absent):.1%})"
    )


print()
print("=" * 120)
print("4. PRESENT TARGET — ACTIVATION EFFECT")
print("=" * 120)

for activated in [True, False]:

    xs = [
        x for x in present
        if x["target_activated"] == activated
    ]

    if not xs:
        continue

    print(
        f"target_activated={activated}: "
        f"n={len(xs)} "
        f"meanΔlogit="
        f"{mean([x['delta_logit'] for x in xs]):+.6f} "
        f"positive="
        f"{sum(x['delta_logit'] > 0 for x in xs)}/{len(xs)}"
    )


print()
print("=" * 120)
print("5. BY PROBE STYLE")
print("=" * 120)

styles = sorted(
    set(
        x["probe_style"]
        for x in records
    )
)

for style in styles:

    xs = [
        x for x in records
        if x["probe_style"] == style
    ]

    if not xs:
        continue

    aucv = roc_auc(
        [x["present"] for x in xs],
        [x["delta_logit"] for x in xs],
    )

    print(
        f"{style:12s} "
        f"n={len(xs):3d} "
        f"AUC={aucv:.6f} "
        f"meanPresent="
        f"{mean([x['delta_logit'] for x in xs if x['present']]):+.4f} "
        f"meanAbsent="
        f"{mean([x['delta_logit'] for x in xs if not x['present']]):+.4f}"
    )


print()
print("=" * 120)
print("6. PER TARGET ROLE")
print("=" * 120)

by_role = defaultdict(list)

for x in records:
    by_role[x["target"]].append(x)

for role in sorted(by_role):

    xs = by_role[role]

    p = [
        x["delta_logit"]
        for x in xs
        if x["present"]
    ]

    a = [
        x["delta_logit"]
        for x in xs
        if not x["present"]
    ]

    print(
        f"{role:26s} "
        f"n={len(xs):2d} "
        f"present_n={len(p):2d} "
        f"absent_n={len(a):2d} "
        f"Pmean={mean(p):+.4f} "
        f"Amean={mean(a):+.4f}"
    )


print()
print("=" * 120)
print("7. WORST FALSE POSITIVES — ABSENT TARGET RISES")
print("=" * 120)

fps = sorted(
    [
        x for x in absent
        if x["delta_logit"] > 0
    ],
    key=lambda x: x["delta_logit"],
    reverse=True,
)

for x in fps[:15]:

    print(
        f"R{x['round']:02d} "
        f"{x['system']:40s} "
        f"TARGET={x['target']:24s} "
        f"Δlogit={x['delta_logit']:+.4f} "
        f"P:{x['p_before']:.4f}->{x['p_after']:.4f} "
        f"ACT={','.join(x['activated']) or 'NONE'}"
    )

    print(
        "   Q:",
        x["query"]
    )


print()
print("=" * 120)
print("8. WORST FALSE NEGATIVES — PRESENT TARGET FALLS")
print("=" * 120)

fns = sorted(
    [
        x for x in present
        if x["delta_logit"] < 0
    ],
    key=lambda x: x["delta_logit"],
)

for x in fns[:15]:

    print(
        f"R{x['round']:02d} "
        f"{x['system']:40s} "
        f"TARGET={x['target']:24s} "
        f"Δlogit={x['delta_logit']:+.4f} "
        f"P:{x['p_before']:.4f}->{x['p_after']:.4f} "
        f"TARGET_ACT={x['target_activated']}"
    )

    print(
        "   Q:",
        x["query"]
    )
