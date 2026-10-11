import json
import re
from pathlib import Path
from collections import defaultdict, Counter

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


def target(row):
    p = str(row.get("prompt", ""))
    m = re.search(
        r"TARGET CANDIDATE CAPABILITY:\s*([^\n]+)",
        p
    )
    return m.group(1).strip() if m else "UNKNOWN"


def probs(x):
    possible = [
        x.get("role_probs"),
        x.get("role_probabilities"),
        x.get("roles"),
    ]

    for k in ["belief", "public_belief", "state"]:
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


def pair(a, b):
    if a > b:
        return 1.0
    if a < b:
        return 0.0
    return 0.5


def auc(p, truth):
    truth = set(truth)

    pos = [r for r in p if r in truth]
    neg = [r for r in p if r not in truth]

    vals = [
        pair(p[t], p[f])
        for t in pos
        for f in neg
    ]

    return sum(vals) / len(vals)


manifest = json.loads(
    (ROOT / "manifest_privileged.json").read_text()
)

truth_map = {
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

    responses = {
        sid(x): x
        for x in load(
            ROOT / f"responses_dev_round_{rnd:02d}.jsonl"
        )
    }

    pool = defaultdict(list)

    for x in load(
        ROOT / f"candidate_pool_dev_round_{rnd:02d}.jsonl"
    ):
        pool[sid(x)].append(x)

    for s in sorted(selected):

        if s not in before or s not in after:
            continue

        tr = truth_map[s]

        c = selected[s]
        resp = responses.get(s, {})

        t = target(c)

        act = set(
            map(
                str,
                resp.get(
                    "selected_roles_training_only",
                    []
                )
            )
        )

        pb = probs(before[s])
        pa = probs(after[s])

        ab = auc(pb, tr)
        aa = auc(pa, tr)
        delta = aa - ab

        pool_targets = [
            target(x)
            for x in pool[s]
        ]

        true_pool_targets = [
            x for x in pool_targets
            if x in tr
        ]

        ordered = sorted(
            pb.items(),
            key=lambda z: z[1],
            reverse=True
        )

        rank = {
            r: i + 1
            for i, (r, _) in enumerate(ordered)
        }

        if t in tr and t in act:
            mech = "DIRECT_TRUE_TARGET_ACTIVATED"

        elif t in tr and (act & tr):
            mech = "TRUE_TARGET_OTHER_TRUE_ACTIVATED"

        elif t in tr:
            mech = "TRUE_TARGET_NO_TRUE_ACTIVATION"

        elif t not in tr and (act & tr):
            mech = "ABSENT_TARGET_FALLBACK_TRUE"

        elif act:
            mech = "ABSENT_TARGET_OTHER_ACTIVATION"

        else:
            mech = "NO_ACTIVATION"

        records.append({
            "round": rnd,
            "system": s,
            "target": t,
            "target_rank": rank.get(t),
            "truth": sorted(tr),
            "activated": sorted(act),
            "mechanism": mech,
            "delta": delta,
            "pair_delta": delta * 51,
            "probe_style": c.get("probe_style", "?"),
            "true_targets_in_pool": len(true_pool_targets),
            "pool_targets": pool_targets,
            "selected_target_true": t in tr,
            "query": c.get("query", ""),
        })


print("=" * 120)
print("ALL 25 ROUNDS — SELECTED-PROBE MECHANISM AUDIT")
print("=" * 120)

print()
print("TOTAL DECISIONS =", len(records))


def summarize(name, xs):
    if not xs:
        return

    ds = [x["delta"] for x in xs]

    pos = sum(d > 0 for d in ds)
    zero = sum(abs(d) < 1e-12 for d in ds)
    neg = sum(d < 0 for d in ds)

    print(
        f"{name:38s} "
        f"n={len(xs):3d} "
        f"+/0/-={pos:3d}/{zero:3d}/{neg:3d} "
        f"meanΔ={sum(ds)/len(ds):+.6f} "
        f"pairs={sum(d*51 for d in ds):+.1f}"
    )


print()
print("=" * 120)
print("MECHANISM SUMMARY")
print("=" * 120)

for mech in sorted(
    set(x["mechanism"] for x in records)
):
    summarize(
        mech,
        [
            x for x in records
            if x["mechanism"] == mech
        ]
    )


print()
print("=" * 120)
print("SIMPLE PRESENT-vs-ABSENT TARGET SUMMARY")
print("=" * 120)

summarize(
    "SELECTED TARGET IS TRUE",
    [
        x for x in records
        if x["selected_target_true"]
    ]
)

summarize(
    "SELECTED TARGET IS ABSENT",
    [
        x for x in records
        if not x["selected_target_true"]
    ]
)


print()
print("=" * 120)
print("CANDIDATE-POOL COVERAGE")
print("=" * 120)

n = len(records)

available = [
    x for x in records
    if x["true_targets_in_pool"] > 0
]

selected_true_when_available = [
    x for x in available
    if x["selected_target_true"]
]

print(
    "states with >=1 candidate targeting a TRUE role:",
    f"{len(available)}/{n}",
    f"({len(available)/n:.1%})"
)

if available:
    print(
        "Selector chose TRUE target when one was available:",
        f"{len(selected_true_when_available)}/{len(available)}",
        f"({len(selected_true_when_available)/len(available):.1%})"
    )

counts = Counter(
    x["true_targets_in_pool"]
    for x in records
)

print(
    "true-target candidates per G4 pool:",
    dict(sorted(counts.items()))
)


print()
print("=" * 120)
print("BY PROBE STYLE")
print("=" * 120)

for style in sorted(
    set(x["probe_style"] for x in records)
):
    summarize(
        style,
        [
            x for x in records
            if x["probe_style"] == style
        ]
    )


print()
print("=" * 120)
print("BY TARGET PRE-BELIEF RANK")
print("=" * 120)

buckets = [
    ("rank 1-3", 1, 3),
    ("rank 4-6", 4, 6),
    ("rank 7-10", 7, 10),
    ("rank 11-20", 11, 20),
]

for name, lo, hi in buckets:
    xs = [
        x for x in records
        if x["target_rank"] is not None
        and lo <= x["target_rank"] <= hi
    ]

    summarize(name, xs)


print()
print("=" * 120)
print("TOP 15 POSITIVE")
print("=" * 120)

for x in sorted(
    records,
    key=lambda z: z["delta"],
    reverse=True
)[:15]:

    print(
        f"R{x['round']:02d} "
        f"{x['system']:40s} "
        f"Δ={x['delta']:+.6f} "
        f"pairs={x['pair_delta']:+.1f} "
        f"rank={str(x['target_rank']):>2s} "
        f"target={x['target']:24s} "
        f"act={','.join(x['activated']) or 'NONE':28s} "
        f"{x['mechanism']}"
    )


print()
print("=" * 120)
print("TOP 15 NEGATIVE")
print("=" * 120)

for x in sorted(
    records,
    key=lambda z: z["delta"]
)[:15]:

    print(
        f"R{x['round']:02d} "
        f"{x['system']:40s} "
        f"Δ={x['delta']:+.6f} "
        f"pairs={x['pair_delta']:+.1f} "
        f"rank={str(x['target_rank']):>2s} "
        f"target={x['target']:24s} "
        f"act={','.join(x['activated']) or 'NONE':28s} "
        f"{x['mechanism']}"
    )
