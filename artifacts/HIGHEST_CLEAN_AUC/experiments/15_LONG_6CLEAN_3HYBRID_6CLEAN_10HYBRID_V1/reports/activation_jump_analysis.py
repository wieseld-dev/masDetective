import json
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(
    "artifacts/HIGHEST_CLEAN_AUC/"
    "experiments/15_LONG_6CLEAN_3HYBRID_6CLEAN_10HYBRID_V1/"
    "data/run"
)

ROUNDS = [10, 16, 21, 24]


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
        x.get(
            "session_id",
            x.get("system_id", "")
        )
    )


def get_probs(x):
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
        f"Could not find role probabilities. Keys={sorted(x.keys())}"
    )


def pair_value(a, b):
    if a > b:
        return 1.0
    if a < b:
        return 0.0
    return 0.5


def auc(probs, truth):
    truth = set(truth)

    pos = [r for r in probs if r in truth]
    neg = [r for r in probs if r not in truth]

    vals = [
        pair_value(probs[t], probs[f])
        for t in pos
        for f in neg
    ]

    return sum(vals) / len(vals)


def extract_target(row):
    prompt = str(row.get("prompt", ""))

    m = re.search(
        r"TARGET CANDIDATE CAPABILITY:\s*([^\n]+)",
        prompt
    )

    return m.group(1).strip() if m else "UNKNOWN"


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

    chosen = {
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

    print()
    print("=" * 125)
    print(f"ROUND {rnd}")
    print("=" * 125)

    for s in sorted(chosen):

        c = chosen[s]
        r = responses.get(s, {})

        target = extract_target(c)

        truth = [
            str(x)
            for x in r.get(
                "truth_roles_training_only",
                []
            )
        ]

        activated = [
            str(x)
            for x in r.get(
                "selected_roles_training_only",
                []
            )
        ]

        pb = get_probs(before[s])
        pa = get_probs(after[s])

        auc_before = auc(pb, truth)
        auc_after = auc(pa, truth)
        delta = auc_after - auc_before

        target_true = target in truth
        target_activated = target in activated

        activated_true = [
            x for x in activated
            if x in truth
        ]

        if target_true and target_activated:
            mechanism = "DIRECT_TRUE_TARGET"
        elif (not target_true) and activated_true:
            mechanism = "ABSENT_TARGET_FALLBACK_TRUE"
        elif target_true and activated_true:
            mechanism = "TRUE_TARGET_OTHER_TRUE_ACTIVATED"
        elif activated_true:
            mechanism = "TRUE_ROLE_ACTIVATED"
        elif activated:
            mechanism = "FALSE_ROLE_ACTIVATED"
        else:
            mechanism = "NO_ACTIVATION_INFO"

        rec = {
            "round": rnd,
            "system": s,
            "target": target,
            "truth": truth,
            "activated": activated,
            "mechanism": mechanism,
            "auc_before": auc_before,
            "auc_after": auc_after,
            "delta": delta,
            "query": c.get("query", ""),
        }

        records.append(rec)

        print()
        print("SYSTEM      :", s)
        print("TARGET      :", target)
        print("TRUTH       :", ", ".join(truth))
        print(
            "ACTIVATED   :",
            ", ".join(activated) if activated else "NONE"
        )
        print("MECHANISM   :", mechanism)
        print(
            f"AUC         : {auc_before:.6f} -> "
            f"{auc_after:.6f}  "
            f"Δ={delta:+.6f}"
        )
        print(
            f"PAIR CHANGE : {delta*51:+.1f}/51"
        )
        print(
            "QUERY       :",
            c.get("query", "")
        )


print()
print()
print("=" * 125)
print("MECHANISM SUMMARY")
print("=" * 125)

groups = defaultdict(list)

for x in records:
    groups[x["mechanism"]].append(x["delta"])

for name, ds in sorted(groups.items()):

    positive = sum(
        d > 0 for d in ds
    )

    print(
        f"{name:35s} "
        f"n={len(ds):2d} "
        f"positive={positive:2d}/{len(ds):2d} "
        f"mean_delta={sum(ds)/len(ds):+.6f}"
    )


print()
print("=" * 125)
print("BIG POSITIVE CASES")
print("=" * 125)

for x in sorted(
    records,
    key=lambda z: z["delta"],
    reverse=True
):

    if x["delta"] <= 0:
        continue

    print(
        f"R{x['round']:02d} "
        f"{x['system']:40s} "
        f"Δ={x['delta']:+.6f} "
        f"TARGET={x['target']:24s} "
        f"ACT={','.join(x['activated']) or 'NONE':24s} "
        f"{x['mechanism']}"
    )
