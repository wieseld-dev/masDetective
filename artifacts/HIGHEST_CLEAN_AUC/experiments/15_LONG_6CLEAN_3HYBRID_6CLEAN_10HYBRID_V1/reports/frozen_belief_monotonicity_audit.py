import json
import math
import re
from pathlib import Path

from adaptive_v11_active.belief import FrozenBeliefBundle
from adaptive_v11_active.role_belief import role_only_predict
from adaptive_v11_active.schema import EvidenceFrame, PublicBeliefState


ROOT = Path(
    "artifacts/HIGHEST_CLEAN_AUC/"
    "experiments/15_LONG_6CLEAN_3HYBRID_6CLEAN_10HYBRID_V1/"
    "data/run"
)

BELIEF_PATH = Path(
    "artifacts/HIGHEST_CLEAN_AUC/"
    "models/frozen_belief.joblib"
)

GRID = [-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0]


def load(path):
    if not path.exists():
        return []

    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def sid(x):
    return str(
        x.get("session_id", x.get("system_id", ""))
    )


def target_of(row):
    prompt = str(row.get("prompt", ""))

    m = re.search(
        r"TARGET CANDIDATE CAPABILITY:\s*([^\n]+)",
        prompt
    )

    return m.group(1).strip() if m else None


def logit(p):
    p = min(max(float(p), 1e-8), 1.0 - 1e-8)
    return math.log(p / (1.0 - p))


bundle = FrozenBeliefBundle.load(BELIEF_PATH)

records = []


for rnd in range(1, 26):

    before_rows = load(
        ROOT / f"states_dev_round_{rnd-1:02d}.jsonl"
    )

    cand_rows = load(
        ROOT / f"candidates_dev_round_{rnd:02d}.jsonl"
    )

    resp_rows = load(
        ROOT / f"responses_dev_round_{rnd:02d}.jsonl"
    )

    score_rows = load(
        ROOT / f"scored_dev_round_{rnd:02d}.jsonl"
    )

    before = {
        sid(x): PublicBeliefState.from_public_dict(x)
        for x in before_rows
    }

    candidates = {
        sid(x): x
        for x in cand_rows
    }

    responses = {
        sid(x): x
        for x in resp_rows
    }

    scored = {
        sid(x): x
        for x in score_rows
    }

    for system_id, cand in candidates.items():

        if system_id not in before:
            continue

        if system_id not in responses:
            continue

        if system_id not in scored:
            continue

        target = target_of(cand)

        if target is None:
            continue

        state = before[system_id]
        resp = responses[system_id]
        sc = scored[system_id]

        v135_original = {
            str(k): float(v)
            for k, v in dict(
                sc.get("evidence", {})
            ).items()
        }

        v134_original = {
            str(k): float(v)
            for k, v in dict(
                sc.get("metadata", {}).get(
                    "evidence_v134",
                    {}
                )
            ).items()
        }

        if target not in v135_original:
            continue

        probs = []
        logits = []

        for value in GRID:

            v135 = dict(v135_original)
            v135[target] = value

            frame = EvidenceFrame(
                query=str(cand.get("query", "")),
                answer=str(resp.get("final_output", "")),
                v135=v135,
                v134=v134_original,
            )

            history = [
                *state.history,
                frame,
            ]

            _, role_probs = role_only_predict(
                bundle,
                history
            )

            p = float(role_probs[target])

            probs.append(p)
            logits.append(logit(p))

        nondecreasing = all(
            probs[i+1] >= probs[i] - 1e-12
            for i in range(len(probs) - 1)
        )

        strictly_positive_slope = (
            probs[-1] > probs[0]
        )

        records.append({
            "round": rnd,
            "system": system_id,
            "target": target,
            "actual_sensor": float(
                v135_original[target]
            ),
            "old_p": float(
                state.role_probs[target]
            ),
            "probs": probs,
            "logits": logits,
            "monotonic": nondecreasing,
            "positive_total_slope":
                strictly_positive_slope,
        })


print("=" * 140)
print("FROZEN BELIEF — TARGET V13.5 MONOTONICITY AUDIT")
print("=" * 140)

print()
print("CONTEXTS =", len(records))

mono = sum(
    x["monotonic"]
    for x in records
)

positive = sum(
    x["positive_total_slope"]
    for x in records
)

print(
    f"Fully monotonic increasing: "
    f"{mono}/{len(records)} "
    f"({mono/len(records):.1%})"
)

print(
    f"P(target | +8) > P(target | -8): "
    f"{positive}/{len(records)} "
    f"({positive/len(records):.1%})"
)


print()
print("=" * 140)
print("MEAN BELIEF CURVE")
print("=" * 140)

for j, value in enumerate(GRID):

    vals = [
        x["probs"][j]
        for x in records
    ]

    mean_p = (
        sum(vals) / len(vals)
    )

    print(
        f"V13.5={value:+5.1f} "
        f"mean P(target)={mean_p:.6f}"
    )


print()
print("=" * 140)
print("MEAN LOGIT CURVE")
print("=" * 140)

for j, value in enumerate(GRID):

    vals = [
        x["logits"][j]
        for x in records
    ]

    mean_l = (
        sum(vals) / len(vals)
    )

    print(
        f"V13.5={value:+5.1f} "
        f"mean logit(target)={mean_l:+.6f}"
    )


print()
print("=" * 140)
print("NON-MONOTONIC EXAMPLES")
print("=" * 140)

bad = [
    x for x in records
    if not x["monotonic"]
]

for x in bad[:25]:

    print()
    print(
        f"R{x['round']:02d} "
        f"{x['system']} "
        f"TARGET={x['target']} "
        f"actual={x['actual_sensor']:+.3f} "
        f"oldP={x['old_p']:.4f}"
    )

    print(
        "  " +
        "  ".join(
            f"{g:+.0f}:{p:.4f}"
            for g, p in zip(
                GRID,
                x["probs"]
            )
        )
    )


print()
print("=" * 140)
print("AVERAGE LOCAL STEP EFFECTS")
print("=" * 140)

for j in range(len(GRID) - 1):

    lo = GRID[j]
    hi = GRID[j+1]

    ds = [
        x["logits"][j+1] -
        x["logits"][j]
        for x in records
    ]

    mean_d = (
        sum(ds) / len(ds)
    )

    neg = sum(
        d < -1e-12
        for d in ds
    )

    zero = sum(
        abs(d) <= 1e-12
        for d in ds
    )

    pos = len(ds) - neg - zero

    print(
        f"{lo:+.0f} -> {hi:+.0f}: "
        f"mean Δlogit={mean_d:+.6f} "
        f"+/0/-={pos}/{zero}/{neg}"
    )
