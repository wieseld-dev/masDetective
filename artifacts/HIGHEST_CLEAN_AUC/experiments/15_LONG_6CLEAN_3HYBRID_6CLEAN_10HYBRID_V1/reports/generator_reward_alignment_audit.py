import inspect
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import adaptive_v11_active.role_reward as rr


ROOT = Path("artifacts/v14_adaptive_role_v1")
ROUNDS = range(1, 8)


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
        x.get(
            "session_id",
            x.get("system_id", "")
        )
    )


def key(row):
    s = sid(row)

    action = row.get("action", {})
    if not isinstance(action, dict):
        action = {}

    idx = row.get(
        "candidate_index",
        action.get("variant", 0)
    )

    return s, int(idx)


def target_of(row):
    # Explicit field first, if one exists.
    for k in [
        "target_role",
        "target",
        "focus_role",
        "capability",
    ]:
        v = row.get(k)

        if isinstance(v, str) and v.strip():
            return v.strip()

    action = row.get("action")

    if isinstance(action, dict):
        for k in [
            "target_role",
            "role",
            "focus_role",
        ]:
            v = action.get(k)

            if isinstance(v, str) and v.strip():
                return v.strip()

    prompt = str(row.get("prompt", ""))

    patterns = [
        r"TARGET CANDIDATE CAPABILITY:\s*([^\n]+)",
        r"TARGET ROLE:\s*([^\n]+)",
        r"TARGET CAPABILITY:\s*([^\n]+)",
    ]

    for pat in patterns:
        m = re.search(
            pat,
            prompt,
            flags=re.I
        )

        if m:
            return m.group(1).strip()

    return None


def truth_of(*rows):
    for row in rows:
        if not isinstance(row, dict):
            continue

        for k in [
            "truth_roles_training_only",
            "truth_roles",
            "roles_training_only",
        ]:
            v = row.get(k)

            if isinstance(v, (list, tuple)):
                return set(map(str, v))

    return set()


def average_ranks(xs):
    xs = np.asarray(xs, dtype=float)

    order = np.argsort(
        xs,
        kind="mergesort"
    )

    ranks = np.empty(
        len(xs),
        dtype=float
    )

    i = 0

    while i < len(xs):
        j = i + 1

        while (
            j < len(xs)
            and xs[order[j]] == xs[order[i]]
        ):
            j += 1

        avg = 0.5 * (
            i + (j - 1)
        ) + 1.0

        ranks[
            order[i:j]
        ] = avg

        i = j

    return ranks


def pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) < 2:
        return float("nan")

    if np.std(a) < 1e-12:
        return float("nan")

    if np.std(b) < 1e-12:
        return float("nan")

    return float(
        np.corrcoef(a, b)[0, 1]
    )


def spearman(a, b):
    return pearson(
        average_ranks(a),
        average_ranks(b)
    )


def auc(labels, scores):
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


def mean(xs):
    return (
        float(np.mean(xs))
        if xs
        else float("nan")
    )


print("=" * 125)
print("ACTUAL GENERATOR REWARD SOURCE")
print("=" * 125)

print()
print("adaptive_role_reward:")
print(inspect.getsource(rr.adaptive_role_reward))

print()
print("role_log_score:")
print(inspect.getsource(rr.role_log_score))


records = []


for rnd in ROUNDS:

    cand_path = (
        ROOT /
        f"candidates_train_round_{rnd:02d}.jsonl"
    )

    response_path = (
        ROOT /
        f"responses_train_round_{rnd:02d}.jsonl"
    )

    score_path = (
        ROOT /
        f"scored_train_round_{rnd:02d}.jsonl"
    )

    train_path = (
        ROOT /
        f"grpo_train_round_{rnd:02d}.jsonl"
    )

    candidates = {
        key(x): x
        for x in load(cand_path)
    }

    responses = {
        key(x): x
        for x in load(response_path)
    }

    scored = {
        key(x): x
        for x in load(score_path)
    }

    training = {
        key(x): x
        for x in load(train_path)
    }

    print()
    print(
        f"[R{rnd:02d}] "
        f"candidates={len(candidates)} "
        f"responses={len(responses)} "
        f"scored={len(scored)} "
        f"grpo={len(training)}"
    )

    shared = sorted(
        set(candidates)
        & set(scored)
        & set(training)
    )

    print(
        f"[R{rnd:02d}] joined={len(shared)}"
    )

    for k in shared:

        c = candidates[k]
        sc = scored[k]
        tr = training[k]
        resp = responses.get(k, {})

        target = target_of(c)

        if target is None:
            continue

        truth = truth_of(
            sc,
            resp,
            c,
            tr,
        )

        if not truth:
            continue

        evidence = {
            str(role): float(value)
            for role, value in dict(
                sc.get(
                    "evidence",
                    {}
                )
            ).items()
        }

        if target not in evidence:
            continue

        target_present = int(
            target in truth
        )

        target_e = float(
            evidence[target]
        )

        # This is the binary-hypothesis direction we actually want:
        #
        # PRESENT -> high target evidence
        # ABSENT  -> low target evidence
        #
        # Therefore higher signed_target_evidence
        # is always "better discrimination".
        signed_target_evidence = (
            target_e
            if target_present
            else -target_e
        )

        reward = float(
            tr.get(
                "reward",
                float("nan")
            )
        )

        advantage = float(
            tr.get(
                "advantage",
                float("nan")
            )
        )

        breakdown = tr.get(
            "reward_breakdown_training_only",
            {}
        )

        records.append({
            "round": rnd,
            "system": k[0],
            "candidate_index": k[1],
            "target": target,
            "present": target_present,
            "target_evidence": target_e,
            "signed_target_evidence":
                signed_target_evidence,
            "correct_target_sign": int(
                (
                    target_present
                    and target_e > 0
                )
                or
                (
                    not target_present
                    and target_e < 0
                )
            ),
            "reward": reward,
            "advantage": advantage,
            "breakdown": breakdown,
            "query": c.get(
                "query",
                ""
            ),
        })


print()
print("=" * 125)
print("JOIN SUMMARY")
print("=" * 125)

print("TOTAL USABLE CANDIDATES =", len(records))

if not records:
    raise SystemExit(
        "ERROR: no joined training records. "
        "Inspect the R01-R07 file counts printed above."
    )


present = [
    x for x in records
    if x["present"]
]

absent = [
    x for x in records
    if not x["present"]
]


print(
    "PRESENT TARGET =",
    len(present)
)

print(
    "ABSENT TARGET  =",
    len(absent)
)


labels = [
    x["present"]
    for x in records
]

sensor_scores = [
    x["target_evidence"]
    for x in records
]


print()
print("=" * 125)
print("TARGET SENSOR DISCRIMINATION ON GENERATOR TRAIN DATA")
print("=" * 125)

print(
    "AUC target V13.5: PRESENT vs ABSENT =",
    f"{auc(labels, sensor_scores):.6f}"
)

print(
    "PRESENT mean target evidence =",
    f"{mean([x['target_evidence'] for x in present]):+.6f}"
)

print(
    "ABSENT mean target evidence  =",
    f"{mean([x['target_evidence'] for x in absent]):+.6f}"
)

print(
    "Correct target-evidence sign =",
    f"{sum(x['correct_target_sign'] for x in records)}/{len(records)} "
    f"({sum(x['correct_target_sign'] for x in records)/len(records):.1%})"
)


valid_reward = [
    x for x in records
    if math.isfinite(x["reward"])
]

valid_adv = [
    x for x in records
    if math.isfinite(x["advantage"])
]


print()
print("=" * 125)
print("REWARD ALIGNMENT WITH TARGET PRESENT/ABSENT DISCRIMINATION")
print("=" * 125)

for label, xs, field in [
    (
        "GENERATOR REWARD",
        valid_reward,
        "reward",
    ),
    (
        "ACTUAL GRPO ADVANTAGE",
        valid_adv,
        "advantage",
    ),
]:

    a = [
        x[field]
        for x in xs
    ]

    b = [
        x["signed_target_evidence"]
        for x in xs
    ]

    correct = [
        x[field]
        for x in xs
        if x["correct_target_sign"]
    ]

    wrong = [
        x[field]
        for x in xs
        if not x["correct_target_sign"]
    ]

    print()
    print(label)

    print(
        "  Pearson with desired signed target evidence =",
        f"{pearson(a, b):+.6f}"
    )

    print(
        "  Spearman with desired signed target evidence =",
        f"{spearman(a, b):+.6f}"
    )

    print(
        "  mean when target evidence has CORRECT sign =",
        f"{mean(correct):+.6f}"
    )

    print(
        "  mean when target evidence has WRONG sign   =",
        f"{mean(wrong):+.6f}"
    )


print()
print("=" * 125)
print("GRADIENT DIRECTION VS TARGET DISCRIMINATION")
print("=" * 125)

positive_adv = [
    x for x in valid_adv
    if x["advantage"] > 0
]

negative_adv = [
    x for x in valid_adv
    if x["advantage"] < 0
]


for name, xs in [
    ("ADVANTAGE > 0", positive_adv),
    ("ADVANTAGE < 0", negative_adv),
]:

    if not xs:
        continue

    good = sum(
        x["correct_target_sign"]
        for x in xs
    )

    print(
        f"{name:18s} "
        f"n={len(xs):4d} "
        f"correct_target_sign="
        f"{good}/{len(xs)} "
        f"({good/len(xs):.1%}) "
        f"mean_signed_target_evidence="
        f"{mean([x['signed_target_evidence'] for x in xs]):+.4f}"
    )


print()
print("=" * 125)
print("PER ROUND")
print("=" * 125)

for rnd in ROUNDS:

    xs = [
        x for x in records
        if x["round"] == rnd
    ]

    if not xs:
        continue

    rewards = [
        x["reward"]
        for x in xs
        if math.isfinite(x["reward"])
    ]

    signed = [
        x["signed_target_evidence"]
        for x in xs
        if math.isfinite(x["reward"])
    ]

    advs = [
        x["advantage"]
        for x in xs
        if math.isfinite(x["advantage"])
    ]

    signed_adv = [
        x["signed_target_evidence"]
        for x in xs
        if math.isfinite(x["advantage"])
    ]

    sensor_auc = auc(
        [x["present"] for x in xs],
        [x["target_evidence"] for x in xs],
    )

    reward_rho = spearman(
        rewards,
        signed,
    )

    adv_rho = spearman(
        advs,
        signed_adv,
    )

    print(
        f"R{rnd:02d} "
        f"n={len(xs):4d} "
        f"sensorAUC={sensor_auc:.4f} "
        f"reward_rho={reward_rho:+.4f} "
        f"adv_rho={adv_rho:+.4f}"
    )


print()
print("=" * 125)
print("REWARD BREAKDOWN KEYS")
print("=" * 125)

keys = sorted({
    k
    for x in records
    if isinstance(x["breakdown"], dict)
    for k in x["breakdown"]
})

print(keys)


for bk in keys:

    xs = []

    for x in records:

        bd = x["breakdown"]

        if not isinstance(bd, dict):
            continue

        v = bd.get(bk)

        try:
            v = float(v)
        except Exception:
            continue

        if not math.isfinite(v):
            continue

        xs.append(
            (
                v,
                x["signed_target_evidence"]
            )
        )

    if len(xs) < 2:
        continue

    vals = [x[0] for x in xs]
    sigs = [x[1] for x in xs]

    print(
        f"{bk:40s} "
        f"n={len(xs):4d} "
        f"pearson={pearson(vals, sigs):+.5f} "
        f"spearman={spearman(vals, sigs):+.5f}"
    )


print()
print("=" * 125)
print("TOP POSITIVE-ADVANTAGE BUT BAD TARGET DISCRIMINATION")
print("=" * 125)

bad_positive = sorted(
    [
        x for x in valid_adv
        if (
            x["advantage"] > 0
            and not x["correct_target_sign"]
        )
    ],
    key=lambda x: x["advantage"],
    reverse=True
)

for x in bad_positive[:20]:

    print(
        f"R{x['round']:02d} "
        f"ADV={x['advantage']:+.4f} "
        f"REWARD={x['reward']:+.4f} "
        f"{'PRESENT' if x['present'] else 'ABSENT ':7s} "
        f"TARGET={x['target']:24s} "
        f"E={x['target_evidence']:+.3f} "
        f"SIGNED={x['signed_target_evidence']:+.3f}"
    )

    print(
        "   Q:",
        x["query"]
    )


print()
print("=" * 125)
print("TOP NEGATIVE-ADVANTAGE DESPITE GOOD TARGET DISCRIMINATION")
print("=" * 125)

bad_negative = sorted(
    [
        x for x in valid_adv
        if (
            x["advantage"] < 0
            and x["correct_target_sign"]
        )
    ],
    key=lambda x: x["advantage"]
)

for x in bad_negative[:20]:

    print(
        f"R{x['round']:02d} "
        f"ADV={x['advantage']:+.4f} "
        f"REWARD={x['reward']:+.4f} "
        f"{'PRESENT' if x['present'] else 'ABSENT ':7s} "
        f"TARGET={x['target']:24s} "
        f"E={x['target_evidence']:+.3f} "
        f"SIGNED={x['signed_target_evidence']:+.3f}"
    )

    print(
        "   Q:",
        x["query"]
    )
