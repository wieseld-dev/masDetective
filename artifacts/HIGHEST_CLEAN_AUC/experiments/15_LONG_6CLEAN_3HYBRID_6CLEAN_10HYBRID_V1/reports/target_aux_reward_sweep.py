import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from adaptive_v11_active.reward import group_relative_advantages


ROOT = Path("artifacts/v14_adaptive_role_v1")
ROUNDS = range(1, 8)

LAMBDAS = [
    0.0,
    0.05,
    0.10,
    0.20,
    0.30,
    0.50,
    0.75,
    1.00,
]

TAUS = [
    1.0,
    2.0,
    4.0,
]


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


def key(x):
    action = x.get("action", {})

    if not isinstance(action, dict):
        action = {}

    idx = x.get(
        "candidate_index",
        action.get("variant", 0)
    )

    return sid(x), int(idx)


def target_of(row):
    for k in [
        "target_role",
        "target",
        "focus_role",
        "capability",
    ]:
        v = row.get(k)

        if isinstance(v, str) and v.strip():
            return v.strip()

    p = str(row.get("prompt", ""))

    for pat in [
        r"TARGET CANDIDATE CAPABILITY:\s*([^\n]+)",
        r"TARGET ROLE:\s*([^\n]+)",
        r"TARGET CAPABILITY:\s*([^\n]+)",
    ]:
        m = re.search(
            pat,
            p,
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


records = []


for rnd in ROUNDS:

    candidates = {
        key(x): x
        for x in load(
            ROOT / f"candidates_train_round_{rnd:02d}.jsonl"
        )
    }

    responses = {
        key(x): x
        for x in load(
            ROOT / f"responses_train_round_{rnd:02d}.jsonl"
        )
    }

    scored = {
        key(x): x
        for x in load(
            ROOT / f"scored_train_round_{rnd:02d}.jsonl"
        )
    }

    grpo = {
        key(x): x
        for x in load(
            ROOT / f"grpo_train_round_{rnd:02d}.jsonl"
        )
    }

    shared = (
        set(candidates)
        & set(scored)
        & set(grpo)
    )

    for k in shared:

        c = candidates[k]
        sc = scored[k]
        g = grpo[k]
        resp = responses.get(k, {})

        target = target_of(c)

        if target is None:
            continue

        truth = truth_of(
            sc,
            resp,
            c,
            g
        )

        if not truth:
            continue

        evidence = {
            str(r): float(v)
            for r, v in dict(
                sc.get("evidence", {})
            ).items()
        }

        if target not in evidence:
            continue

        present = (
            target in truth
        )

        e = float(
            evidence[target]
        )

        signed = (
            e if present
            else -e
        )

        records.append({
            "round": rnd,
            "system": k[0],
            "candidate": k[1],
            "target": target,
            "present": present,
            "evidence": e,
            "signed": signed,
            "correct_sign": signed > 0,
            "global_reward": float(
                g["reward"]
            ),
        })


groups = defaultdict(list)

for x in records:
    groups[
        (x["round"], x["system"])
    ].append(x)


print("=" * 125)
print("TARGET-AUXILIARY REWARD SWEEP")
print("=" * 125)

print()
print("ROWS   =", len(records))
print("GROUPS =", len(groups))


results = []


for tau in TAUS:

    for lam in LAMBDAS:

        positive_rows = []

        top_rows = []

        global_regrets = []

        changed_top = 0

        for group_key, xs in groups.items():

            old_best = max(
                range(len(xs)),
                key=lambda i:
                    xs[i]["global_reward"]
            )

            augmented = []

            for x in xs:

                target_aux = math.tanh(
                    x["signed"] / tau
                )

                new_reward = (
                    x["global_reward"]
                    + lam * target_aux
                )

                augmented.append(
                    new_reward
                )

            new_best = max(
                range(len(xs)),
                key=lambda i:
                    augmented[i]
            )

            if new_best != old_best:
                changed_top += 1

            chosen = xs[new_best]

            chosen = dict(chosen)

            chosen["target_aux"] = math.tanh(
                chosen["signed"] / tau
            )

            chosen["aug_reward"] = (
                augmented[new_best]
            )

            top_rows.append(
                chosen
            )

            best_global = max(
                x["global_reward"]
                for x in xs
            )

            global_regrets.append(
                best_global
                - chosen["global_reward"]
            )

            advs = group_relative_advantages(
                augmented
            )

            for x, a in zip(xs, advs):

                if float(a) > 0:
                    y = dict(x)

                    y["new_advantage"] = float(a)

                    y["target_aux"] = math.tanh(
                        y["signed"] / tau
                    )

                    positive_rows.append(y)


        top_correct = sum(
            x["correct_sign"]
            for x in top_rows
        ) / len(top_rows)

        top_signed = float(
            np.mean([
                x["signed"]
                for x in top_rows
            ])
        )

        top_aux = float(
            np.mean([
                x["target_aux"]
                for x in top_rows
            ])
        )

        top_global = float(
            np.mean([
                x["global_reward"]
                for x in top_rows
            ])
        )

        mean_regret = float(
            np.mean(global_regrets)
        )

        pos_correct = sum(
            x["correct_sign"]
            for x in positive_rows
        ) / len(positive_rows)

        pos_signed = float(
            np.mean([
                x["signed"]
                for x in positive_rows
            ])
        )

        results.append({
            "tau": tau,
            "lambda": lam,
            "top_correct": top_correct,
            "top_signed": top_signed,
            "top_aux": top_aux,
            "top_global": top_global,
            "global_regret": mean_regret,
            "positive_correct": pos_correct,
            "positive_signed": pos_signed,
            "changed_top": changed_top,
        })


print()
print(
    f"{'TAU':>5s} "
    f"{'LAM':>6s} "
    f"{'TOP_OK':>8s} "
    f"{'TOP_SIGNED':>11s} "
    f"{'TOP_GLOBAL':>11s} "
    f"{'G_REGRET':>9s} "
    f"{'ADV+_OK':>8s} "
    f"{'ADV+SIGNED':>11s} "
    f"{'CHG_TOP':>8s}"
)


for x in results:

    print(
        f"{x['tau']:5.1f} "
        f"{x['lambda']:6.2f} "
        f"{x['top_correct']:8.3f} "
        f"{x['top_signed']:11.4f} "
        f"{x['top_global']:11.4f} "
        f"{x['global_regret']:9.4f} "
        f"{x['positive_correct']:8.3f} "
        f"{x['positive_signed']:11.4f} "
        f"{x['changed_top']:8d}"
    )


print()
print("=" * 125)
print("BEST TARGET ALIGNMENT WITH LOW GLOBAL-REWARD REGRET")
print("=" * 125)

eligible = [
    x for x in results
    if x["global_regret"] <= 0.03
]

eligible.sort(
    key=lambda x: (
        x["positive_correct"],
        x["positive_signed"],
        -x["global_regret"],
    ),
    reverse=True
)

for i, x in enumerate(
    eligible[:15],
    1
):

    print(
        f"{i:02d}. "
        f"tau={x['tau']:.1f} "
        f"lambda={x['lambda']:.2f} "
        f"ADV+ correct={x['positive_correct']:.1%} "
        f"ADV+ signed={x['positive_signed']:+.4f} "
        f"top correct={x['top_correct']:.1%} "
        f"global regret={x['global_regret']:.4f} "
        f"changed_top={x['changed_top']}"
    )
