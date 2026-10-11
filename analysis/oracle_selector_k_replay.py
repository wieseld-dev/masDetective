from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path("artifacts/adaptive_v11_curriculum_v2")
OUT = Path("artifacts/oracle_selector_k_replay_v1")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1337
RANDOM_REPEATS = 200
MAX_BUDGET = 8


def is_chosen(v):
    if v is True or v == 1:
        return True
    if isinstance(v, str):
        return v.strip().lower() in {"true", "1", "yes"}
    return False


def load():
    # Important: a trajectory is parent-directory + system_id.
    # We do not merge different curriculum cycles/stages.
    traj = defaultdict(lambda: defaultdict(list))
    truth = {}

    files = sorted(
        ROOT.rglob("responses_train_round_*.jsonl")
    )

    for p in files:
        parent = str(p.parent)

        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                x = json.loads(line)

                sid = (
                    x.get("system_id")
                    or x.get("session_id")
                )

                rnd = x.get("round")

                gt = tuple(sorted(
                    x.get("truth_roles_training_only")
                    or []
                ))

                selected = tuple(sorted(
                    x.get("selected_roles_training_only")
                    or []
                ))

                if (
                    not sid
                    or rnd is None
                    or len(gt) not in {3,4,5,6}
                ):
                    continue

                key = (parent, sid)

                if key in truth and truth[key] != gt:
                    raise RuntimeError(
                        f"truth mismatch: {key}"
                    )

                truth[key] = gt

                traj[key][int(rnd)].append({
                    "candidate_index":
                        x.get("candidate_index"),
                    "chosen":
                        is_chosen(
                            x.get(
                                "chosen_for_continuation"
                            )
                        ),
                    "selected": selected,
                    "query": x.get("query"),
                })

    # Require valid candidate groups and one actual chosen
    # candidate in each retained round.
    clean = {}

    for key, rm in traj.items():
        good = {}

        for rnd, rows in rm.items():
            if len(rows) not in {6,8}:
                continue

            chosen = [
                r for r in rows
                if r["chosen"]
            ]

            if len(chosen) != 1:
                continue

            good[rnd] = rows

        if good:
            clean[key] = good

    return clean, truth


def choose_actual(rows, seen):
    for r in rows:
        if r["chosen"]:
            return r
    raise RuntimeError("no actual chosen row")


def choose_oracle(rows, seen):
    # Maximize NEW hidden workers exposed.
    scored = []

    for r in rows:
        s = set(r["selected"])

        new = len(s - seen)

        scored.append((
            new,
            len(s),
            -int(r["candidate_index"] or 0),
            r,
        ))

    scored.sort(
        key=lambda x: (
            x[0], x[1], x[2]
        ),
        reverse=True,
    )

    return scored[0][3]


def choose_random(rows, seen, rng):
    return rng.choice(rows)


def run_strategy(
    trajectories,
    truths,
    strategy,
    budget,
    random_seed=None,
):
    records = []

    for key, rm in trajectories.items():
        rounds = sorted(rm)

        if len(rounds) < budget:
            continue

        gt = set(truths[key])

        seen = set()

        rng = (
            random.Random(
                hash((random_seed, key))
                & 0xffffffff
            )
            if random_seed is not None
            else None
        )

        for rnd in rounds[:budget]:
            rows = rm[rnd]

            if strategy == "actual":
                row = choose_actual(
                    rows, seen
                )

            elif strategy == "oracle":
                row = choose_oracle(
                    rows, seen
                )

            elif strategy == "random":
                row = choose_random(
                    rows, seen, rng
                )

            elif strategy == "all":
                for r in rows:
                    seen.update(r["selected"])
                continue

            else:
                raise ValueError(strategy)

            seen.update(row["selected"])

        records.append({
            "k": len(gt),
            "k_hat": len(seen),
            "k_correct":
                len(seen) == len(gt),
            "coverage":
                len(seen & gt) / len(gt),
            "exact":
                seen == gt,
        })

    return records


def summarize(rows):
    if not rows:
        return None

    n = len(rows)

    return {
        "n": n,
        "k_accuracy":
            sum(x["k_correct"] for x in rows) / n,
        "mean_coverage":
            sum(x["coverage"] for x in rows) / n,
        "exact_set":
            sum(x["exact"] for x in rows) / n,
        "mae":
            sum(
                abs(x["k_hat"] - x["k"])
                for x in rows
            ) / n,
    }


def random_summary(
    trajectories,
    truths,
    budget,
):
    runs = []

    for rep in range(RANDOM_REPEATS):
        rows = run_strategy(
            trajectories,
            truths,
            "random",
            budget,
            random_seed=SEED + rep,
        )

        runs.append(summarize(rows))

    keys = [
        "k_accuracy",
        "mean_coverage",
        "exact_set",
        "mae",
    ]

    out = {
        "n": runs[0]["n"],
        "repeats": RANDOM_REPEATS,
    }

    for k in keys:
        vals = [x[k] for x in runs]

        out[k] = sum(vals) / len(vals)

        vals2 = sorted(vals)

        out[k + "_p05"] = vals2[
            int(.05 * (len(vals2)-1))
        ]

        out[k + "_p95"] = vals2[
            int(.95 * (len(vals2)-1))
        ]

    return out


def main():
    trajectories, truths = load()

    print("=" * 100)
    print("ORACLE SELECTOR K-RECOVERY REPLAY")
    print("=" * 100)

    print(
        f"trajectory_instances="
        f"{len(trajectories)}"
    )

    result = {}

    for budget in range(1, MAX_BUDGET + 1):
        actual = summarize(
            run_strategy(
                trajectories,
                truths,
                "actual",
                budget,
            )
        )

        oracle = summarize(
            run_strategy(
                trajectories,
                truths,
                "oracle",
                budget,
            )
        )

        ceiling = summarize(
            run_strategy(
                trajectories,
                truths,
                "all",
                budget,
            )
        )

        rnd = random_summary(
            trajectories,
            truths,
            budget,
        )

        result[str(budget)] = {
            "actual": actual,
            "random": rnd,
            "oracle": oracle,
            "all_candidates": ceiling,
        }

        print()
        print("=" * 100)
        print(f"BUDGET = {budget} PROBES")
        print("=" * 100)

        if actual is None:
            print("No trajectories")
            continue

        for name, x in [
            ("ACTUAL SELECTOR", actual),
            ("RANDOM", rnd),
            ("ORACLE NOVELTY", oracle),
            ("ALL CANDIDATES", ceiling),
        ]:
            print(
                f"{name:18s} "
                f"n={x['n']:4d} "
                f"Kacc={x['k_accuracy']:.3f} "
                f"coverage={x['mean_coverage']:.3f} "
                f"exact={x['exact_set']:.3f} "
                f"MAE={x['mae']:.3f}"
            )

    (OUT / "summary.json").write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "[saved]",
        OUT / "summary.json",
    )


if __name__ == "__main__":
    main()
