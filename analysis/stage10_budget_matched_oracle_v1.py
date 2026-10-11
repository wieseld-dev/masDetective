#!/usr/bin/env python3

import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path("artifacts/adaptive_v11_curriculum_v2/stage_10")
OUT = Path("artifacts/stage10_budget_matched_oracle_v1")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260922
MC = 5000


def readj(p):
    with Path(p).open() as f:
        return [
            json.loads(x)
            for x in f
            if x.strip()
        ]


def pct(x):
    return f"{100*x:.2f}%"


def load_cycle(cycle):
    base = ROOT / cycle

    rounds = {}

    for rnd in range(1, 9):
        p = base / f"responses_train_round_{rnd:02d}.jsonl"
        rows = readj(p)

        groups = defaultdict(list)

        for x in rows:
            if (
                x.get("execution_status") == "ok"
                and bool(x.get("valid", True))
            ):
                groups[str(x["session_id"])].append(x)

        rounds[rnd] = groups

    return rounds


def evaluate_cycle(cycle):
    rounds = load_cycle(cycle)

    sids = sorted(
        set.intersection(
            *[
                set(rounds[r].keys())
                for r in rounds
            ]
        )
    )

    current_coverages = []
    oracle_coverages = []

    current_exact = 0
    oracle_exact = 0

    current_by_k = defaultdict(list)
    oracle_by_k = defaultdict(list)

    per_system = []

    for sid in sids:
        truth_sets = []

        for rnd in range(1, 9):
            for x in rounds[rnd][sid]:
                truth_sets.append(
                    tuple(
                        sorted(
                            x["truth_roles_training_only"]
                        )
                    )
                )

        if len(set(truth_sets)) != 1:
            raise RuntimeError(
                f"{sid}: inconsistent truth"
            )

        truth = set(truth_sets[0])
        K = len(truth)

        current_seen = set()
        oracle_seen = set()

        current_trace = []
        oracle_trace = []

        for rnd in range(1, 9):
            pool = rounds[rnd][sid]

            if len(pool) != 8:
                raise RuntimeError(
                    f"{cycle} {sid} r{rnd}: "
                    f"expected 8 candidates, got {len(pool)}"
                )

            chosen = [
                x for x in pool
                if x.get("chosen_for_continuation")
            ]

            if len(chosen) != 1:
                raise RuntimeError(
                    f"{cycle} {sid} r{rnd}: "
                    f"chosen={len(chosen)}"
                )

            c = chosen[0]

            c_roles = set(
                c.get(
                    "selected_roles_training_only",
                    [],
                )
            ) & truth

            current_seen.update(c_roles)

            current_trace.append({
                "round": rnd,
                "candidate_index":
                    int(c["candidate_index"]),
                "new_roles":
                    sorted(
                        c_roles
                        - (
                            current_seen
                            - c_roles
                        )
                    ),
            })

            # Same-budget oracle:
            # choose exactly ONE candidate this round,
            # maximizing NEW roles not already exposed.
            scored = []

            for x in pool:
                roles = set(
                    x.get(
                        "selected_roles_training_only",
                        [],
                    )
                ) & truth

                new_roles = (
                    roles
                    - oracle_seen
                )

                scored.append((
                    len(new_roles),
                    len(roles),
                    -int(x["candidate_index"]),
                    x,
                    roles,
                    new_roles,
                ))

            scored.sort(
                reverse=True,
                key=lambda z:
                    (
                        z[0],
                        z[1],
                        z[2],
                    ),
            )

            (
                gain,
                _,
                _,
                best,
                best_roles,
                new_roles,
            ) = scored[0]

            oracle_seen.update(
                best_roles
            )

            oracle_trace.append({
                "round":
                    rnd,

                "candidate_index":
                    int(
                        best[
                            "candidate_index"
                        ]
                    ),

                "new_roles":
                    sorted(new_roles),

                "gain":
                    int(gain),
            })

        current_cov = (
            len(current_seen)
            / len(truth)
        )

        oracle_cov = (
            len(oracle_seen)
            / len(truth)
        )

        current_coverages.append(
            current_cov
        )

        oracle_coverages.append(
            oracle_cov
        )

        current_exact += int(
            current_seen == truth
        )

        oracle_exact += int(
            oracle_seen == truth
        )

        current_by_k[K].append(
            current_cov
        )

        oracle_by_k[K].append(
            oracle_cov
        )

        per_system.append({
            "system_id": sid,
            "K": K,

            "current_coverage":
                current_cov,

            "oracle_coverage":
                oracle_cov,

            "current_seen":
                sorted(current_seen),

            "oracle_seen":
                sorted(oracle_seen),

            "truth_roles":
                sorted(truth),

            "current_trace":
                current_trace,

            "oracle_trace":
                oracle_trace,
        })


    # --------------------------------------------------
    # Random selector baseline with SAME 8-query budget.
    # --------------------------------------------------

    rng = random.Random(
        SEED
        + (
            1 if cycle == "cycle_01"
            else 2
        )
    )

    random_trial_means = []
    random_trial_exacts = []

    for _ in range(MC):
        covs = []
        exacts = []

        for sid in sids:
            truth = set(
                next(iter(
                    {
                        tuple(
                            sorted(
                                x[
                                    "truth_roles_training_only"
                                ]
                            )
                        )
                        for x
                        in rounds[1][sid]
                    }
                ))
            )

            seen = set()

            for rnd in range(1, 9):
                x = rng.choice(
                    rounds[rnd][sid]
                )

                seen.update(
                    set(
                        x.get(
                            "selected_roles_training_only",
                            [],
                        )
                    )
                    & truth
                )

            covs.append(
                len(seen)
                / len(truth)
            )

            exacts.append(
                int(seen == truth)
            )

        random_trial_means.append(
            mean(covs)
        )

        random_trial_exacts.append(
            mean(exacts)
        )


    result = {
        "cycle":
            cycle,

        "systems":
            len(sids),

        "query_budget":
            8,

        "current_selector": {
            "mean_role_coverage":
                mean(
                    current_coverages
                ),

            "exact_role_exposure":
                current_exact
                / len(sids),
        },

        "marginal_oracle": {
            "mean_role_coverage":
                mean(
                    oracle_coverages
                ),

            "exact_role_exposure":
                oracle_exact
                / len(sids),
        },

        "random_selector": {
            "trials":
                MC,

            "mean_role_coverage":
                mean(
                    random_trial_means
                ),

            "mean_exact_role_exposure":
                mean(
                    random_trial_exacts
                ),
        },

        "by_K": {
            str(k): {
                "n":
                    len(
                        current_by_k[k]
                    ),

                "current_coverage":
                    mean(
                        current_by_k[k]
                    ),

                "oracle_coverage":
                    mean(
                        oracle_by_k[k]
                    ),
            }
            for k in sorted(
                current_by_k
            )
        },

        "per_system":
            per_system,
    }

    return result


report = {
    "cycle_01":
        evaluate_cycle(
            "cycle_01"
        ),

    "cycle_02":
        evaluate_cycle(
            "cycle_02"
        ),

    "note":
        (
            "Logged-pool same-budget diagnostic. "
            "Oracle selects one candidate per logged round "
            "using privileged marginal unseen-role exposure. "
            "Future pools were generated under the original "
            "trajectory, so this is not a fully counterfactual "
            "adaptive rollout."
        ),

    "internal_test_used":
        False,
}


(
    OUT / "REPORT.json"
).write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
)


print("=" * 100)
print("STAGE-10 SAME-BUDGET SELECTOR ORACLE")
print("=" * 100)

for cycle in [
    "cycle_01",
    "cycle_02",
]:
    x = report[cycle]

    print()
    print(
        "=" * 42,
        cycle.upper(),
        "=" * 42,
    )

    print(
        "systems:",
        x["systems"],
    )

    print(
        "query budget:",
        x["query_budget"],
    )

    print()

    print(
        "CURRENT coverage:",
        pct(
            x[
                "current_selector"
            ][
                "mean_role_coverage"
            ]
        ),
    )

    print(
        "CURRENT exact   :",
        pct(
            x[
                "current_selector"
            ][
                "exact_role_exposure"
            ]
        ),
    )

    print()

    print(
        "RANDOM coverage :",
        pct(
            x[
                "random_selector"
            ][
                "mean_role_coverage"
            ]
        ),
    )

    print(
        "RANDOM exact    :",
        pct(
            x[
                "random_selector"
            ][
                "mean_exact_role_exposure"
            ]
        ),
    )

    print()

    print(
        "ORACLE coverage :",
        pct(
            x[
                "marginal_oracle"
            ][
                "mean_role_coverage"
            ]
        ),
    )

    print(
        "ORACLE exact    :",
        pct(
            x[
                "marginal_oracle"
            ][
                "exact_role_exposure"
            ]
        ),
    )

    print()
    print("BY K:")

    for k, v in (
        x["by_K"].items()
    ):
        print(
            f"K={k}: "
            f"n={v['n']} "
            f"current={pct(v['current_coverage'])} "
            f"oracle={pct(v['oracle_coverage'])}"
        )


print()
print(
    "NOTE: SAME QUERY BUDGET = 8"
)

print(
    "INTERNAL TEST USED: NO"
)

print(
    "AUDIT: DONE"
)
