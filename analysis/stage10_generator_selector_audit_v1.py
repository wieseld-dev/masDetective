#!/usr/bin/env python3

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(
    "artifacts/adaptive_v11_curriculum_v2/stage_10"
)

OUT = Path(
    "artifacts/stage10_generator_selector_audit_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def readj(path):
    with Path(path).open() as f:
        return [
            json.loads(x)
            for x in f
            if x.strip()
        ]


def pct(x):
    return f"{100*x:.2f}%"


def safe_mean(xs):
    return mean(xs) if xs else float("nan")


def response_ok(x):
    return (
        x.get("execution_status") == "ok"
        and bool(x.get("valid", True))
    )


def analyze_train_cycle(cycle):
    base = ROOT / cycle

    final_selected_union = defaultdict(set)
    final_pool_union = defaultdict(set)
    truth_by_sid = {}

    round_reports = []

    for rnd in range(1, 9):
        rp = (
            base
            / f"responses_train_round_{rnd:02d}.jsonl"
        )

        dp = (
            base
            / f"selector_diag_train_round_{rnd:02d}.jsonl"
        )

        rows = readj(rp)
        diags = readj(dp)

        groups = defaultdict(list)

        for x in rows:
            groups[str(x["session_id"])].append(x)

        if len(groups) != 48:
            raise RuntimeError(
                f"{cycle} r{rnd}: "
                f"expected 48 systems, got {len(groups)}"
            )

        pool_coverages = []
        chosen_coverages = []
        oracle_single_coverages = []

        pool_exact = 0
        chosen_exact = 0
        oracle_single_exact = 0

        valid_candidates = 0
        total_candidates = 0

        for sid, rr in groups.items():
            truths = {
                tuple(sorted(x["truth_roles_training_only"]))
                for x in rr
            }

            if len(truths) != 1:
                raise RuntimeError(
                    f"{cycle} r{rnd} {sid}: "
                    "inconsistent truth roles"
                )

            truth = set(next(iter(truths)))
            truth_by_sid[sid] = truth

            good = [
                x for x in rr
                if response_ok(x)
            ]

            total_candidates += len(rr)
            valid_candidates += len(good)

            pool_union = set()

            for x in good:
                pool_union.update(
                    x.get(
                        "selected_roles_training_only",
                        [],
                    )
                )

            chosen = [
                x for x in rr
                if x.get("chosen_for_continuation")
            ]

            if len(chosen) != 1:
                raise RuntimeError(
                    f"{cycle} r{rnd} {sid}: "
                    f"expected 1 chosen, got {len(chosen)}"
                )

            chosen_roles = set(
                chosen[0].get(
                    "selected_roles_training_only",
                    [],
                )
            )

            single_covs = []

            for x in good:
                sr = set(
                    x.get(
                        "selected_roles_training_only",
                        [],
                    )
                )

                single_covs.append(
                    len(sr & truth)
                    / len(truth)
                )

            oracle_single_cov = (
                max(single_covs)
                if single_covs
                else 0.0
            )

            pool_cov = (
                len(pool_union & truth)
                / len(truth)
            )

            chosen_cov = (
                len(chosen_roles & truth)
                / len(truth)
            )

            pool_coverages.append(
                pool_cov
            )

            chosen_coverages.append(
                chosen_cov
            )

            oracle_single_coverages.append(
                oracle_single_cov
            )

            pool_exact += int(
                pool_union == truth
            )

            chosen_exact += int(
                chosen_roles == truth
            )

            oracle_single_exact += int(
                oracle_single_cov == 1.0
            )

            final_pool_union[sid].update(
                pool_union
            )

            final_selected_union[sid].update(
                chosen_roles
            )

        diag_by_sid = {
            str(x["session_id"]): x
            for x in diags
        }

        chosen_ranks = []
        regrets = []
        selector_top1 = 0
        pool_positive = 0

        for sid in groups:
            d = diag_by_sid[sid]

            rank = int(
                d["chosen_rank"]
            )

            chosen_ranks.append(rank)

            regrets.append(
                float(d["regret"])
            )

            selector_top1 += int(
                rank == 1
            )

            pool_positive += int(
                bool(
                    d["pool_has_positive"]
                )
            )

        round_reports.append({
            "round": rnd,

            "systems":
                len(groups),

            "valid_candidate_rate":
                valid_candidates
                / total_candidates,

            "mean_pool_role_coverage":
                safe_mean(
                    pool_coverages
                ),

            "mean_chosen_role_coverage":
                safe_mean(
                    chosen_coverages
                ),

            "mean_oracle_single_coverage":
                safe_mean(
                    oracle_single_coverages
                ),

            "pool_exact_role_exposure":
                pool_exact
                / len(groups),

            "chosen_exact_role_exposure":
                chosen_exact
                / len(groups),

            "oracle_single_exact_role_exposure":
                oracle_single_exact
                / len(groups),

            "selector_top1":
                selector_top1
                / len(groups),

            "mean_chosen_rank":
                safe_mean(
                    chosen_ranks
                ),

            "mean_selector_regret":
                safe_mean(
                    regrets
                ),

            "pool_has_positive":
                pool_positive
                / len(groups),
        })


    final_selected_cov = []
    final_pool_cov = []

    final_selected_exact = 0
    final_pool_exact = 0

    final_selected_k_exact = 0
    final_pool_k_exact = 0

    for sid, truth in truth_by_sid.items():
        su = (
            final_selected_union[sid]
            & truth
        )

        pu = (
            final_pool_union[sid]
            & truth
        )

        final_selected_cov.append(
            len(su) / len(truth)
        )

        final_pool_cov.append(
            len(pu) / len(truth)
        )

        final_selected_exact += int(
            su == truth
        )

        final_pool_exact += int(
            pu == truth
        )

        final_selected_k_exact += int(
            len(su) == len(truth)
        )

        final_pool_k_exact += int(
            len(pu) == len(truth)
        )

    final = {
        "systems":
            len(truth_by_sid),

        "selected_path_mean_role_coverage":
            safe_mean(
                final_selected_cov
            ),

        "pool_oracle_mean_role_coverage":
            safe_mean(
                final_pool_cov
            ),

        "selected_path_exact_role_exposure":
            final_selected_exact
            / len(truth_by_sid),

        "pool_oracle_exact_role_exposure":
            final_pool_exact
            / len(truth_by_sid),

        "selected_path_K_exposure_exact":
            final_selected_k_exact
            / len(truth_by_sid),

        "pool_oracle_K_exposure_exact":
            final_pool_k_exact
            / len(truth_by_sid),
    }

    return {
        "rounds":
            round_reports,

        "final":
            final,
    }


def analyze_dev():
    base = ROOT / "dev_eval"

    unions = defaultdict(set)
    truth_by_sid = {}

    round_reports = []

    for rnd in range(1, 9):
        rp = (
            base
            / f"responses_dev_round_{rnd:02d}.jsonl"
        )

        rows = readj(rp)

        if len(rows) != 18:
            raise RuntimeError(
                f"dev r{rnd}: "
                f"expected 18 rows, got {len(rows)}"
            )

        coverages = []
        exact = 0

        for x in rows:
            sid = str(
                x["session_id"]
            )

            truth = set(
                x[
                    "truth_roles_training_only"
                ]
            )

            truth_by_sid[sid] = truth

            selected = set(
                x.get(
                    "selected_roles_training_only",
                    [],
                )
            )

            cov = (
                len(selected & truth)
                / len(truth)
            )

            coverages.append(cov)

            exact += int(
                selected == truth
            )

            unions[sid].update(
                selected
            )

        round_reports.append({
            "round":
                rnd,

            "mean_selected_role_coverage":
                safe_mean(
                    coverages
                ),

            "single_probe_exact_role_exposure":
                exact / len(rows),
        })


    cumulative_cov = []
    cumulative_exact = 0
    cumulative_k_exact = 0

    by_k = defaultdict(
        lambda: {
            "n": 0,
            "exact": 0,
            "coverage": [],
        }
    )

    for sid, truth in truth_by_sid.items():
        exposed = (
            unions[sid]
            & truth
        )

        cov = (
            len(exposed)
            / len(truth)
        )

        cumulative_cov.append(
            cov
        )

        exact = int(
            exposed == truth
        )

        k_exact = int(
            len(exposed)
            == len(truth)
        )

        cumulative_exact += exact
        cumulative_k_exact += k_exact

        K = len(truth)

        by_k[K]["n"] += 1
        by_k[K]["exact"] += exact
        by_k[K]["coverage"].append(
            cov
        )

    return {
        "rounds":
            round_reports,

        "final": {
            "systems":
                len(truth_by_sid),

            "mean_role_coverage":
                safe_mean(
                    cumulative_cov
                ),

            "exact_role_exposure":
                cumulative_exact
                / len(truth_by_sid),

            "K_exposure_exact":
                cumulative_k_exact
                / len(truth_by_sid),

            "by_K": {
                str(k): {
                    "n":
                        v["n"],

                    "exact_role_exposure":
                        (
                            v["exact"]
                            / v["n"]
                        ),

                    "mean_role_coverage":
                        safe_mean(
                            v["coverage"]
                        ),
                }
                for k, v
                in sorted(
                    by_k.items()
                )
            },
        },
    }


report = {
    "cycle_01":
        analyze_train_cycle(
            "cycle_01"
        ),

    "cycle_02":
        analyze_train_cycle(
            "cycle_02"
        ),

    "dev_selected_path":
        analyze_dev(),

    "interpretation":
        (
            "Privileged routing/exposure audit only. "
            "This is an upper bound on information "
            "made available to a future black-box sensor."
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
print("STAGE-10 GENERATOR / SELECTOR AUDIT")
print("=" * 100)

for name in [
    "cycle_01",
    "cycle_02",
]:
    x = report[name]

    print()
    print(
        "=" * 45,
        name.upper(),
        "=" * 45,
    )

    for r in x["rounds"]:
        print(
            f"r{r['round']:02d} "
            f"pool_cov={pct(r['mean_pool_role_coverage'])} "
            f"chosen_cov={pct(r['mean_chosen_role_coverage'])} "
            f"oracle1_cov={pct(r['mean_oracle_single_coverage'])} "
            f"pool_exact={pct(r['pool_exact_role_exposure'])} "
            f"selector_top1={pct(r['selector_top1'])} "
            f"rank={r['mean_chosen_rank']:.2f} "
            f"regret={r['mean_selector_regret']:.3f}"
        )

    f = x["final"]

    print()
    print("AFTER ALL 8 ROUNDS:")
    print(
        "selected path mean role coverage :",
        pct(
            f[
                "selected_path_mean_role_coverage"
            ]
        ),
    )

    print(
        "pool oracle mean role coverage   :",
        pct(
            f[
                "pool_oracle_mean_role_coverage"
            ]
        ),
    )

    print(
        "selected exact role exposure     :",
        pct(
            f[
                "selected_path_exact_role_exposure"
            ]
        ),
    )

    print(
        "pool oracle exact role exposure  :",
        pct(
            f[
                "pool_oracle_exact_role_exposure"
            ]
        ),
    )


d = report[
    "dev_selected_path"
]["final"]

print()
print(
    "=" * 40,
    "DEV SELECTED PATH",
    "=" * 40,
)

print(
    "systems                 :",
    d["systems"],
)

print(
    "8-round mean coverage   :",
    pct(
        d[
            "mean_role_coverage"
        ]
    ),
)

print(
    "8-round exact exposure  :",
    pct(
        d[
            "exact_role_exposure"
        ]
    ),
)

print(
    "8-round K exposure exact:",
    pct(
        d[
            "K_exposure_exact"
        ]
    ),
)

print()
print("BY K:")

for k, x in (
    d["by_K"].items()
):
    print(
        f"K={k}: "
        f"n={x['n']} "
        f"coverage={pct(x['mean_role_coverage'])} "
        f"exact={pct(x['exact_role_exposure'])}"
    )

print()
print(
    "NOTE: PRIVILEGED EXPOSURE UPPER BOUND ONLY"
)

print(
    "INTERNAL TEST USED: NO"
)

print(
    "AUDIT: DONE"
)
