from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from v11.role_universe import TRAIN_ROLES


ROOT = Path(
    "artifacts/ten_system_belief_trace_control_v14_dropin"
)


# ============================================================
# Helpers
# ============================================================

def auc_pairwise(scores, labels):
    pos = [
        float(s)
        for s, y in zip(scores, labels)
        if int(y) == 1
    ]

    neg = [
        float(s)
        for s, y in zip(scores, labels)
        if int(y) == 0
    ]

    if not pos or not neg:
        return None

    wins = 0.0
    total = 0

    for p in pos:
        for n in neg:
            total += 1

            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5

    return wins / total


def pearson(xs, ys):
    if len(xs) < 2:
        return None

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)

    if np.std(x) < 1e-12:
        return None

    if np.std(y) < 1e-12:
        return None

    return float(
        np.corrcoef(x, y)[0, 1]
    )


def mean_or_none(xs):
    return (
        float(np.mean(xs))
        if xs
        else None
    )


def fmt(x, nd=4):
    if x is None:
        return "N/A"

    return f"{float(x):.{nd}f}"


def reconstruct_candidate_targets(
    belief_before,
    round_number,
    n_candidates,
):
    """
    Exactly reconstructs the scheduling logic used by
    V14BeliefDropInPolicy in the completed run.
    """

    role_probs = {
        str(k): float(v)
        for k, v in belief_before.items()
    }

    roles = [
        r
        for r in TRAIN_ROLES
        if r in role_probs
    ]

    uncertain = sorted(
        roles,
        key=lambda r: (
            abs(role_probs[r] - 0.5),
            -role_probs[r],
            r,
        ),
    )

    # Historical loop sends rnd - 1 into PublicBeliefState.
    round_index = int(round_number) - 1

    offset = (
        7 * round_index
    ) % len(uncertain)

    return [
        uncertain[
            (offset + j)
            % len(uncertain)
        ]
        for j in range(n_candidates)
    ]


def role_name_in_query(role, query):
    q = (
        str(query)
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    r = (
        str(role)
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    return r in q


# ============================================================
# Load final truth / summaries
# ============================================================

summaries = {}

for p in sorted(
    ROOT.glob("mas*_summary.json")
):
    d = json.loads(
        p.read_text()
    )

    summaries[
        d["system_id"]
    ] = d


if not summaries:
    raise RuntimeError(
        f"No summary files found under {ROOT}"
    )


# ============================================================
# Aggregate containers
# ============================================================

rows = []

sensor_all_scores = []
sensor_all_activation_labels = []
sensor_all_install_labels = []

sensor_target_scores = []
sensor_target_activation_labels = []

chosen_mis = []
chosen_abs_target_deltas = []
chosen_target_activations = []
chosen_target_scores = []

candidate_installed_flags = []
chosen_target_installed_flags = []

khat_all = []
khat_final = []

system_stats = {}


# ============================================================
# Parse all trajectories
# ============================================================

for tp in sorted(
    ROOT.glob("mas*_trajectory.json")
):
    sid = tp.name.replace(
        "_trajectory.json",
        "",
    )

    if sid not in summaries:
        continue

    summary = summaries[sid]

    true_roles = set(
        summary["true_roles"]
    )

    traj = json.loads(
        tp.read_text()
    )

    all_candidate_targets = []
    all_chosen_targets = []

    sys_rows = []

    for rec in traj:
        rnd = int(
            rec["round"]
        )

        belief_before = {
            str(k): float(v)
            for k, v
            in rec["belief_before"].items()
        }

        belief_after = {
            str(k): float(v)
            for k, v
            in rec["belief_after"].items()
        }

        queries = list(
            rec.get(
                "candidate_queries",
                [],
            )
        )

        mis = [
            float(x)
            for x in rec.get(
                "candidate_expected_mi",
                [],
            )
        ]

        chosen_index = int(
            rec["chosen_index"]
        )

        targets = reconstruct_candidate_targets(
            belief_before,
            rnd,
            len(queries),
        )

        all_candidate_targets.extend(
            targets
        )

        if not (
            0 <= chosen_index < len(targets)
        ):
            raise RuntimeError(
                f"{sid} round={rnd}: bad chosen_index"
            )

        target = targets[
            chosen_index
        ]

        all_chosen_targets.append(
            target
        )

        query = str(
            rec.get(
                "query",
                "",
            )
        )

        activated = set(
            rec.get(
                "selected_roles_eval_only",
                [],
            )
        )

        fp = {
            str(k): float(v)
            for k, v
            in rec.get(
                "fingerprint_scores",
                {},
            ).items()
        }

        target_installed = (
            target in true_roles
        )

        target_activated = (
            target in activated
        )

        delta_target = (
            belief_after[target]
            - belief_before[target]
        )

        chosen_mi = (
            mis[chosen_index]
            if chosen_index < len(mis)
            else float("nan")
        )

        target_fp = fp.get(
            target,
            float("nan"),
        )

        # ----------------------------------------------------
        # Did historical MI actually choose its own argmax?
        # ----------------------------------------------------

        argmax_ok = None

        if mis:
            argmax_ok = (
                chosen_index
                == int(np.argmax(mis))
            )

        # ----------------------------------------------------
        # Candidate pool truth composition — eval only.
        # ----------------------------------------------------

        candidate_installed = [
            int(r in true_roles)
            for r in targets
        ]

        candidate_installed_flags.extend(
            candidate_installed
        )

        chosen_target_installed_flags.append(
            int(target_installed)
        )

        # ----------------------------------------------------
        # False-role inflation after this observation.
        # ----------------------------------------------------

        false_roles = [
            r
            for r in belief_before
            if r not in true_roles
        ]

        false_deltas = [
            belief_after[r]
            - belief_before[r]
            for r in false_roles
        ]

        false_positive_deltas = [
            d
            for d in false_deltas
            if d > 0
        ]

        false_inflation_fraction = (
            sum(d > 0 for d in false_deltas)
            / len(false_deltas)
            if false_deltas
            else 0.0
        )

        max_false_increase = (
            max(false_deltas)
            if false_deltas
            else 0.0
        )

        mean_false_delta = (
            float(np.mean(false_deltas))
            if false_deltas
            else 0.0
        )

        true_deltas = [
            belief_after[r]
            - belief_before[r]
            for r in true_roles
        ]

        mean_true_delta = (
            float(np.mean(true_deltas))
            if true_deltas
            else 0.0
        )

        # ----------------------------------------------------
        # Sensor pooled diagnostics
        # ----------------------------------------------------

        for role, score in fp.items():
            sensor_all_scores.append(
                score
            )

            sensor_all_activation_labels.append(
                int(role in activated)
            )

            sensor_all_install_labels.append(
                int(role in true_roles)
            )

        if not math.isnan(
            target_fp
        ):
            sensor_target_scores.append(
                target_fp
            )

            sensor_target_activation_labels.append(
                int(target_activated)
            )

        # ----------------------------------------------------
        # Chosen MI -> realized result diagnostics
        # ----------------------------------------------------

        if not math.isnan(
            chosen_mi
        ):
            chosen_mis.append(
                chosen_mi
            )

            chosen_abs_target_deltas.append(
                abs(delta_target)
            )

            chosen_target_activations.append(
                int(target_activated)
            )

            if not math.isnan(
                target_fp
            ):
                chosen_target_scores.append(
                    target_fp
                )

        # ----------------------------------------------------
        # K drift
        # ----------------------------------------------------

        kp_after = {
            int(k): float(v)
            for k, v
            in rec.get(
                "k_probs_after",
                {},
            ).items()
        }

        khat = (
            max(
                kp_after,
                key=kp_after.get,
            )
            if kp_after
            else None
        )

        if khat is not None:
            khat_all.append(
                khat
            )

        row = {
            "system_id":
                sid,

            "round":
                rnd,

            "true_roles":
                sorted(true_roles),

            "candidate_targets":
                targets,

            "candidate_queries":
                queries,

            "candidate_expected_mi":
                mis,

            "chosen_index":
                chosen_index,

            "chosen_target":
                target,

            "chosen_query":
                query,

            "chosen_mi":
                chosen_mi,

            "mi_argmax_correct":
                argmax_ok,

            "target_installed":
                bool(target_installed),

            "target_activated":
                bool(target_activated),

            "selected_roles_eval_only":
                sorted(activated),

            "target_fingerprint_score":
                target_fp,

            "target_belief_before":
                belief_before[target],

            "target_belief_after":
                belief_after[target],

            "target_belief_delta":
                delta_target,

            "role_name_in_query":
                role_name_in_query(
                    target,
                    query,
                ),

            "candidate_installed_fraction":
                float(
                    np.mean(
                        candidate_installed
                    )
                )
                if candidate_installed
                else None,

            "false_role_inflation_fraction":
                false_inflation_fraction,

            "max_false_role_increase":
                max_false_increase,

            "mean_false_role_delta":
                mean_false_delta,

            "mean_true_role_delta":
                mean_true_delta,

            "k_probs_after":
                kp_after,

            "khat_after":
                khat,
        }

        rows.append(
            row
        )

        sys_rows.append(
            row
        )

    final_k = int(
        summary["predicted_k"]
    )

    khat_final.append(
        final_k
    )

    installed_target_rounds = [
        r
        for r in sys_rows
        if r["target_installed"]
    ]

    absent_target_rounds = [
        r
        for r in sys_rows
        if not r["target_installed"]
    ]

    system_stats[sid] = {
        "belief_auc":
            float(
                summary["belief_auc"]
            ),

        "final_k":
            final_k,

        "candidate_target_coverage":
            len(
                set(
                    all_candidate_targets
                )
            ),

        "chosen_target_coverage":
            len(
                set(
                    all_chosen_targets
                )
            ),

        "chosen_targets":
            all_chosen_targets,

        "installed_target_rounds":
            len(
                installed_target_rounds
            ),

        "installed_target_activation_rate":
            mean_or_none(
                [
                    int(
                        r[
                            "target_activated"
                        ]
                    )
                    for r
                    in installed_target_rounds
                ]
            ),

        "installed_target_positive_delta_rate":
            mean_or_none(
                [
                    int(
                        r[
                            "target_belief_delta"
                        ] > 0
                    )
                    for r
                    in installed_target_rounds
                ]
            ),

        "absent_target_negative_delta_rate":
            mean_or_none(
                [
                    int(
                        r[
                            "target_belief_delta"
                        ] < 0
                    )
                    for r
                    in absent_target_rounds
                ]
            ),

        "mean_false_inflation_fraction":
            mean_or_none(
                [
                    r[
                        "false_role_inflation_fraction"
                    ]
                    for r in sys_rows
                ]
            ),
    }


# ============================================================
# Aggregate diagnostics
# ============================================================

installed_target_rows = [
    r
    for r in rows
    if r["target_installed"]
]

absent_target_rows = [
    r
    for r in rows
    if not r["target_installed"]
]

activated_target_rows = [
    r
    for r in rows
    if r["target_activated"]
]

nonactivated_target_rows = [
    r
    for r in rows
    if not r["target_activated"]
]


result = {
    "n_systems":
        len(system_stats),

    "n_real_mas_observations":
        len(rows),

    "generator_scheduling": {
        "mean_candidate_role_coverage_per_system":
            mean_or_none(
                [
                    x[
                        "candidate_target_coverage"
                    ]
                    for x
                    in system_stats.values()
                ]
            ),

        "mean_chosen_role_coverage_per_system":
            mean_or_none(
                [
                    x[
                        "chosen_target_coverage"
                    ]
                    for x
                    in system_stats.values()
                ]
            ),

        "candidate_target_installed_fraction":
            mean_or_none(
                candidate_installed_flags
            ),

        "chosen_target_installed_fraction":
            mean_or_none(
                chosen_target_installed_flags
            ),

        "explicit_target_name_in_chosen_query_rate":
            mean_or_none(
                [
                    int(
                        r[
                            "role_name_in_query"
                        ]
                    )
                    for r in rows
                ]
            ),
    },

    "target_activation": {
        "installed_target_rounds":
            len(
                installed_target_rows
            ),

        "installed_target_activation_rate":
            mean_or_none(
                [
                    int(
                        r[
                            "target_activated"
                        ]
                    )
                    for r
                    in installed_target_rows
                ]
            ),

        "absent_target_activation_count":
            sum(
                int(
                    r[
                        "target_activated"
                    ]
                )
                for r
                in absent_target_rows
            ),
    },

    "sensor": {
        "pooled_activation_auc_all_roles":
            auc_pairwise(
                sensor_all_scores,
                sensor_all_activation_labels,
            ),

        "target_only_activation_auc":
            auc_pairwise(
                sensor_target_scores,
                sensor_target_activation_labels,
            ),

        "pooled_installed_role_auc_raw_fingerprint":
            auc_pairwise(
                sensor_all_scores,
                sensor_all_install_labels,
            ),

        "mean_target_score_when_activated":
            mean_or_none(
                [
                    r[
                        "target_fingerprint_score"
                    ]
                    for r
                    in activated_target_rows
                    if not math.isnan(
                        r[
                            "target_fingerprint_score"
                        ]
                    )
                ]
            ),

        "mean_target_score_when_not_activated":
            mean_or_none(
                [
                    r[
                        "target_fingerprint_score"
                    ]
                    for r
                    in nonactivated_target_rows
                    if not math.isnan(
                        r[
                            "target_fingerprint_score"
                        ]
                    )
                ]
            ),
    },

    "belief_update": {
        "installed_target_positive_delta_rate":
            mean_or_none(
                [
                    int(
                        r[
                            "target_belief_delta"
                        ] > 0
                    )
                    for r
                    in installed_target_rows
                ]
            ),

        "absent_target_negative_delta_rate":
            mean_or_none(
                [
                    int(
                        r[
                            "target_belief_delta"
                        ] < 0
                    )
                    for r
                    in absent_target_rows
                ]
            ),

        "activated_target_positive_delta_rate":
            mean_or_none(
                [
                    int(
                        r[
                            "target_belief_delta"
                        ] > 0
                    )
                    for r
                    in activated_target_rows
                ]
            ),

        "not_activated_target_negative_delta_rate":
            mean_or_none(
                [
                    int(
                        r[
                            "target_belief_delta"
                        ] < 0
                    )
                    for r
                    in nonactivated_target_rows
                ]
            ),

        "mean_false_role_inflation_fraction_per_round":
            mean_or_none(
                [
                    r[
                        "false_role_inflation_fraction"
                    ]
                    for r in rows
                ]
            ),

        "mean_max_false_role_increase_per_round":
            mean_or_none(
                [
                    r[
                        "max_false_role_increase"
                    ]
                    for r in rows
                ]
            ),

        "mean_true_role_delta":
            mean_or_none(
                [
                    r[
                        "mean_true_role_delta"
                    ]
                    for r in rows
                ]
            ),

        "mean_false_role_delta":
            mean_or_none(
                [
                    r[
                        "mean_false_role_delta"
                    ]
                    for r in rows
                ]
            ),
    },

    "historical_mi_selector": {
        "argmax_sanity_rate":
            mean_or_none(
                [
                    int(
                        r[
                            "mi_argmax_correct"
                        ]
                    )
                    for r in rows
                    if r[
                        "mi_argmax_correct"
                    ] is not None
                ]
            ),

        "corr_predicted_mi_vs_abs_target_belief_delta":
            pearson(
                chosen_mis,
                chosen_abs_target_deltas,
            ),

        "corr_predicted_mi_vs_target_activation":
            pearson(
                chosen_mis,
                chosen_target_activations,
            ),

        "corr_predicted_mi_vs_target_fingerprint_score":
            pearson(
                chosen_mis[
                    :len(
                        chosen_target_scores
                    )
                ],
                chosen_target_scores,
            )
            if chosen_target_scores
            else None,
    },

    "k_posterior": {
        "all_round_khat_counts":
            dict(
                Counter(
                    khat_all
                )
            ),

        "final_khat_counts":
            dict(
                Counter(
                    khat_final
                )
            ),
    },

    "per_system":
        system_stats,
}


# ============================================================
# Save machine-readable report
# ============================================================

out_json = (
    ROOT
    / "bridge_diagnostic_v1.json"
)

out_json.write_text(
    json.dumps(
        {
            "summary":
                result,

            "rounds":
                rows,
        },
        indent=2,
    )
)


# ============================================================
# Human-readable output
# ============================================================

print()
print("=" * 118)
print("V14 → HISTORICAL MI / SENSOR / BAYES BRIDGE DIAGNOSTIC")
print("=" * 118)

print(
    f"Systems                         : "
    f"{result['n_systems']}"
)

print(
    f"Real MAS observations           : "
    f"{result['n_real_mas_observations']}"
)

print()

g = result[
    "generator_scheduling"
]

print("GENERATOR / SCHEDULING")
print("-" * 118)

print(
    f"Mean candidate-role coverage    : "
    f"{fmt(g['mean_candidate_role_coverage_per_system'], 2)} / 20"
)

print(
    f"Mean chosen-role coverage       : "
    f"{fmt(g['mean_chosen_role_coverage_per_system'], 2)} / 20"
)

print(
    f"Candidate targets installed     : "
    f"{fmt(g['candidate_target_installed_fraction'])}"
)

print(
    f"Chosen targets installed        : "
    f"{fmt(g['chosen_target_installed_fraction'])}"
)

print(
    f"Target role phrase in query     : "
    f"{fmt(g['explicit_target_name_in_chosen_query_rate'])}"
)

print()

a = result[
    "target_activation"
]

print("TARGET ACTIVATION — EVAL ONLY")
print("-" * 118)

print(
    f"Installed-target opportunities  : "
    f"{a['installed_target_rounds']}"
)

print(
    f"Installed target activated      : "
    f"{fmt(a['installed_target_activation_rate'])}"
)

print(
    f"Absent target activated         : "
    f"{a['absent_target_activation_count']} "
    f"(should be 0)"
)

print()

s = result[
    "sensor"
]

print("OLD SENSOR ON V14 QUERY DISTRIBUTION")
print("-" * 118)

print(
    f"Activation AUC — all roles      : "
    f"{fmt(s['pooled_activation_auc_all_roles'])}"
)

print(
    f"Activation AUC — target only    : "
    f"{fmt(s['target_only_activation_auc'])}"
)

print(
    f"Raw score installed-role AUC    : "
    f"{fmt(s['pooled_installed_role_auc_raw_fingerprint'])}"
)

print(
    f"Target score | activated        : "
    f"{fmt(s['mean_target_score_when_activated'])}"
)

print(
    f"Target score | not activated    : "
    f"{fmt(s['mean_target_score_when_not_activated'])}"
)

print()

b = result[
    "belief_update"
]

print("BAYES / BELIEF UPDATE")
print("-" * 118)

print(
    f"Installed target belief ↑       : "
    f"{fmt(b['installed_target_positive_delta_rate'])}"
)

print(
    f"Absent target belief ↓          : "
    f"{fmt(b['absent_target_negative_delta_rate'])}"
)

print(
    f"Activated target belief ↑       : "
    f"{fmt(b['activated_target_positive_delta_rate'])}"
)

print(
    f"Non-activated target belief ↓   : "
    f"{fmt(b['not_activated_target_negative_delta_rate'])}"
)

print(
    f"False roles increasing / round  : "
    f"{fmt(b['mean_false_role_inflation_fraction_per_round'])}"
)

print(
    f"Mean max false-role increase    : "
    f"{fmt(b['mean_max_false_role_increase_per_round'], 6)}"
)

print(
    f"Mean true-role belief Δ         : "
    f"{fmt(b['mean_true_role_delta'], 6)}"
)

print(
    f"Mean false-role belief Δ        : "
    f"{fmt(b['mean_false_role_delta'], 6)}"
)

print()

m = result[
    "historical_mi_selector"
]

print("HISTORICAL MI SELECTOR")
print("-" * 118)

print(
    f"argmax implementation sanity    : "
    f"{fmt(m['argmax_sanity_rate'])}"
)

print(
    f"corr(MI, |target belief Δ|)     : "
    f"{fmt(m['corr_predicted_mi_vs_abs_target_belief_delta'])}"
)

print(
    f"corr(MI, target activation)     : "
    f"{fmt(m['corr_predicted_mi_vs_target_activation'])}"
)

print(
    f"corr(MI, target sensor score)   : "
    f"{fmt(m['corr_predicted_mi_vs_target_fingerprint_score'])}"
)

print()

k = result[
    "k_posterior"
]

print("K POSTERIOR")
print("-" * 118)

print(
    "Khat across all rounds          : "
    f"{k['all_round_khat_counts']}"
)

print(
    "Final Khat                      : "
    f"{k['final_khat_counts']}"
)

print()

print("PER SYSTEM")
print("-" * 118)

for sid, x in system_stats.items():
    print(
        f"{sid:36s} "
        f"AUC={x['belief_auc']:.4f}  "
        f"candCoverage={x['candidate_target_coverage']:2d}/20  "
        f"chosenCoverage={x['chosen_target_coverage']:2d}/20  "
        f"installed→activation="
        f"{fmt(x['installed_target_activation_rate'])}  "
        f"installed→belief↑="
        f"{fmt(x['installed_target_positive_delta_rate'])}  "
        f"absent→belief↓="
        f"{fmt(x['absent_target_negative_delta_rate'])}"
    )

print()

print("WORST FALSE-TARGET UPDATES")
print("-" * 118)

bad_absent = sorted(
    [
        r
        for r in rows
        if (
            not r["target_installed"]
            and r["target_belief_delta"] > 0
        )
    ],
    key=lambda r:
        r["target_belief_delta"],
    reverse=True,
)

for r in bad_absent[:12]:
    q = (
        r["chosen_query"]
        .replace("\n", " ")
    )

    print(
        f"{r['system_id']} "
        f"R{r['round']} "
        f"target={r['chosen_target']:28s} "
        f"Δ={r['target_belief_delta']:+.5f} "
        f"score={r['target_fingerprint_score']:.4f} "
        f"MI={r['chosen_mi']:.5f}"
    )

    print(
        f"    {q[:180]}"
    )

print()
print(
    "Saved detailed report to:",
    out_json,
)
print("=" * 118)
