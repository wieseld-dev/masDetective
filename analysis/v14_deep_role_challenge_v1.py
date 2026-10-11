from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import analysis.ten_system_belief_trace_control_v14_dropin as base

from v14.belief_dropin_policy import V14BeliefDropInPolicy
from v11.role_universe import ROLE_DESCRIPTIONS, anchor_probe


OUT = Path("artifacts/v14_deep_role_challenge_v1")
OUT.mkdir(parents=True, exist_ok=True)

ROLES = tuple(base.ROLES)

PROBES_PER_ROLE = 10
TEMPERATURE = 0.85
TOP_P = 0.95
MAX_NEW_TOKENS = 192
GEN_RETRIES = 12
MAS_RETRIES = 3

CHALLENGES = [
    (
        "challenge01_code_finance_access",
        (
            "code",
            "financial_analysis",
            "accessibility_review",
        ),
    ),
    (
        "challenge02_math_docs_support",
        (
            "math",
            "documentation",
            "customer_support_triage",
        ),
    ),
    (
        "challenge03_plan_db_fact",
        (
            "planning",
            "database_design",
            "fact_checking",
        ),
    ),
]


# ============================================================
# Basic utilities
# ============================================================

def dump_json(path: Path, obj):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, obj):
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(obj, ensure_ascii=False)
            + "\n"
        )


def read_jsonl(path: Path):
    if not path.exists():
        return []

    rows = []

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        line = line.strip()

        if line:
            rows.append(
                json.loads(line)
            )

    return rows


def pair_auc(scores, true_roles):
    truth = set(true_roles)

    pos = [
        float(scores[r])
        for r in ROLES
        if r in truth
    ]

    neg = [
        float(scores[r])
        for r in ROLES
        if r not in truth
    ]

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


def evaluate_scores(scores, true_roles):
    ranking = sorted(
        ROLES,
        key=lambda r: -float(scores[r]),
    )

    top3 = ranking[:3]

    return {
        "auc":
            pair_auc(
                scores,
                true_roles,
            ),

        "top3":
            top3,

        "top3_exact":
            set(top3)
            == set(true_roles),

        "ranking":
            ranking,
    }


# ============================================================
# V14 generation
# ============================================================

def native_v14_sample(
    policy,
    observation,
    **kwargs,
):
    """
    Skip V14BeliefDropInPolicy's historical belief scheduler.
    Directly use the native V13/V14 role-conditioned
    generation mechanism with the V14 adapter.
    """
    return super(
        V14BeliefDropInPolicy,
        policy,
    ).sample_probe(
        observation,
        **kwargs,
    )


def build_semantic_neighbors(embedder):
    descriptions = [
        ROLE_DESCRIPTIONS[r]
        for r in ROLES
    ]

    E = np.asarray(
        embedder.encode(
            descriptions,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )

    E /= np.maximum(
        np.linalg.norm(
            E,
            axis=1,
            keepdims=True,
        ),
        1e-12,
    )

    S = E @ E.T

    neighbors = {}

    for i, role in enumerate(ROLES):
        order = np.argsort(-S[i])

        neighbors[role] = [
            ROLES[j]
            for j in order
            if j != i
        ][:4]

    return neighbors


def generate_probe(
    policy,
    role,
    neighbors,
    avoid_for_role,
    seed_base,
):
    observation = {
        "candidate_role":
            role,

        "candidate_description":
            ROLE_DESCRIPTIONS[role],

        "anchor_probe":
            anchor_probe(role),

        "confusable_roles":
            [
                {
                    "role": r,
                    "description":
                        ROLE_DESCRIPTIONS[r],
                }
                for r in neighbors[role]
            ],

        # Keep response history out of this diagnostic.
        "public_history":
            [],

        # Explicitly force diversity across the 10 probes.
        "avoid_probes":
            list(avoid_for_role)[-12:],
    }

    last = None

    for retry in range(GEN_RETRIES):
        seed = seed_base + retry

        random.seed(seed)
        np.random.seed(
            seed % (2**32 - 1)
        )
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        sample = native_v14_sample(
            policy,
            observation,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        q = str(
            sample.query or ""
        ).strip()

        ok, why = base.surface_valid(q)

        valid = (
            bool(sample.valid)
            and bool(ok)
            and bool(q)
            and q not in avoid_for_role
        )

        last = {
            "query": q,
            "sample_valid":
                bool(sample.valid),
            "surface_valid":
                bool(ok),
            "why":
                str(why),
        }

        if valid:
            return q

    raise RuntimeError(
        f"Failed to generate distinct valid "
        f"probe for {role}: {last}"
    )


# ============================================================
# MAS + historical visible sensor
# ============================================================

def make_target(
    sid,
    true_roles,
):
    spec = base.SingleRouterSystemSpec(
        system_id=sid,
        split="new_challenge_deep_role_v1",
        execution_mode="mixed",
        style="s1",
        worker_roles=tuple(true_roles),
    )

    return base.SingleRouterFanoutMAS(
        spec
    )


def query_target(
    target,
    query,
):
    last_error = None

    for attempt in range(MAS_RETRIES):
        try:
            return str(
                target.query(query)
            )

        except Exception as e:
            last_error = repr(e)

            if attempt + 1 < MAS_RETRIES:
                time.sleep(
                    2 * (attempt + 1)
                )

    raise RuntimeError(
        f"MAS query failed after "
        f"{MAS_RETRIES} attempts: "
        f"{last_error}"
    )


def visible_sensor(
    pipe,
    query,
    output,
):
    qemb = np.asarray(
        pipe["embedder"].encode(
            [query],
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )

    oemb = np.asarray(
        pipe["embedder"].encode(
            [output],
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )

    X = np.concatenate(
        [qemb, oemb],
        axis=1,
    )

    P = base.vb.visible_probs(
        pipe["fp_models"],
        pipe["fp_cals"],
        X,
    )[0]

    return {
        role: float(p)
        for role, p in zip(
            ROLES,
            P,
        )
    }


# ============================================================
# Aggregation
# ============================================================

def aggregate_system(
    rows,
    true_roles,
    threshold,
):
    by_role = defaultdict(list)

    for row in rows:
        by_role[
            row["target_role"]
        ].append(row)

    missing = [
        r
        for r in ROLES
        if len(by_role[r])
        != PROBES_PER_ROLE
    ]

    if missing:
        raise RuntimeError(
            "Incomplete role collection: "
            + repr(
                {
                    r: len(by_role[r])
                    for r in missing
                }
            )
        )

    aggregators = {
        "mean": {},
        "median": {},
        "top3_mean": {},
        "max": {},
        "threshold_rate": {},
    }

    role_details = {}

    for role in ROLES:
        rr = sorted(
            by_role[role],
            key=lambda x:
                int(x["probe_index"]),
        )

        scores = np.asarray(
            [
                float(x["target_score"])
                for x in rr
            ],
            dtype=float,
        )

        top3 = np.sort(
            scores
        )[-3:]

        activations = [
            role
            in set(
                x.get(
                    "selected_roles_eval_only",
                    [],
                )
            )
            for x in rr
        ]

        aggregators[
            "mean"
        ][role] = float(
            np.mean(scores)
        )

        aggregators[
            "median"
        ][role] = float(
            np.median(scores)
        )

        aggregators[
            "top3_mean"
        ][role] = float(
            np.mean(top3)
        )

        aggregators[
            "max"
        ][role] = float(
            np.max(scores)
        )

        aggregators[
            "threshold_rate"
        ][role] = float(
            np.mean(
                scores >= threshold
            )
        )

        role_details[role] = {
            "n":
                len(scores),

            "scores":
                scores.tolist(),

            "mean":
                float(np.mean(scores)),

            "median":
                float(np.median(scores)),

            "std":
                float(np.std(scores)),

            "min":
                float(np.min(scores)),

            "max":
                float(np.max(scores)),

            "threshold_rate":
                float(
                    np.mean(
                        scores >= threshold
                    )
                ),

            # Evaluation/audit only.
            "activation_rate_eval_only":
                float(
                    np.mean(
                        activations
                    )
                ),
        }

    metrics = {
        name:
            evaluate_scores(
                scores,
                true_roles,
            )
        for name, scores
        in aggregators.items()
    }

    truth = set(true_roles)

    installed_activation = [
        role_details[r][
            "activation_rate_eval_only"
        ]
        for r in ROLES
        if r in truth
    ]

    absent_activation = [
        role_details[r][
            "activation_rate_eval_only"
        ]
        for r in ROLES
        if r not in truth
    ]

    return {
        "primary_aggregator":
            "mean",

        "aggregators":
            aggregators,

        "metrics":
            metrics,

        "role_details":
            role_details,

        "audit_activation": {
            "installed_mean_activation_rate":
                float(
                    np.mean(
                        installed_activation
                    )
                ),

            "absent_mean_activation_rate":
                float(
                    np.mean(
                        absent_activation
                    )
                ),
        },
    }


# ============================================================
# One NEW CHALLENGE system
# ============================================================

def run_system(
    pipe,
    neighbors,
    system_index,
    sid,
    true_roles,
):
    print()
    print("#" * 120)
    print(
        f"NEW CHALLENGE "
        f"{system_index}/3: {sid}"
    )
    print("#" * 120)

    print(
        "Installed roles are hidden from "
        "probe generation and scoring."
    )

    print(
        f"{len(ROLES)} candidate roles × "
        f"{PROBES_PER_ROLE} probes = "
        f"{len(ROLES) * PROBES_PER_ROLE} "
        "queries."
    )

    target = make_target(
        sid,
        true_roles,
    )

    journal = (
        OUT
        / f"{sid}_observations.jsonl"
    )

    existing = read_jsonl(
        journal
    )

    completed = {
        (
            str(x["target_role"]),
            int(x["probe_index"]),
        )
        for x in existing
    }

    used_by_role = defaultdict(list)

    for x in existing:
        used_by_role[
            str(x["target_role"])
        ].append(
            str(x["query"])
        )

    print(
        f"RESUME: {len(existing)}/"
        f"{len(ROLES) * PROBES_PER_ROLE} "
        "observations already present."
    )


    for role_index, role in enumerate(
        ROLES
    ):
        print()
        print("=" * 100)
        print(
            f"ROLE {role_index + 1:02d}/20: "
            f"{role}"
        )
        print("=" * 100)

        for probe_index in range(
            1,
            PROBES_PER_ROLE + 1,
        ):
            key = (
                role,
                probe_index,
            )

            if key in completed:
                print(
                    f"  probe "
                    f"{probe_index:02d}/10 "
                    "already complete"
                )
                continue

            seed_base = (
                int(base.SEED)
                + 100_000_000
                * system_index
                + 100_000
                * role_index
                + 100
                * probe_index
            )

            query = generate_probe(
                pipe["policy"],
                role,
                neighbors,
                used_by_role[role],
                seed_base,
            )

            print()
            print(
                f"[{role}] "
                f"probe "
                f"{probe_index:02d}/10"
            )

            print(
                "QUERY:",
                query,
            )

            output = query_target(
                target,
                query,
            )

            scores = visible_sensor(
                pipe,
                query,
                output,
            )

            target_score = float(
                scores[role]
            )

            # Evaluation-only router trace.
            # Read only AFTER output + sensor score exist.
            tr = getattr(
                target,
                "_last_trace",
                None,
            )

            selected_eval = (
                list(
                    map(
                        str,
                        getattr(
                            tr,
                            "selected_roles",
                            (),
                        ),
                    )
                )
                if tr
                else []
            )

            row = {
                "system_id":
                    sid,

                "target_role":
                    role,

                "probe_index":
                    probe_index,

                "query":
                    query,

                "visible_output":
                    output,

                "target_score":
                    target_score,

                "all_sensor_scores":
                    scores,

                "selected_roles_eval_only":
                    selected_eval,
            }

            append_jsonl(
                journal,
                row,
            )

            existing.append(
                row
            )

            completed.add(
                key
            )

            used_by_role[
                role
            ].append(
                query
            )

            print(
                f"TARGET SCORE = "
                f"{target_score:.5f}"
            )

            print(
                "ACTIVATED AUDIT ONLY =",
                role in set(
                    selected_eval
                ),
            )


    # ========================================================
    # Evaluation only after all 200 observations exist
    # ========================================================

    result = aggregate_system(
        existing,
        true_roles,
        float(
            pipe["threshold"]
        ),
    )

    result.update(
        {
            "system_id":
                sid,

            "true_roles":
                list(
                    true_roles
                ),

            "true_k":
                len(
                    true_roles
                ),

            "candidate_roles":
                list(
                    ROLES
                ),

            "probes_per_role":
                PROBES_PER_ROLE,

            "total_queries":
                len(ROLES)
                * PROBES_PER_ROLE,

            "protocol":
                (
                    "Non-adaptive deep diagnostic. "
                    "Ten distinct V14 targeted probes "
                    "for every candidate role. "
                    "Visible query+response sensor only. "
                    "No MI selector, no Bayes update, "
                    "no K feedback."
                ),
        }
    )

    dump_json(
        OUT
        / f"{sid}_summary.json",
        result,
    )

    print()
    print("#" * 120)
    print(
        f"FINAL — {sid}"
    )
    print("#" * 120)

    print(
        "TRUE ROLES:",
        sorted(true_roles),
    )

    for name in (
        "mean",
        "median",
        "top3_mean",
        "max",
        "threshold_rate",
    ):
        m = result[
            "metrics"
        ][name]

        print(
            f"{name:<18} "
            f"AUC={m['auc']:.4f}  "
            f"Top3={m['top3']}  "
            f"Exact={m['top3_exact']}"
        )

    print(
        "Installed target activation rate "
        "(AUDIT ONLY) = "
        f"{result['audit_activation']['installed_mean_activation_rate']:.4f}"
    )

    return result


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 120)
    print(
        "V14 DEEP ROLE NEW CHALLENGE V1"
    )
    print("=" * 120)

    print(
        "3 new MAS compositions"
    )
    print(
        "20 roles × 10 probes each"
    )
    print(
        "600 maximum real MAS calls"
    )
    print(
        "PRIMARY aggregation fixed in advance: "
        "MEAN target sensor score"
    )
    print(
        "No adaptive selector in this diagnostic"
    )

    pipe = base.load_pipeline()

    if "policy" not in pipe:
        raise RuntimeError(
            "load_pipeline() did not expose policy"
        )

    neighbors = build_semantic_neighbors(
        pipe["embedder"]
    )

    print()
    print(
        "FIXED SEMANTIC CONFUSERS"
    )

    for role in ROLES:
        print(
            f"{role:<28} "
            f"-> {neighbors[role]}"
        )

    results = []

    for i, (
        sid,
        true_roles,
    ) in enumerate(
        CHALLENGES,
        1,
    ):
        summary_path = (
            OUT
            / f"{sid}_summary.json"
        )

        if summary_path.exists():
            print()
            print(
                f"RESUME: {sid} "
                "already complete"
            )

            result = json.loads(
                summary_path.read_text(
                    encoding="utf-8"
                )
            )

        else:
            result = run_system(
                pipe,
                neighbors,
                i,
                sid,
                true_roles,
            )

        results.append(
            result
        )


    print()
    print("=" * 120)
    print(
        "THREE-SYSTEM FINAL SUMMARY"
    )
    print("=" * 120)

    aggregate = {}

    for method in (
        "mean",
        "median",
        "top3_mean",
        "max",
        "threshold_rate",
    ):
        aucs = [
            float(
                r["metrics"][method]["auc"]
            )
            for r in results
        ]

        exact = sum(
            int(
                r["metrics"][method][
                    "top3_exact"
                ]
            )
            for r in results
        )

        aggregate[
            method
        ] = {
            "mean_auc":
                float(np.mean(aucs)),

            "median_auc":
                float(np.median(aucs)),

            "top3_exact":
                exact,
        }

        print(
            f"{method:<18} "
            f"meanAUC="
            f"{np.mean(aucs):.4f}  "
            f"medianAUC="
            f"{np.median(aucs):.4f}  "
            f"Top3Exact="
            f"{exact}/3"
        )


    installed_act = [
        float(
            r[
                "audit_activation"
            ][
                "installed_mean_activation_rate"
            ]
        )
        for r in results
    ]

    absent_act = [
        float(
            r[
                "audit_activation"
            ][
                "absent_mean_activation_rate"
            ]
        )
        for r in results
    ]

    print()
    print(
        "Installed target activation "
        "(AUDIT ONLY) = "
        f"{np.mean(installed_act):.4f}"
    )

    print(
        "Absent target activation "
        "(AUDIT ONLY) = "
        f"{np.mean(absent_act):.4f}"
    )

    dump_json(
        OUT
        / "aggregate_summary.json",
        {
            "challenge_systems": [
                {
                    "system_id": sid,
                    "true_roles":
                        list(true_roles),
                }
                for sid, true_roles
                in CHALLENGES
            ],

            "primary_aggregator":
                "mean",

            "aggregate":
                aggregate,

            "audit_activation": {
                "installed_mean":
                    float(
                        np.mean(
                            installed_act
                        )
                    ),

                "absent_mean":
                    float(
                        np.mean(
                            absent_act
                        )
                    ),
            },
        },
    )


if __name__ == "__main__":
    main()
