from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

import analysis.ten_system_belief_trace_control_v14_dropin as base

from v14.belief_dropin_policy import V14BeliefDropInPolicy
from v11.role_universe import (
    ROLE_DESCRIPTIONS,
    anchor_probe,
)


OUT = Path(
    "artifacts/v14_full_role_sweep_v1"
)
OUT.mkdir(
    parents=True,
    exist_ok=True,
)

ROLES = tuple(base.ROLES)

SCREEN_REPEATS = 1
CONFIRM_TOP = 6

TEMPERATURE = 0.90
TOP_P = 0.92
MAX_NEW_TOKENS = 192
MAX_RETRIES = 12


# ============================================================
# Utilities
# ============================================================

def write_json(path: Path, obj):
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


def native_v14_sample(
    policy,
    observation,
    **kwargs,
):
    """
    Bypass V14BeliefDropInPolicy's belief->role scheduler.

    Use the underlying native V13/V14 role-conditioned prompt
    with the V14 adapter weights already loaded in `policy`.
    """
    return super(
        V14BeliefDropInPolicy,
        policy,
    ).sample_probe(
        observation,
        **kwargs,
    )


def build_fixed_neighbors(
    embedder,
):
    texts = [
        ROLE_DESCRIPTIONS[r]
        for r in ROLES
    ]

    V = np.asarray(
        embedder.encode(
            texts,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )

    norms = np.linalg.norm(
        V,
        axis=1,
        keepdims=True,
    )

    V = V / np.maximum(
        norms,
        1e-12,
    )

    S = V @ V.T

    result = {}

    for i, role in enumerate(ROLES):
        order = np.argsort(
            -S[i]
        )

        neighbors = [
            ROLES[j]
            for j in order
            if j != i
        ][:4]

        result[role] = neighbors

    return result


def generate_probe(
    policy,
    role,
    neighbors,
    avoid,
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

        # V14 training/evaluation was effectively
        # query-only here. Do not introduce response-conditioned
        # history in this experiment.
        "public_history":
            [],

        "avoid_probes":
            list(avoid)[-6:],
    }

    last = None

    for retry in range(
        MAX_RETRIES
    ):
        seed = (
            seed_base
            + retry
        )

        random.seed(
            seed
        )

        torch.manual_seed(
            seed
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                seed
            )

        sample = native_v14_sample(
            policy,
            observation,

            temperature=
                TEMPERATURE,

            top_p=
                TOP_P,

            max_new_tokens=
                MAX_NEW_TOKENS,
        )

        q = str(
            sample.query
            or ""
        ).strip()

        ok, why = base.surface_valid(
            q
        )

        valid = (
            bool(sample.valid)
            and ok
            and q
            and q not in avoid
        )

        last = (
            q,
            valid,
            why,
        )

        if valid:
            return q

    raise RuntimeError(
        f"Could not generate valid probe "
        f"for role={role}. "
        f"last={last}"
    )


def sensor_scores(
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
        [
            qemb,
            oemb,
        ],
        axis=1,
    )

    P = base.vb.visible_probs(
        pipe["fp_models"],
        pipe["fp_cals"],
        X,
    )[0]

    return {
        role: float(p)
        for role, p
        in zip(
            ROLES,
            P,
        )
    }


def make_target(
    sid,
    true_roles,
):
    spec = base.SingleRouterSystemSpec(
        system_id=
            sid,

        split=
            "transparent_control",

        execution_mode=
            "mixed",

        style=
            "s1",

        worker_roles=
            tuple(
                true_roles
            ),
    )

    return base.SingleRouterFanoutMAS(
        spec
    )


def gap_k(
    scores,
):
    """
    Truth-free K heuristic.

    Candidate K is restricted to the historical
    K universe {3,4,5,6}. Pick the largest score
    drop after rank k.
    """

    ranked = sorted(
        scores.items(),
        key=lambda x: -x[1],
    )

    candidates = []

    for k in (3, 4, 5, 6):
        if k >= len(ranked):
            continue

        gap = (
            ranked[k - 1][1]
            - ranked[k][1]
        )

        candidates.append(
            (
                gap,
                k,
            )
        )

    return max(
        candidates
    )[1]


# ============================================================
# One system
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
        f"SYSTEM {system_index:02d}/10: "
        f"{sid}"
    )
    print("#" * 120)

    print(
        "Inference protocol:"
    )
    print(
        "  Phase 1: one targeted V14 probe "
        "for EACH of 20 candidate roles"
    )
    print(
        "  Phase 2: independent confirmation "
        f"for top-{CONFIRM_TOP}"
    )
    print(
        "  No historical ObservationNet / MI"
    )
    print(
        "  No historical Bayes update"
    )
    print(
        "  No K feedback during probing"
    )
    print(
        "  Runtime input = query + visible final response"
    )

    target = make_target(
        sid,
        true_roles,
    )

    records = []

    used_queries = set()

    screen_scores = {}

    first_probe = {}


    # ========================================================
    # PHASE 1 — COMPLETE ROLE SWEEP
    # ========================================================

    print()
    print("=" * 120)
    print(
        "PHASE 1 — COMPLETE 20-ROLE SCREENING"
    )
    print("=" * 120)

    for role_index, role in enumerate(
        ROLES
    ):
        seed_base = (
            base.SEED
            + 10_000_000
            * system_index
            + 10_000
            * role_index
        )

        q = generate_probe(
            pipe["policy"],
            role,
            neighbors,
            used_queries,
            seed_base,
        )

        used_queries.add(
            q
        )

        first_probe[
            role
        ] = q

        print()
        print(
            f"[SCREEN {role_index+1:02d}/20] "
            f"TARGET = {role}"
        )

        print(
            "QUERY:"
        )
        print(
            q
        )

        output = str(
            target.query(
                q
            )
        )

        scores = sensor_scores(
            pipe,
            q,
            output,
        )

        target_score = float(
            scores[role]
        )

        screen_scores[
            role
        ] = target_score

        # Audit only. Never used in inference.
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

        rec = {
            "phase":
                "screen",

            "target_role":
                role,

            "query":
                q,

            "visible_output":
                output,

            "target_score":
                target_score,

            "all_sensor_scores":
                scores,

            "selected_roles_eval_only":
                selected_eval,
        }

        records.append(
            rec
        )

        write_json(
            OUT
            / f"{sid}_trajectory.json",
            records,
        )

        print(
            f"TARGET SENSOR SCORE = "
            f"{target_score:.5f}"
        )

        top5 = sorted(
            scores.items(),
            key=lambda x:
                -x[1],
        )[:5]

        print(
            "SENSOR TOP-5 = "
            + ", ".join(
                f"{r}:{p:.3f}"
                for r, p in top5
            )
        )


    screening_ranking = sorted(
        ROLES,
        key=lambda r:
            -screen_scores[r],
    )

    confirm_roles = (
        screening_ranking[
            :CONFIRM_TOP
        ]
    )

    print()
    print("=" * 120)
    print(
        "PHASE 1 RANKING"
    )
    print("=" * 120)

    for i, role in enumerate(
        screening_ranking,
        1,
    ):
        print(
            f"{i:02d}. "
            f"{role:<28} "
            f"{screen_scores[role]:.5f}"
        )

    print()
    print(
        "CONFIRMING:",
        confirm_roles,
    )


    # ========================================================
    # PHASE 2 — CONFIRM TOP-6
    # ========================================================

    confirm_scores = {}

    print()
    print("=" * 120)
    print(
        "PHASE 2 — TOP-6 CONFIRMATION"
    )
    print("=" * 120)

    for confirm_index, role in enumerate(
        confirm_roles
    ):
        seed_base = (
            base.SEED
            + 50_000_000
            + 10_000_000
            * system_index
            + 10_000
            * confirm_index
        )

        avoid = set(
            used_queries
        )

        avoid.add(
            first_probe[role]
        )

        q = generate_probe(
            pipe["policy"],
            role,
            neighbors,
            avoid,
            seed_base,
        )

        used_queries.add(
            q
        )

        print()
        print(
            f"[CONFIRM "
            f"{confirm_index+1:02d}/{CONFIRM_TOP}] "
            f"TARGET = {role}"
        )

        print(
            "QUERY:"
        )
        print(
            q
        )

        output = str(
            target.query(
                q
            )
        )

        scores = sensor_scores(
            pipe,
            q,
            output,
        )

        target_score = float(
            scores[role]
        )

        confirm_scores[
            role
        ] = target_score

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

        rec = {
            "phase":
                "confirm",

            "target_role":
                role,

            "query":
                q,

            "visible_output":
                output,

            "target_score":
                target_score,

            "all_sensor_scores":
                scores,

            "selected_roles_eval_only":
                selected_eval,
        }

        records.append(
            rec
        )

        write_json(
            OUT
            / f"{sid}_trajectory.json",
            records,
        )

        print(
            "SCREEN SCORE   = "
            f"{screen_scores[role]:.5f}"
        )

        print(
            "CONFIRM SCORE  = "
            f"{target_score:.5f}"
        )

        print(
            "MEAN           = "
            f"{0.5 * (screen_scores[role] + target_score):.5f}"
        )


    # ========================================================
    # FINAL SCORE
    # ========================================================

    final_scores = dict(
        screen_scores
    )

    for role in confirm_roles:
        final_scores[
            role
        ] = 0.5 * (
            screen_scores[role]
            + confirm_scores[role]
        )


    # Truth is used only from this point for evaluation.
    metrics = base.eval_final(
        true_roles,
        final_scores,
    )

    inferred_k = gap_k(
        final_scores
    )

    metrics[
        "predicted_k"
    ] = inferred_k

    metrics[
        "predicted_k_correct"
    ] = (
        inferred_k
        == len(true_roles)
    )


    print()
    print("#" * 120)
    print(
        f"FINAL REVEAL — {sid}"
    )
    print("#" * 120)

    print(
        "TRUE INSTALLED ROLES:",
        sorted(
            true_roles
        ),
    )

    print()
    print(
        "FINAL ROLE RANKING:"
    )

    truth_set = set(
        true_roles
    )

    for rank, role in enumerate(
        metrics["ranking"],
        1,
    ):
        mark = (
            "  <-- INSTALLED"
            if role in truth_set
            else ""
        )

        confirmation = (
            f" screen={screen_scores[role]:.4f}"
            f" confirm={confirm_scores[role]:.4f}"
            if role in confirm_scores
            else ""
        )

        print(
            f"{rank:02d}. "
            f"{role:<28} "
            f"{final_scores[role]:.5f}"
            f"{confirmation}"
            f"{mark}"
        )

    print()
    print(
        f"AUC = "
        f"{metrics['belief_auc']:.4f}"
    )

    print(
        "KNOWN-K TOP3 =",
        metrics[
            "oracle_k_top3"
        ],
    )

    print(
        "KNOWN-K TOP3 EXACT =",
        metrics[
            "oracle_k_top3_exact"
        ],
    )

    print(
        "GAP-HEURISTIC K =",
        inferred_k,
    )

    result = {
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

        "protocol":
            {
                "screen_queries":
                    len(ROLES),

                "confirmation_queries":
                    CONFIRM_TOP,

                "total_queries":
                    len(ROLES)
                    + CONFIRM_TOP,

                "selector":
                    "none_complete_role_sweep",

                "sensor":
                    "historical_visible_qo_detector",

                "score":
                    (
                        "target sensor score; "
                        "confirmed top roles use mean of "
                        "screen + independent confirmation"
                    ),

                "k_estimator":
                    "largest score gap among k=3..6",
            },

        "screen_scores":
            screen_scores,

        "confirmation_scores":
            confirm_scores,

        "final_role_probs":
            final_scores,

        "predicted_k":
            inferred_k,

        **metrics,
    }

    write_json(
        OUT
        / f"{sid}_summary.json",
        result,
    )

    return result


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 120)
    print(
        "V14 COMPLETE ROLE SWEEP + CONFIRMATION"
    )
    print("=" * 120)

    print(
        "Loading V14 policy + historical visible sensor..."
    )

    pipe = base.load_pipeline()

    print(
        "Building fixed semantic confusers..."
    )

    neighbors = build_fixed_neighbors(
        pipe["embedder"]
    )

    print()
    print(
        "FIXED CONFUSERS:"
    )

    for role in ROLES:
        print(
            f"{role:<28} "
            f"→ {neighbors[role]}"
        )

    results = []

    for i, (
        sid,
        true_roles,
    ) in enumerate(
        base.SYSTEMS,
        1,
    ):

        summary_path = (
            OUT
            / f"{sid}_summary.json"
        )

        if summary_path.exists():
            print()
            print(
                f"RESUME: {sid} already completed."
            )

            result = json.loads(
                summary_path.read_text()
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


    # Rewrite deterministic systems file.
    with (
        OUT
        / "systems.jsonl"
    ).open(
        "w"
    ) as f:
        for r in results:
            f.write(
                json.dumps(
                    r,
                    ensure_ascii=False,
                )
                + "\n"
            )


    aucs = [
        float(
            r[
                "belief_auc"
            ]
        )
        for r in results
    ]

    exact = sum(
        int(
            r[
                "oracle_k_top3_exact"
            ]
        )
        for r in results
    )

    kexact = sum(
        int(
            r[
                "predicted_k_correct"
            ]
        )
        for r in results
    )

    print()
    print("=" * 120)
    print(
        "TEN-SYSTEM FINAL SUMMARY"
    )
    print("=" * 120)

    for r in results:
        print(
            f"{r['system_id']:<36} "
            f"AUC={r['belief_auc']:.4f} "
            f"Top3Exact={r['oracle_k_top3_exact']} "
            f"Khat={r['predicted_k']}"
        )

    print()
    print(
        f"MEAN AUC = "
        f"{float(np.mean(aucs)):.4f}"
    )

    print(
        f"MEDIAN AUC = "
        f"{float(np.median(aucs)):.4f}"
    )

    print(
        f"KNOWN-K TOP3 EXACT = "
        f"{exact}/{len(results)}"
    )

    print(
        f"GAP-K EXACT = "
        f"{kexact}/{len(results)}"
    )


if __name__ == "__main__":
    main()
