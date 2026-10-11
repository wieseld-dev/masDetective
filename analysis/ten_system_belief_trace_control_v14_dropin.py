#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer


sys.path.insert(
    0,
    str(Path(".").resolve()),
)

sys.path.insert(
    0,
    str(Path("analysis").resolve()),
)


import observation_model_v1 as om
import visible_adaptive_bridge_v1 as vb

from adaptive_v11_active.reward import surface_valid
from adaptive_v11_active.schema import PublicBeliefState

from v14.belief_dropin_policy import V14BeliefDropInPolicy

from targets.single_router_zoo_v11 import (
    validate_v11_design,
)

validate_v11_design()

from targets.single_router_fanout_mas import (
    SingleRouterFanoutMAS,
    SingleRouterSystemSpec,
)

from v11.role_universe import TRAIN_ROLES


OUT = Path(
    "artifacts/ten_system_belief_trace_control_v14_dropin"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


ROLES = list(
    TRAIN_ROLES
)

assert len(ROLES) == 20


KS = (
    3,
    4,
    5,
    6,
)


SYSTEMS = [
    (
        "mas01_code_plan_research",
        (
            "code",
            "planning",
            "research",
        ),
    ),

    (
        "mas02_risk_access_translation",
        (
            "risk_assessment",
            "accessibility_review",
            "translation",
        ),
    ),

    (
        "mas03_data_db_requirements",
        (
            "data",
            "database_design",
            "requirements_analysis",
        ),
    ),

    (
        "mas04_support_incident_ops",
        (
            "customer_support_triage",
            "incident_management",
            "operations",
        ),
    ),

    (
        "mas05_docs_qa_writing",
        (
            "documentation",
            "qa_testing",
            "writing",
        ),
    ),

    (
        "mas06_math_design_experiment",
        (
            "math",
            "design",
            "experiment_design",
        ),
    ),

    (
        "mas07_fact_finance_research",
        (
            "fact_checking",
            "financial_analysis",
            "research",
        ),
    ),

    (
        "mas08_access_db_ops",
        (
            "accessibility_review",
            "database_design",
            "operations",
        ),
    ),

    (
        "mas09_plan_translation_docs",
        (
            "planning",
            "translation",
            "documentation",
        ),
    ),

    (
        "mas10_code_risk_support",
        (
            "code",
            "risk_assessment",
            "customer_support_triage",
        ),
    ),
]


for sid, rs in SYSTEMS:
    unknown = (
        set(rs)
        - set(ROLES)
    )

    if unknown:
        raise RuntimeError(
            f"{sid}: unknown roles {unknown}"
        )

    assert len(rs) == 3
    assert len(set(rs)) == 3


ROUNDS = 8
GROUP_SIZE = 8

PARTICLES_PER_K = 128

SEED = 20260927


def norm_weights(w):
    w = np.asarray(
        w,
        dtype=float,
    )

    return (
        w
        / max(
            w.sum(),
            1e-12,
        )
    )


def get_k_probs(
    W,
    Klabels,
):
    return {
        str(k):
            float(
                W[
                    Klabels == k
                ].sum()
            )
        for k in KS
    }


def get_role_probs(
    W,
    Z,
):
    p = W @ Z

    return {
        r:
            float(
                p[i]
            )
        for i, r
        in enumerate(ROLES)
    }


def make_public_state(
    sid,
    rnd,
    kp,
    rp,
    history,
):
    return {
        "session_id":
            sid,

        "split":
            "dev",

        "round_index":
            rnd,

        "k_probs":
            kp,

        "role_probs":
            rp,

        "history":
            history,
    }


def print_belief(
    title,
    kp,
    rp,
):
    print()
    print("=" * 108)
    print(title)
    print("=" * 108)

    print(
        "P(K):",
        "  ".join(
            f"K={k}: "
            f"{kp.get(str(k), 0.0):.4f}"
            for k in KS
        ),
    )

    print()
    print(
        "ROLE BELIEFS "
        "(NO TRUTH MARKERS DURING INFERENCE)"
    )

    for rank, (
        role,
        p,
    ) in enumerate(
        sorted(
            rp.items(),
            key=lambda x:
                -x[1],
        ),
        1,
    ):
        print(
            f"{rank:02d}. "
            f"{role:<28} "
            f"{p:.5f}"
        )


def eval_final(
    truth,
    rp,
):
    truth = set(
        truth
    )

    yy = np.asarray(
        [
            int(
                r in truth
            )
            for r in ROLES
        ],
        dtype=int,
    )

    ss = np.asarray(
        [
            rp[r]
            for r in ROLES
        ],
        dtype=float,
    )

    auc = float(
        roc_auc_score(
            yy,
            ss,
        )
    )


    true_scores = [
        rp[r]
        for r in truth
    ]

    false_scores = [
        rp[r]
        for r in ROLES
        if r not in truth
    ]


    correct = 0
    ties = 0
    total = 0

    for tr in truth:
        for fr in ROLES:
            if fr in truth:
                continue

            total += 1

            if rp[tr] > rp[fr]:
                correct += 1

            elif rp[tr] == rp[fr]:
                ties += 1


    pair_acc = (
        correct
        + 0.5 * ties
    ) / total


    perfect_ordering = (
        min(
            true_scores
        )
        >
        max(
            false_scores
        )
    )


    ranking = sorted(
        ROLES,
        key=lambda r:
            -rp[r],
    )


    top3 = ranking[:3]


    return {
        "belief_auc":
            auc,

        "present_vs_absent_correct":
            correct,

        "present_vs_absent_ties":
            ties,

        "present_vs_absent_total":
            total,

        "present_vs_absent_pair_accuracy":
            pair_acc,

        "perfect_present_above_absent":
            bool(
                perfect_ordering
            ),

        "oracle_k_top3":
            top3,

        "oracle_k_top3_exact":
            set(top3)
            == truth,

        "min_true_belief":
            float(
                min(
                    true_scores
                )
            ),

        "max_false_belief":
            float(
                max(
                    false_scores
                )
            ),

        "ranking":
            ranking,
    }


def load_pipeline():
    print(
        "Loading historical "
        "visible-belief pipeline...",
        flush=True,
    )

    root = Path(
        "artifacts/adaptive_v11_curriculum_v2"
    )

    obsdir = Path(
        "artifacts/observation_model_v1"
    )

    visdir = Path(
        "artifacts/visible_output_fingerprint_v1"
    )

    caldir = Path(
        "artifacts/calibrated_bayes_diag_v1"
    )


    examples, obs_roles, _, _ = (
        om.collect_examples(
            root
        )
    )


    if list(obs_roles) != ROLES:
        raise RuntimeError(
            "role ordering mismatch\n"
            f"obs={obs_roles}\n"
            f"roles={ROLES}"
        )


    split = json.loads(
        (
            obsdir
            / "split_systems.json"
        ).read_text()
    )


    train_systems = set(
        map(
            str,
            split["train"],
        )
    )

    val_systems = set(
        map(
            str,
            split["val"],
        )
    )


    tri = np.asarray(
        [
            i
            for i, x
            in enumerate(
                examples
            )
            if str(
                x["system_id"]
            )
            in train_systems
        ],
        dtype=int,
    )


    vai = np.asarray(
        [
            i
            for i, x
            in enumerate(
                examples
            )
            if str(
                x["system_id"]
            )
            in val_systems
        ],
        dtype=int,
    )


    Qhist = np.load(
        obsdir
        / "query_embeddings.npy"
    ).astype(
        np.float32
    )

    Ohist = np.load(
        visdir
        / "output_embeddings.npy"
    ).astype(
        np.float32
    )


    Xhist = np.concatenate(
        [
            Qhist,
            Ohist,
        ],
        axis=1,
    )


    Yhist = vb.labels(
        examples,
        ROLES,
        "selected_roles",
    )


    (
        fp_models,
        fp_cals,
        threshold,
        tpr,
        fpr,
        _,
    ) = vb.fit_visible_detector(
        Xhist,
        Yhist,
        tri,
        vai,
    )


    embedder = SentenceTransformer(
        "sentence-transformers/"
        "all-MiniLM-L6-v2"
    )


    ckpt = torch.load(
        obsdir
        / "model.pt",
        map_location="cpu",
    )


    fwd = om.ObservationNet(
        qdim=
            ckpt[
                "query_embedding_dim"
            ],

        n_roles=
            len(
                ROLES
            ),

        hidden=
            ckpt[
                "hidden"
            ],

        dropout=
            ckpt[
                "dropout"
            ],
    )


    fwd.load_state_dict(
        ckpt[
            "state_dict"
        ]
    )

    fwd.eval()


    ri = {
        r: i
        for i, r
        in enumerate(
            ROLES
        )
    }


    pos = np.zeros(
        len(
            ROLES
        ),
        dtype=float,
    )


    for i in tri:
        for r in examples[
            i
        ][
            "selected_roles"
        ]:
            pos[
                ri[r]
            ] += 1


    posw = np.clip(
        (
            len(tri)
            - pos
        )
        / np.maximum(
            pos,
            1.0,
        ),
        1.0,
        20.0,
    )


    log_posw = np.log(
        posw
    )


    cs = json.loads(
        (
            caldir
            / "summary.json"
        ).read_text()
    )


    T = float(
        cs[
            "true_Z_probability_calibration"
        ][
            "corrected_plus_temperature"
        ][
            "best_T"
        ]
    )


    beta = float(
        cs[
            "validation_posterior_update"
        ][
            "selected_beta_by_validation_log_gain"
        ]
    )


    policy = (
        V14BeliefDropInPolicy(
            "Qwen/Qwen2.5-1.5B-Instruct",

            adapter_path=
                "artifacts/"
                "v14_benign_policy",

            trainable=False,

            use_4bit=False,

            max_seq_length=1536,
        )
    )


    max_round = max(
        int(
            x["round"]
        )
        for x in examples
    )


    return {
        "examples":
            examples,

        "embedder":
            embedder,

        "fp_models":
            fp_models,

        "fp_cals":
            fp_cals,

        "threshold":
            threshold,

        "tpr":
            tpr,

        "fpr":
            fpr,

        "fwd":
            fwd,

        "log_posw":
            log_posw,

        "T":
            T,

        "beta":
            beta,

        "policy":
            policy,

        "max_round":
            max_round,
    }


def initial_particles(
    system_seed,
):
    # Inference does NOT know true K.
    #
    # P(K=3)=P(K=4)=P(K=5)=P(K=6)=0.25
    kp0 = {
        str(k):
            1.0 / len(KS)
        for k in KS
    }


    expected_k = sum(
        k
        * kp0[
            str(k)
        ]
        for k in KS
    )


    marginal = (
        expected_k
        / len(
            ROLES
        )
    )


    rp0 = {
        r:
            marginal
        for r in ROLES
    }


    rng = np.random.default_rng(
        system_seed
    )


    (
        Z,
        W,
        Klabels,
        _,
    ) = vb.stratified_particles(
        kp0,
        rp0,
        ROLES,
        PARTICLES_PER_K,
        rng,
    )


    W = norm_weights(
        W
    )


    return (
        Z,
        W,
        Klabels,
    )


def run_system(
    pipe,
    system_index,
    sid,
    true_roles,
):
    print()
    print("#" * 120)
    print(
        f"SYSTEM {system_index:02d}/10: {sid}"
    )
    print("#" * 120)

    print(
        "Inference protocol:"
    )

    print(
        "  candidate universe = 20 roles"
    )

    print(
        "  true K is NOT supplied"
    )

    print(
        "  true installed roles are NOT supplied "
        "to generator / selector / fingerprint / posterior"
    )

    print(
        "  runtime observable = "
        "query + final response only"
    )

    print(
        "  hidden router may activate "
        "1, 2, or 3 workers"
    )

    print(
        "  truth is revealed only AFTER "
        "all 8 rounds"
    )


    spec = SingleRouterSystemSpec(
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


    # The inference code gets only .query(...)
    # from this black-box target.
    target = SingleRouterFanoutMAS(
        spec
    )


    (
        Z,
        W,
        Klabels,
    ) = initial_particles(
        SEED
        + 10000
        * system_index
    )


    history = []
    records = []


    kp = get_k_probs(
        W,
        Klabels,
    )

    rp = get_role_probs(
        W,
        Z,
    )


    print_belief(
        "INITIAL PUBLIC BELIEF",
        kp,
        rp,
    )


    for rnd in range(
        1,
        ROUNDS + 1,
    ):
        kp_before = get_k_probs(
            W,
            Klabels,
        )

        rp_before = get_role_probs(
            W,
            Z,
        )


        print()
        print("-" * 120)
        print(
            f"{sid} — ROUND {rnd}/{ROUNDS}"
        )
        print("-" * 120)


        print_belief(
            f"BELIEF BEFORE ROUND {rnd}",
            kp_before,
            rp_before,
        )


        public = make_public_state(
            sid,
            rnd - 1,
            kp_before,
            rp_before,
            history,
        )


        st = (
            PublicBeliefState
            .from_public_dict(
                public
            )
        )


        seen = [
            h.query
            for h in st.history
        ]


        candidates = []


        for j in range(
            GROUP_SIZE
        ):
            last = (
                "",
                False,
                "",
            )


            for retry in range(
                12
            ):
                sample_seed = (
                    SEED
                    + 1000000
                    * system_index
                    + 10000
                    * rnd
                    + 100
                    * j
                    + retry
                )


                random.seed(
                    sample_seed
                )

                torch.manual_seed(
                    sample_seed
                )


                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(
                        sample_seed
                    )


                sample = (
                    pipe[
                        "policy"
                    ]
                    .sample_probe(
                        {
                            "belief_state":
                                st.to_public_dict(),

                            "candidate_index":
                                j,

                            "max_history_turns":
                                5,
                        },

                        temperature=
                            0.85,

                        top_p=
                            0.95,

                        max_new_tokens=
                            192,
                    )
                )


                q = str(
                    sample.query
                    or ""
                ).strip()


                ok, why = surface_valid(
                    q
                )


                valid = (
                    bool(
                        sample.valid
                    )
                    and ok
                    and q not in seen
                    and q not in candidates
                )


                last = (
                    q,
                    valid,
                    why,
                )


                if valid:
                    break


            q, valid, why = last


            if valid:
                candidates.append(
                    q
                )


        if not candidates:
            raise RuntimeError(
                f"{sid}: generator produced "
                f"no valid candidates at round {rnd}"
            )


        Q = np.asarray(
            pipe[
                "embedder"
            ].encode(
                candidates,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )


        belief_vector = np.asarray(
            [
                rp_before[r]
                for r in ROLES
            ],
            dtype=np.float32,
        )


        k_vector = np.asarray(
            [
                kp_before[
                    str(k)
                ]
                for k in KS
            ],
            dtype=np.float32,
        )


        round_norm = float(
            rnd
            / max(
                pipe[
                    "max_round"
                ],
                1,
            )
        )


        L = vb.particle_logits(
            pipe[
                "fwd"
            ],

            Q,

            belief_vector,

            k_vector,

            round_norm,

            Z,

            "cpu",

            4096,
        )


        pY = vb.calibrated_hidden_probs(
            L,
            pipe[
                "log_posw"
            ],
            pipe[
                "T"
            ],
        )


        pE = vb.detector_channel_probs(
            pY,
            pipe[
                "tpr"
            ],
            pipe[
                "fpr"
            ],
        )


        mi = np.asarray(
            vb.mi_k(
                pE,
                W,
                Klabels,
                kp_before,
            ),
            dtype=float,
        )


        chosen = int(
            np.argmax(
                mi
            )
        )


        query = candidates[
            chosen
        ]


        print()
        print(
            "CANDIDATE QUERIES"
        )


        for j, (
            q,
            score,
        ) in enumerate(
            zip(
                candidates,
                mi,
            )
        ):
            mark = (
                "  <-- CHOSEN"
                if j == chosen
                else ""
            )

            print(
                f"\n[{j}] "
                f"expected information gain = "
                f"{score:.6f}"
                f"{mark}"
            )

            print(
                f"    {q}"
            )


        print()
        print(
            "QUERY SENT TO BLACK-BOX MAS:"
        )

        print(
            query
        )


        # =====================================================
        # ONLY BLACK-BOX RUNTIME CALL
        # =====================================================

        output = str(
            target.query(
                query
            )
        )


        print()
        print(
            "VISIBLE FINAL RESPONSE:"
        )

        print(
            output
        )


        # =====================================================
        # FINGERPRINT
        #
        # ONLY query + final output are used.
        # No truth and no router trace.
        # =====================================================

        qemb = np.asarray(
            pipe[
                "embedder"
            ].encode(
                [
                    query
                ],
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )


        oemb = np.asarray(
            pipe[
                "embedder"
            ].encode(
                [
                    output
                ],
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


        P = vb.visible_probs(
            pipe[
                "fp_models"
            ],

            pipe[
                "fp_cals"
            ],

            X,
        )[0]


        E = (
            P
            >= pipe[
                "threshold"
            ]
        ).astype(
            np.float64
        )


        print()
        print(
            "VISIBLE FINGERPRINT SCORES "
            "FROM QUERY + RESPONSE ONLY:"
        )


        for rank, (
            role,
            p,
        ) in enumerate(
            sorted(
                zip(
                    ROLES,
                    P,
                ),
                key=lambda x:
                    -x[1],
            ),
            1,
        ):
            print(
                f"{rank:02d}. "
                f"{role:<28} "
                f"{float(p):.5f}"
            )


        # =====================================================
        # POSTERIOR UPDATE
        #
        # This line is the internal BELIEF update.
        # Still NO truth / selected roles.
        # =====================================================

        W = vb.update_weights(
            pE[
                chosen
            ],

            E,

            W,

            pipe[
                "beta"
            ],
        )


        W = norm_weights(
            W
        )


        kp_after = get_k_probs(
            W,
            Klabels,
        )

        rp_after = get_role_probs(
            W,
            Z,
        )


        print_belief(
            f"BELIEF AFTER ROUND {rnd}",
            kp_after,
            rp_after,
        )


        # =====================================================
        # AUDIT-ONLY TRACE
        #
        # IMPORTANT:
        # Read only AFTER posterior was already updated.
        # It is never passed into selector/fingerprint/update.
        # =====================================================

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


        print()
        print(
            "AUDIT-ONLY ROUTING TRACE "
            "(READ AFTER UPDATE; NOT USED ABOVE):"
        )

        print(
            selected_eval
        )


        records.append({
            "round":
                rnd,

            "belief_before":
                rp_before,

            "k_probs_before":
                kp_before,

            "candidate_queries":
                candidates,

            "candidate_expected_mi":
                mi.tolist(),

            "chosen_index":
                chosen,

            "query":
                query,

            "final_output":
                output,

            "fingerprint_scores":
                {
                    r:
                        float(
                            P[i]
                        )
                    for i, r
                    in enumerate(
                        ROLES
                    )
                },

            "belief_after":
                rp_after,

            "k_probs_after":
                kp_after,

            "selected_roles_eval_only":
                selected_eval,
        })


        try:
            from adaptive_v11_active.schema import (
                HistoryFrame,
            )

            frame = HistoryFrame(
                query=
                    query,

                response=
                    output,
            )

            history = [
                *st.history,
                frame,
            ]

        except Exception:
            history = list(
                st.history
            )


        (
            OUT
            / f"{sid}_trajectory.json"
        ).write_text(
            json.dumps(
                records,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )


    # =========================================================
    # ONLY NOW REVEAL INSTALLED ROLES
    # =========================================================

    kp_final = get_k_probs(
        W,
        Klabels,
    )

    rp_final = get_role_probs(
        W,
        Z,
    )


    inferred_k = int(
        max(
            kp_final,
            key=
                kp_final.get,
        )
    )


    metrics = eval_final(
        true_roles,
        rp_final,
    )


    metrics[
        "predicted_k"
    ] = inferred_k

    metrics[
        "predicted_k_correct"
    ] = (
        inferred_k == 3
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
        "FINAL BELIEF RANKING:"
    )


    truth_set = set(
        true_roles
    )


    for rank, role in enumerate(
        metrics[
            "ranking"
        ],
        1,
    ):
        mark = (
            "  <-- INSTALLED"
            if role
            in truth_set
            else ""
        )

        print(
            f"{rank:02d}. "
            f"{role:<28} "
            f"{rp_final[role]:.5f}"
            f"{mark}"
        )


    print()

    print(
        "BELIEF AUC = "
        f"{metrics['belief_auc']:.4f}"
    )


    print(
        "PRESENT-vs-ABSENT ordering = "
        f"{metrics['present_vs_absent_correct']}"
        f"/"
        f"{metrics['present_vs_absent_total']}"
        " correct"
    )


    print(
        "PAIR ACCURACY = "
        f"{metrics['present_vs_absent_pair_accuracy']:.4f}"
    )


    print(
        "PERFECT ORDERING "
        "(all 3 installed above all 17 absent) =",
        metrics[
            "perfect_present_above_absent"
        ],
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
        "INFERRED K =",
        inferred_k,
    )


    print(
        "PREDICTED-K CORRECT =",
        metrics[
            "predicted_k_correct"
        ],
    )


    result = {
        "system_id":
            sid,

        "true_roles":
            list(
                true_roles
            ),

        "true_k":
            3,

        "rounds":
            ROUNDS,

        "group_size":
            GROUP_SIZE,

        "execution_description":
            (
                "hierarchical hidden router; "
                "router may activate 1, 2, or 3 workers"
            ),

        "runtime_observables":
            [
                "query",
                "final_response",
            ],

        "truth_usage":
            (
                "evaluation only after "
                "all inference rounds"
            ),

        "final_k_probs":
            kp_final,

        "final_role_probs":
            rp_final,

        **metrics,
    }


    (
        OUT
        / f"{sid}_summary.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


    return result


def main():
    if not os.getenv(
        "OPENAI_API_KEY"
    ):
        raise SystemExit(
            "OPENAI_API_KEY missing; "
            "source .env first"
        )


    print("=" * 120)
    print(
        "TEN-SYSTEM TRANSPARENT "
        "INSTALLED-ROLE BELIEF CONTROL"
    )
    print("=" * 120)

    print()
    print(
        "This is the historical visible-belief "
        "pipeline, rerun transparently."
    )

    print(
        "It is NOT the new neural generic ranker."
    )

    print()
    print(
        "During inference:"
    )

    print(
        "  - no installed-role truth"
    )

    print(
        "  - no true K"
    )

    print(
        "  - no routing trace"
    )

    print(
        "  - only query + final response"
    )

    print()
    print(
        "Truth is revealed only after "
        "8 rounds for evaluation."
    )


    pipe = load_pipeline()


    results = []


    for i, (
        sid,
        true_roles,
    ) in enumerate(
        SYSTEMS,
        1,
    ):
        result = run_system(
            pipe,
            i,
            sid,
            true_roles,
        )

        results.append(
            result
        )


        with (
            OUT
            / "systems.jsonl"
        ).open(
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )


    aucs = np.asarray(
        [
            x[
                "belief_auc"
            ]
            for x in results
        ],
        dtype=float,
    )


    pair_accs = np.asarray(
        [
            x[
                "present_vs_absent_pair_accuracy"
            ]
            for x in results
        ],
        dtype=float,
    )


    summary = {
        "experiment":
            "ten_system_belief_trace_control_v14_dropin",

        "systems":
            len(
                results
            ),

        "rounds_per_system":
            ROUNDS,

        "total_black_box_queries":
            len(
                results
            )
            * ROUNDS,

        "mean_belief_auc":
            float(
                aucs.mean()
            ),

        "median_belief_auc":
            float(
                np.median(
                    aucs
                )
            ),

        "mean_present_vs_absent_pair_accuracy":
            float(
                pair_accs.mean()
            ),

        "perfect_ordering_systems":
            int(
                sum(
                    x[
                        "perfect_present_above_absent"
                    ]
                    for x in results
                )
            ),

        "oracle_k_top3_exact_systems":
            int(
                sum(
                    x[
                        "oracle_k_top3_exact"
                    ]
                    for x in results
                )
            ),

        "predicted_k_exact_systems":
            int(
                sum(
                    x[
                        "predicted_k_correct"
                    ]
                    for x in results
                )
            ),

        "per_system":
            results,
    }


    (
        OUT
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


    print()
    print("=" * 120)
    print(
        "TEN-SYSTEM FINAL SUMMARY"
    )
    print("=" * 120)


    for x in results:
        print(
            f"{x['system_id']:<36} "
            f"AUC={x['belief_auc']:.4f} "
            f"pairAcc="
            f"{x['present_vs_absent_pair_accuracy']:.4f} "
            f"perfect="
            f"{x['perfect_present_above_absent']} "
            f"top3Exact="
            f"{x['oracle_k_top3_exact']} "
            f"Khat={x['predicted_k']}"
        )


    print()

    print(
        "MEAN BELIEF AUC =",
        f"{summary['mean_belief_auc']:.4f}",
    )

    print(
        "MEDIAN BELIEF AUC =",
        f"{summary['median_belief_auc']:.4f}",
    )

    print(
        "MEAN PRESENT-vs-ABSENT "
        "PAIR ACCURACY =",
        f"{summary['mean_present_vs_absent_pair_accuracy']:.4f}",
    )

    print(
        "PERFECT ORDERING SYSTEMS =",
        f"{summary['perfect_ordering_systems']}/10",
    )

    print(
        "KNOWN-K TOP3 EXACT =",
        f"{summary['oracle_k_top3_exact_systems']}/10",
    )

    print(
        "PREDICTED-K EXACT =",
        f"{summary['predicted_k_exact_systems']}/10",
    )

    print()

    print(
        "Saved:",
        OUT
        / "summary.json",
    )


if __name__ == "__main__":
    main()
