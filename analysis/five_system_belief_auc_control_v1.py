#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("analysis").resolve()))

import observation_model_v1 as om
import visible_adaptive_bridge_v1 as vb

from adaptive_v11_active.reward import surface_valid
from adaptive_v11_active.schema import PublicBeliefState
from adaptive_v11_curriculum.policy import CurriculumAdaptiveProbePolicyV11

from targets.single_router_zoo_v11 import validate_v11_design
validate_v11_design()

from targets.single_router_fanout_mas import (
    SingleRouterFanoutMAS,
    SingleRouterSystemSpec,
)

from v11.role_universe import TRAIN_ROLES


SYSTEMS = [
    (
        "ctrl_k3_01_code_plan_research",
        ("code", "planning", "research"),
    ),
    (
        "ctrl_k3_02_risk_access_translation",
        ("risk_assessment", "accessibility_review", "translation"),
    ),
    (
        "ctrl_k3_03_data_db_requirements",
        ("data", "database_design", "requirements_analysis"),
    ),
    (
        "ctrl_k3_04_support_incident_ops",
        ("customer_support_triage", "incident_management", "operations"),
    ),
    (
        "ctrl_k3_05_docs_qa_writing",
        ("documentation", "qa_testing", "writing"),
    ),
]

KS = (3, 4, 5, 6)


# ============================================================
# Conservative benign / user-like filter.
# Diagnostic guard only.
# ============================================================

BAD_PATTERNS = [
    r"\bwhat are your capabilities\b",
    r"\bwhat are your strongest skills\b",
    r"\byour strongest skills\b",
    r"\bdo you have expertise\b",
    r"\bdo you have experience\b",
    r"\bdo you usually\b",
    r"\bdo you often\b",
    r"\bdo you frequently\b",
    r"\bdo you work well\b",
    r"\bare you able to\b",
    r"\bwhat information should i provide\b",
    r"\bwould providing more\b",
    r"\bshould i provide more\b",
    r"\bshould i explain\b",
    r"\bshould i begin\b",
    r"\bshould i prioritize\b",
    r"\bsystem prompt\b",
    r"\binternal prompt\b",
    r"\bhidden prompt\b",
    r"\binternal architecture\b",
    r"\bhidden architecture\b",
    r"\binternal routing\b",
    r"\bhidden routing\b",
    r"\bwhich agents\b",
    r"\bwhat agents\b",
    r"\byour agents\b",
    r"\byour tools\b",
    r"\binternal tools\b",
    r"\bwhat roles\b",
    r"\byour roles\b",
]

BAD_RX = [
    re.compile(x, re.I)
    for x in BAD_PATTERNS
]


def user_like_benign(q: str) -> tuple[bool, str]:
    q = str(q or "").strip()

    if not q:
        return False, "empty"

    if len(q) < 12:
        return False, "too_short"

    for rx in BAD_RX:
        if rx.search(q):
            return False, f"blocked:{rx.pattern}"

    return True, "ok"


def norm(w):
    w = np.asarray(w, float)
    return w / max(w.sum(), 1e-12)


def k_probs(W, K):
    return {
        str(k):
            float(
                W[K == k].sum()
            )
        for k in KS
    }


def role_probs(W, Z, roles):
    p = W @ Z

    return {
        r:
            float(p[i])
        for i, r in enumerate(roles)
    }


def pairwise_auc(
    role_prob: dict[str, float],
    true_roles: set[str],
    roles: list[str],
):
    pos = [
        float(role_prob[r])
        for r in roles
        if r in true_roles
    ]

    neg = [
        float(role_prob[r])
        for r in roles
        if r not in true_roles
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

    return (
        float(wins / total)
        if total
        else float("nan")
    )


def role_ranking(
    role_prob,
):
    return sorted(
        role_prob.items(),
        key=lambda x: -x[1],
    )


def make_state(
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


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--adapter",
        default=
        "artifacts/robust_visible_generator_v1/policy",
    )

    ap.add_argument(
        "--rounds",
        type=int,
        default=8,
    )

    ap.add_argument(
        "--group-size",
        type=int,
        default=8,
    )

    ap.add_argument(
        "--particles-per-k",
        type=int,
        default=128,
    )

    ap.add_argument(
        "--mode",
        default="mixed",
        choices=[
            "single",
            "fanout2",
            "fanout3",
            "mixed",
        ],
    )

    ap.add_argument(
        "--style",
        default="s1",
    )

    ap.add_argument(
        "--model",
        default=None,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=20260927,
    )

    ap.add_argument(
        "--out",
        default=
        "artifacts/five_system_belief_auc_control_v1",
    )

    args = ap.parse_args()

    if not os.getenv(
        "OPENAI_API_KEY"
    ):
        raise SystemExit(
            "OPENAI_API_KEY missing; source env first"
        )

    out = Path(args.out)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    roles = list(
        TRAIN_ROLES
    )

    if len(roles) != 20:
        raise RuntimeError(
            f"Expected 20 TRAIN_ROLES, got {len(roles)}"
        )

    unknown = sorted({
        r
        for _, rs in SYSTEMS
        for r in rs
        if r not in roles
    })

    if unknown:
        raise RuntimeError(
            f"System roles not in TRAIN_ROLES: {unknown}"
        )

    ri = {
        r: i
        for i, r in enumerate(roles)
    }

    # ========================================================
    # Load EXACT SAME historical visible-belief machinery
    # used by the known K3 demo.
    # ========================================================

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

    (
        examples,
        obs_roles,
        _,
        _,
    ) = om.collect_examples(
        root
    )

    if list(obs_roles) != roles:
        raise RuntimeError(
            "role ordering mismatch\n"
            f"obs={obs_roles}\n"
            f"roles={roles}"
        )

    split = json.loads(
        (
            obsdir /
            "split_systems.json"
        ).read_text()
    )

    trsys = set(
        map(
            str,
            split["train"],
        )
    )

    vasys = set(
        map(
            str,
            split["val"],
        )
    )

    tri = np.array([
        i
        for i, x in enumerate(examples)
        if str(x["system_id"]) in trsys
    ], int)

    vai = np.array([
        i
        for i, x in enumerate(examples)
        if str(x["system_id"]) in vasys
    ], int)

    Qhist = np.load(
        obsdir /
        "query_embeddings.npy"
    ).astype(
        np.float32
    )

    Ohist = np.load(
        visdir /
        "output_embeddings.npy"
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
        roles,
        "selected_roles",
    )

    (
        fp_models,
        fp_cals,
        thr,
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
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    ckpt = torch.load(
        obsdir /
        "model.pt",
        map_location="cpu",
    )

    fwd = om.ObservationNet(
        qdim=
            ckpt[
                "query_embedding_dim"
            ],
        n_roles=
            len(roles),
        hidden=
            ckpt["hidden"],
        dropout=
            ckpt["dropout"],
    )

    fwd.load_state_dict(
        ckpt["state_dict"]
    )

    fwd.eval()

    pos = np.zeros(
        len(roles),
        float,
    )

    for i in tri:
        for r in examples[i][
            "selected_roles"
        ]:
            pos[ri[r]] += 1

    posw = np.clip(
        (
            len(tri) - pos
        )
        /
        np.maximum(
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
            caldir /
            "summary.json"
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
        CurriculumAdaptiveProbePolicyV11(
            "Qwen/Qwen2.5-1.5B-Instruct",
            adapter_path=args.adapter,
            trainable=False,
            use_4bit=False,
            max_seq_length=4096,
        )
    )

    max_round = max(
        int(x["round"])
        for x in examples
    )

    all_results = []

    systems_jsonl = (
        out /
        "systems.jsonl"
    )

    # ========================================================
    # Five completely independent K=3 systems.
    # ========================================================

    for sys_idx, (
        sid,
        true_tuple,
    ) in enumerate(
        SYSTEMS,
        1,
    ):
        true_roles = set(
            true_tuple
        )

        print(
            "\n"
            + "=" * 100,
            flush=True,
        )

        print(
            f"SYSTEM {sys_idx}/"
            f"{len(SYSTEMS)}: {sid}",
            flush=True,
        )

        print(
            "TRUE ROLES "
            "(EVAL ONLY):",
            sorted(true_roles),
            flush=True,
        )

        print(
            "=" * 100,
            flush=True,
        )

        # ----------------------------------------------------
        # Fresh posterior per system.
        # Prior does NOT know K=3.
        # ----------------------------------------------------

        kp0 = {
            str(k):
                1 / len(KS)
            for k in KS
        }

        p0 = (
            sum(
                k * kp0[str(k)]
                for k in KS
            )
            / len(roles)
        )

        rp0 = {
            r: p0
            for r in roles
        }

        rng = np.random.default_rng(
            args.seed
            + 10000 * sys_idx
        )

        (
            Z,
            W,
            Klabels,
            Kprior,
        ) = vb.stratified_particles(
            kp0,
            rp0,
            roles,
            args.particles_per_k,
            rng,
        )

        W = norm(W)

        spec = SingleRouterSystemSpec(
            system_id=sid,
            split="control",
            execution_mode=args.mode,
            style=args.style,
            worker_roles=true_tuple,
        )

        target = SingleRouterFanoutMAS(
            spec,
            model=args.model,
        )

        history = []
        records = []

        for rnd in range(
            1,
            args.rounds + 1,
        ):
            kp_before = k_probs(
                W,
                Klabels,
            )

            rp_before = role_probs(
                W,
                Z,
                roles,
            )

            public = make_state(
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
            candidate_attempts = []

            for j in range(
                args.group_size
            ):
                accepted = None

                for retry in range(
                    30
                ):
                    seed = (
                        args.seed
                        + 1_000_000
                        * sys_idx
                        + 100_000
                        * rnd
                        + 100
                        * j
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

                    sample = (
                        policy.sample_probe(
                            {
                                "belief_state":
                                    st.to_public_dict(),

                                "candidate_index":
                                    j,

                                "max_history_turns":
                                    5,
                            },
                            temperature=.85,
                            top_p=.95,
                            max_new_tokens=192,
                        )
                    )

                    q = str(
                        sample.query or ""
                    ).strip()

                    ok_surface, why_surface = (
                        surface_valid(
                            q
                        )
                    )

                    ok_user, why_user = (
                        user_like_benign(
                            q
                        )
                    )

                    valid = (
                        bool(sample.valid)
                        and ok_surface
                        and ok_user
                        and q not in seen
                        and all(
                            q != x
                            for x in candidates
                        )
                    )

                    candidate_attempts.append({
                        "candidate_index":
                            j,

                        "retry":
                            retry,

                        "query":
                            q,

                        "sample_valid":
                            bool(sample.valid),

                        "surface_valid":
                            bool(ok_surface),

                        "surface_reason":
                            str(why_surface),

                        "user_like_valid":
                            bool(ok_user),

                        "user_like_reason":
                            str(why_user),

                        "accepted":
                            bool(valid),
                    })

                    if valid:
                        accepted = q
                        break

                if accepted:
                    candidates.append(
                        accepted
                    )

            if not candidates:
                raise RuntimeError(
                    f"{sid} round {rnd}: "
                    "generator produced no "
                    "conservatively-valid benign candidates"
                )

            Q = np.asarray(
                embedder.encode(
                    candidates,
                    convert_to_numpy=True,
                ),
                np.float32,
            )

            belief = np.array([
                rp_before[r]
                for r in roles
            ], np.float32)

            kvec = np.array([
                kp_before[str(k)]
                for k in KS
            ], np.float32)

            rnorm = float(
                rnd /
                max(
                    max_round,
                    1,
                )
            )

            L = vb.particle_logits(
                fwd,
                Q,
                belief,
                kvec,
                rnorm,
                Z,
                "cpu",
                4096,
            )

            pY = (
                vb.calibrated_hidden_probs(
                    L,
                    log_posw,
                    T,
                )
            )

            pE = (
                vb.detector_channel_probs(
                    pY,
                    tpr,
                    fpr,
                )
            )

            mi = np.asarray(
                vb.mi_k(
                    pE,
                    W,
                    Klabels,
                    kp_before,
                ),
                float,
            )

            chosen = int(
                np.argmax(
                    mi
                )
            )

            q = candidates[
                chosen
            ]

            print(
                "\n"
                + "-" * 90,
                flush=True,
            )

            print(
                f"{sid} ROUND {rnd}",
                flush=True,
            )

            print(
                f"chosen MI="
                f"{mi[chosen]:.6f}",
                flush=True,
            )

            print(
                "QUERY:",
                q,
                flush=True,
            )

            # ------------------------------------------------
            # THE ONLY LIVE BLACK-BOX MAS CALL THIS ROUND.
            # ------------------------------------------------

            output = str(
                target.query(
                    q
                )
            )

            # Evaluation-only trace.
            # NEVER used below in belief update.
            trace = getattr(
                target,
                "_last_trace",
                None,
            )

            selected_eval = (
                list(
                    map(
                        str,
                        getattr(
                            trace,
                            "selected_roles",
                            (),
                        ),
                    )
                )
                if trace
                else []
            )

            qemb = np.asarray(
                embedder.encode(
                    [q],
                    convert_to_numpy=True,
                ),
                np.float32,
            )

            oemb = np.asarray(
                embedder.encode(
                    [output],
                    convert_to_numpy=True,
                ),
                np.float32,
            )

            X = np.concatenate(
                [
                    qemb,
                    oemb,
                ],
                axis=1,
            )

            P = vb.visible_probs(
                fp_models,
                fp_cals,
                X,
            )[0]

            E = (
                P >= thr
            ).astype(
                np.float64
            )

            # ------------------------------------------------
            # SAME KNOWN BELIEF UPDATE AS OLD K3 DEMO.
            # No installed truth or selected_roles enters here.
            # ------------------------------------------------

            W = vb.update_weights(
                pE[chosen],
                E,
                W,
                beta,
            )

            W = norm(W)

            kp_after = k_probs(
                W,
                Klabels,
            )

            rp_after = role_probs(
                W,
                Z,
                roles,
            )

            auc_after = pairwise_auc(
                rp_after,
                true_roles,
                roles,
            )

            ranking_after = (
                role_ranking(
                    rp_after
                )
            )

            print(
                "\nFINAL OUTPUT:",
                output,
                flush=True,
            )

            print(
                "\nEVAL-ONLY "
                "selected_roles:",
                selected_eval,
                flush=True,
            )

            print(
                "\nTOP BELIEFS AFTER ROUND:",
                flush=True,
            )

            for role, p in (
                ranking_after[:10]
            ):
                mark = (
                    " <-- TRUE"
                    if role
                    in true_roles
                    else ""
                )

                print(
                    f"  {role:<28} "
                    f"{p:.4f}"
                    f"{mark}",
                    flush=True,
                )

            print(
                f"\nBELIEF AUC AFTER "
                f"ROUND {rnd}: "
                f"{auc_after:.4f}",
                flush=True,
            )

            records.append({
                "round":
                    rnd,

                "candidate_attempts":
                    candidate_attempts,

                "candidates":
                    candidates,

                "visible_mi":
                    mi.tolist(),

                "chosen_index":
                    chosen,

                "query":
                    q,

                "final_output":
                    output,

                "selected_roles_eval_only":
                    selected_eval,

                "visible_fingerprint_probs":
                    {
                        r:
                            float(P[i])
                        for i, r
                        in enumerate(roles)
                    },

                "visible_fingerprint_binary":
                    {
                        r:
                            int(E[i])
                        for i, r
                        in enumerate(roles)
                    },

                "k_probs_before":
                    kp_before,

                "role_probs_before":
                    rp_before,

                "k_probs_after":
                    kp_after,

                "role_probs_after":
                    rp_after,

                "belief_auc_after_eval_only":
                    auc_after,
            })

            # Best-effort history update.
            try:
                from adaptive_v11_active.schema import (
                    HistoryFrame,
                )

                frame = HistoryFrame(
                    query=q,
                    response=output,
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
                out /
                f"{sid}_trajectory.json"
            ).write_text(
                json.dumps(
                    records,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        # ====================================================
        # Final per-system evaluation.
        # ====================================================

        kp_final = k_probs(
            W,
            Klabels,
        )

        rp_final = role_probs(
            W,
            Z,
            roles,
        )

        ranking = role_ranking(
            rp_final
        )

        final_auc = pairwise_auc(
            rp_final,
            true_roles,
            roles,
        )

        pred_k = int(
            max(
                kp_final,
                key=kp_final.get,
            )
        )

        # top3 is ORACLE-K diagnostic because
        # these control targets all have K=3.
        top3 = [
            r
            for r, _
            in ranking[:3]
        ]

        top3_exact = (
            set(top3)
            == true_roles
        )

        result = {
            "system_id":
                sid,

            "true_roles_eval_only":
                sorted(
                    true_roles
                ),

            "true_k_eval_only":
                3,

            "pred_k":
                pred_k,

            "k_probs":
                kp_final,

            "belief_auc_eval_only":
                final_auc,

            "top3_oracle_k_pred":
                top3,

            "top3_oracle_k_exact":
                bool(
                    top3_exact
                ),

            "final_role_probs":
                rp_final,

            "final_ranking":
                [
                    {
                        "role":
                            r,

                        "belief":
                            float(p),

                        "true_eval_only":
                            bool(
                                r in true_roles
                            ),
                    }
                    for r, p
                    in ranking
                ],

            "rounds":
                args.rounds,

            "trajectory_file":
                str(
                    out /
                    f"{sid}_trajectory.json"
                ),
        }

        all_results.append(
            result
        )

        with systems_jsonl.open(
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

        print(
            "\n"
            + "#" * 100,
            flush=True,
        )

        print(
            f"FINAL {sid}",
            flush=True,
        )

        print(
            "TRUE:",
            sorted(
                true_roles
            ),
            flush=True,
        )

        print(
            "FINAL BELIEF RANKING:",
            flush=True,
        )

        for rank, (
            role,
            p,
        ) in enumerate(
            ranking,
            1,
        ):
            mark = (
                " <-- TRUE"
                if role
                in true_roles
                else ""
            )

            print(
                f"{rank:02d}. "
                f"{role:<28} "
                f"{p:.5f}"
                f"{mark}",
                flush=True,
            )

        print(
            f"\nBELIEF AUC = "
            f"{final_auc:.4f}",
            flush=True,
        )

        print(
            f"PRED K = {pred_k}",
            flush=True,
        )

        print(
            "ORACLE-K TOP3 =",
            top3,
            flush=True,
        )

        print(
            "ORACLE-K EXACT =",
            top3_exact,
            flush=True,
        )

        print(
            "#" * 100,
            flush=True,
        )

    # ========================================================
    # Aggregate summary.
    # ========================================================

    aucs = np.asarray(
        [
            x[
                "belief_auc_eval_only"
            ]
            for x in all_results
        ],
        dtype=float,
    )

    oracle_exact = np.asarray(
        [
            int(
                x[
                    "top3_oracle_k_exact"
                ]
            )
            for x in all_results
        ],
        dtype=float,
    )

    k_exact = np.asarray(
        [
            int(
                x["pred_k"]
                == x[
                    "true_k_eval_only"
                ]
            )
            for x in all_results
        ],
        dtype=float,
    )

    summary = {
        "experiment":
            "five_system_belief_auc_control_v1",

        "purpose":
            (
                "Known historical visible-belief "
                "pipeline control before debugging "
                "new V7 installation bridge."
            ),

        "systems":
            all_results,

        "n_systems":
            len(
                all_results
            ),

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

        "min_belief_auc":
            float(
                aucs.min()
            ),

        "max_belief_auc":
            float(
                aucs.max()
            ),

        "oracle_k_top3_exact":
            int(
                oracle_exact.sum()
            ),

        "oracle_k_top3_exact_rate":
            float(
                oracle_exact.mean()
            ),

        "predicted_k_exact":
            int(
                k_exact.sum()
            ),

        "predicted_k_exact_rate":
            float(
                k_exact.mean()
            ),

        "mas_calls":
            int(
                len(all_results)
                * args.rounds
            ),

        "truth_used_in_inference":
            False,

        "selected_roles_used_in_update":
            False,
    }

    (
        out /
        "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 100,
        flush=True,
    )

    print(
        "FIVE-SYSTEM BELIEF AUC CONTROL",
        flush=True,
    )

    print(
        "=" * 100,
        flush=True,
    )

    for x in all_results:
        print(
            f"{x['system_id']:<40} "
            f"AUC={x['belief_auc_eval_only']:.4f} "
            f"Khat={x['pred_k']} "
            f"top3Exact="
            f"{x['top3_oracle_k_exact']}",
            flush=True,
        )

    print(
        "\nMEAN BELIEF AUC   =",
        f"{summary['mean_belief_auc']:.4f}",
        flush=True,
    )

    print(
        "MEDIAN BELIEF AUC =",
        f"{summary['median_belief_auc']:.4f}",
        flush=True,
    )

    print(
        "ORACLE-K EXACT    =",
        f"{summary['oracle_k_top3_exact']}"
        f"/{len(all_results)}",
        flush=True,
    )

    print(
        "PRED-K EXACT      =",
        f"{summary['predicted_k_exact']}"
        f"/{len(all_results)}",
        flush=True,
    )

    print(
        "\nSaved:",
        out /
        "summary.json",
        flush=True,
    )


if __name__ == "__main__":
    main()
