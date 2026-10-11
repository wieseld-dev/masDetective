#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import random
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import targets.single_router_zoo_v11

from targets.single_router_zoo_v11 import validate_v11_design
validate_v11_design()

from targets.single_router_fanout_mas import (
    SingleRouterFanoutMAS,
    SingleRouterSystemSpec,
)

from adaptive_v11_active.reward import surface_valid
from adaptive_v11_active.schema import PublicBeliefState
from adaptive_v11_curriculum.policy import (
    CurriculumAdaptiveProbePolicyV11,
)

from v11.role_universe import TRAIN_ROLES

import analysis.observation_model_v1 as om
import analysis.visible_adaptive_bridge_v1 as vb
import analysis.expected_information_gain_selector_v1 as eig
import analysis.train_v13_6b_activation_judge as sensor_tr
import analysis.build_visible_realized_gain_v3 as sensor_v3


# ============================================================
# DELIBERATELY EASY, ONE-OFF END-TO-END DEMO
# ============================================================

TRUE_ROLES = (
    "accessibility_review",
    "risk_assessment",
    "translation",
)

FAKE_ROLES = (
    "math",
    "qa_testing",
    "customer_support_triage",
)

CANDIDATE_ROLES = TRUE_ROLES + FAKE_ROLES

K_TRUE = 3
ROUNDS = 8
GROUP_SIZE = 8
SEED = 20260926
BETA = 0.10

GENERATOR = Path(
    "artifacts/v6_matched_sft_dpo/policy"
)

SENSOR = Path(
    "artifacts/v13_8b_balanced_repair_2048_run1/best_adapter"
)

CALIBRATION = Path(
    "artifacts/visible_realized_gain_v3/channel_calibration.json"
)

OUT = Path(
    "artifacts/cheat_full_adaptive_k3/result.json"
)


def norm(w):
    w = np.asarray(w, dtype=np.float64)
    s = float(w.sum())
    if s <= 0:
        raise RuntimeError("posterior collapsed")
    return w / s


def exact_k3_hypotheses(all_roles):
    """
    Exactly 20 hypotheses:
      choose 3 installed roles from the six candidate roles.

    Z remains 20-dimensional because the observation model was
    trained on the original 20-role universe.
    """
    ri = {
        r: i
        for i, r in enumerate(all_roles)
    }

    rows = []
    names = []

    for combo in combinations(
        CANDIDATE_ROLES,
        K_TRUE,
    ):
        z = np.zeros(
            len(all_roles),
            dtype=np.float32,
        )

        for r in combo:
            z[ri[r]] = 1.0

        rows.append(z)
        names.append(tuple(combo))

    Z = np.stack(rows)

    assert Z.shape == (20, 20)
    assert np.all(Z.sum(axis=1) == 3)

    W = np.ones(
        len(Z),
        dtype=np.float64,
    )
    W /= W.sum()

    Klabels = np.full(
        len(Z),
        3,
        dtype=int,
    )

    return Z, W, Klabels, names


def role_probs(W, Z, roles):
    p = W @ Z

    return {
        r: float(p[i])
        for i, r in enumerate(roles)
    }


PUBLIC_EPS = 1e-4


def public_k_probs():
    # Generator-schema compatibility only.
    # Internally K remains EXACTLY 3.
    e = PUBLIC_EPS
    return {
        "3": 1.0 - 3.0 * e,
        "4": e,
        "5": e,
        "6": e,
    }


def public_role_probs(rp):
    # The true posterior is NOT modified.
    # Only the generator-visible representation is clipped
    # because PublicBeliefState requires strict (0,1).
    return {
        r: float(np.clip(
            float(p),
            PUBLIC_EPS,
            1.0 - PUBLIC_EPS,
        ))
        for r, p in rp.items()
    }


def make_state(
    sid,
    round_index,
    rp,
):
    return {
        "session_id": sid,
        "split": "dev",
        "round_index": round_index,
        "k_probs": public_k_probs(),
        "role_probs": public_role_probs(rp),

        # Belief itself is adaptive. We deliberately avoid
        # brittle cross-version history serialization here.
        # Repeated queries are still rejected globally.
        "history": [],
    }


def map_decision(W, hypothesis_names):
    j = int(np.argmax(W))

    return {
        "roles": list(hypothesis_names[j]),
        "mass": float(W[j]),
        "index": j,
    }


def posterior_update(
    W,
    p_activation_candidate_roles,
    sensor_scores_candidate_roles,
    calibration,
):
    """
    Exact generalized-Bayes update over the 20 possible
    installed sets.

      p(score | Z,q)
        proportional to
      (1-a) + a * LR(score)

    Only the SIX declared candidate roles contribute to the
    posterior. Sensor scoring itself is still performed with
    all 20 roles in one batch.
    """

    log_lr = sensor_v3.sensor_log_lr(
        sensor_scores_candidate_roles,
        list(CANDIDATE_ROLES),
        calibration,
    )

    lr = np.exp(log_lr)[None, :]

    a = np.clip(
        np.asarray(
            p_activation_candidate_roles,
            dtype=np.float64,
        ),
        1e-5,
        1.0 - 1e-5,
    )

    likelihood_ratio = (
        (1.0 - a)
        + a * lr
    )

    loglik = (
        BETA
        * np.log(
            np.clip(
                likelihood_ratio,
                1e-12,
                None,
            )
        ).sum(axis=1)
    )

    logw = (
        np.log(
            np.clip(W, 1e-300, None)
        )
        + loglik
    )

    logw -= np.max(logw)

    return norm(
        np.exp(logw)
    )


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY missing"
        )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    roles = list(TRAIN_ROLES)

    assert len(roles) == 20
    assert set(CANDIDATE_ROLES).issubset(
        roles
    )
    assert set(sensor_tr.ROLE_DESCRIPTIONS) == set(
        roles
    )

    ri = {
        r: i
        for i, r in enumerate(roles)
    }

    candidate_idx = [
        ri[r]
        for r in CANDIDATE_ROLES
    ]

    # ========================================================
    # Exact posterior state: 20 possible K=3 subsets.
    # ========================================================

    Z, W, Klabels, hypothesis_names = (
        exact_k3_hypotheses(roles)
    )

    print("=" * 100)
    print("EASY ADAPTIVE K3 — FULL PIPELINE")
    print("=" * 100)

    print(
        "Candidate universe:",
        CANDIDATE_ROLES,
    )
    print(
        "Known installation count:",
        K_TRUE,
    )
    print(
        "Hypotheses:",
        len(W),
    )
    print(
        "Initial prior per hypothesis:",
        float(W[0]),
    )

    # ========================================================
    # Load activation prediction / selector model.
    # ========================================================

    hist_root = Path(
        "artifacts/adaptive_v11_curriculum_v2"
    )
    obsdir = Path(
        "artifacts/observation_model_v1"
    )
    caldir = Path(
        "artifacts/calibrated_bayes_diag_v1"
    )

    examples, obs_roles, _, _ = (
        om.collect_examples(hist_root)
    )

    if list(obs_roles) != roles:
        raise RuntimeError(
            f"role order mismatch: "
            f"{obs_roles} vs {roles}"
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

    train_idx = np.array([
        i
        for i, x in enumerate(examples)
        if str(x["system_id"])
        in train_systems
    ])

    ckpt = torch.load(
        obsdir / "model.pt",
        map_location="cpu",
    )

    fwd = om.ObservationNet(
        qdim=ckpt[
            "query_embedding_dim"
        ],
        n_roles=len(roles),
        hidden=ckpt["hidden"],
        dropout=ckpt["dropout"],
    )

    fwd.load_state_dict(
        ckpt["state_dict"]
    )
    fwd.eval()

    pos = np.zeros(
        len(roles),
        dtype=np.float64,
    )

    for i in train_idx:
        for r in examples[i][
            "selected_roles"
        ]:
            pos[ri[r]] += 1

    posw = np.clip(
        (
            len(train_idx) - pos
        )
        / np.maximum(
            pos,
            1.0,
        ),
        1.0,
        20.0,
    )

    log_posw = np.log(posw)

    old_cal = json.loads(
        (
            caldir
            / "summary.json"
        ).read_text()
    )

    T = float(
        old_cal[
            "true_Z_probability_calibration"
        ][
            "corrected_plus_temperature"
        ][
            "best_T"
        ]
    )

    embedder = SentenceTransformer(
        "sentence-transformers/"
        "all-MiniLM-L6-v2"
    )

    # ========================================================
    # Generator: V6-DPO.
    # Generates its OWN benign queries from current belief.
    # ========================================================

    policy = CurriculumAdaptiveProbePolicyV11(
        "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path=str(GENERATOR),
        trainable=False,
        use_4bit=False,
        max_seq_length=4096,
    )

    # ========================================================
    # Hidden MAS: only the three actual workers.
    # ========================================================

    sid = "easy_cheat_adaptive_k3"

    spec = SingleRouterSystemSpec(
        system_id=sid,
        split="demo",
        execution_mode="single",
        style="s1",
        worker_roles=TRUE_ROLES,
    )

    target = SingleRouterFanoutMAS(
        spec,
        model=os.getenv(
            "MAS_MODEL",
            "gpt-4o-mini",
        ),
    )

    # ========================================================
    # Current visible activation sensor.
    # ========================================================

    calibration = json.loads(
        CALIBRATION.read_text()
    )

    sensor_tok, sensor_model, sensor_dtype = (
        sensor_v3.load_sensor(
            SENSOR
        )
    )

    seen_queries = set()
    records = []

    # ========================================================
    # Adaptive loop.
    # ========================================================

    for rnd in range(
        1,
        ROUNDS + 1,
    ):
        rp = role_probs(
            W,
            Z,
            roles,
        )

        state = (
            PublicBeliefState
            .from_public_dict(
                make_state(
                    sid,
                    rnd - 1,
                    rp,
                )
            )
        )

        candidates = []

        for j in range(GROUP_SIZE):
            accepted = None

            for retry in range(20):
                seed = (
                    SEED
                    + 100000 * rnd
                    + 100 * j
                    + retry
                )

                random.seed(seed)
                torch.manual_seed(seed)

                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(
                        seed
                    )

                sample = policy.sample_probe(
                    {
                        "belief_state":
                            state.to_public_dict(),
                        "candidate_index": j,
                        "max_history_turns": 5,
                    },
                    temperature=0.85,
                    top_p=0.95,
                    max_new_tokens=192,
                )

                q = str(
                    sample.query or ""
                ).strip()

                ok, why = surface_valid(q)

                valid = (
                    bool(sample.valid)
                    and ok
                    and q
                    and q not in seen_queries
                    and q not in candidates
                )

                if valid:
                    accepted = q
                    break

            if accepted:
                candidates.append(
                    accepted
                )

        if not candidates:
            raise RuntimeError(
                "Generator produced no valid "
                "benign candidates"
            )

        # ----------------------------------------------------
        # Selector.
        #
        # It predicts activation behavior for EVERY candidate
        # under EVERY one of the 20 possible hidden K3 sets.
        #
        # Then chooses maximum MI_Z: information about the
        # installed composition, not ground truth.
        # ----------------------------------------------------

        Q = np.asarray(
            embedder.encode(
                candidates,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )

        belief = np.asarray(
            [
                rp[r]
                for r in roles
            ],
            dtype=np.float32,
        )

        kvec = np.asarray(
            [
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

        round_norm = float(
            rnd / ROUNDS
        )

        logits = vb.particle_logits(
            fwd,
            Q,
            belief,
            kvec,
            round_norm,
            Z,
            "cpu",
            4096,
        )

        p_activation = (
            vb.calibrated_hidden_probs(
                logits,
                log_posw,
                T,
            )
        )

        (
            mi_k,
            mi_identity,
            mi_z,
            predicted_exposure,
        ) = eig.mi_scores(
            p_activation,
            W,
            Klabels,
            {3: 1.0},
        )

        chosen = int(
            np.argmax(mi_z)
        )

        query = candidates[chosen]
        seen_queries.add(query)

        print("\n" + "=" * 100)
        print(
            f"ROUND {rnd}/{ROUNDS}"
        )

        print(
            "\nGENERATOR CANDIDATES "
            "+ SELECTOR DECISION:"
        )

        for j, q in enumerate(candidates):
            mark = (
                "  <-- CHOSEN"
                if j == chosen
                else ""
            )

            print(
                f"[{j}] "
                f"MI_Z={mi_z[j]:.6f} "
                f"MI_ID={mi_identity[j]:.6f} "
                f"EXP={predicted_exposure[j]:.3f}"
                f"{mark}"
            )
            print(
                "    "
                + q.replace(
                    "\n",
                    " ",
                )
            )

        print("\nSENDING TO MAS:")
        print(query)

        # ----------------------------------------------------
        # Real black-box interaction.
        # Router chooses worker itself.
        # ----------------------------------------------------

        output = str(
            target.query(query)
        )

        # Evaluation ONLY. Absolutely never used below in
        # selector/posterior computation.
        trace = getattr(
            target,
            "_last_trace",
            None,
        )

        actual_eval = (
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

        # ----------------------------------------------------
        # Visible sensor.
        #
        # IMPORTANT: still scores ALL 20 roles in the same
        # response batch to preserve canonical protocol.
        # ----------------------------------------------------

        sensor_scores_all = (
            sensor_v3.sensor_score_row(
                model=sensor_model,
                tok=sensor_tok,
                dtype=sensor_dtype,
                roles=roles,
                role_desc=dict(
                    sensor_tr.ROLE_DESCRIPTIONS
                ),
                query=query,
                answer=output,
                max_length=4096,
            )
        )

        sensor_loglr_all = (
            sensor_v3.sensor_log_lr(
                sensor_scores_all,
                roles,
                calibration,
            )
        )

        scores6 = np.asarray(
            [
                sensor_scores_all[i]
                for i in candidate_idx
            ],
            dtype=np.float64,
        )

        loglr6 = {
            r: float(
                sensor_loglr_all[
                    ri[r]
                ]
            )
            for r in CANDIDATE_ROLES
        }

        # ----------------------------------------------------
        # Posterior update.
        #
        # It sees only:
        #   chosen query
        #   final output -> sensor scores
        #   predicted P(activation | hypothetical Z,q)
        #
        # It NEVER sees actual_eval.
        # ----------------------------------------------------

        p_activation6 = (
            p_activation[
                chosen
            ][:, candidate_idx]
        )

        W = posterior_update(
            W,
            p_activation6,
            scores6,
            calibration,
        )

        rp_after = role_probs(
            W,
            Z,
            roles,
        )

        decision = map_decision(
            W,
            hypothesis_names,
        )

        print("\nVISIBLE FINAL OUTPUT:")
        print(output)

        print(
            "\nSENSOR evidence among "
            "the six candidates:"
        )

        for r in sorted(
            CANDIDATE_ROLES,
            key=lambda x:
                loglr6[x],
            reverse=True,
        ):
            print(
                f"  {r:<28} "
                f"logLR={loglr6[r]:+.4f}"
            )

        print(
            "\nPOSTERIOR ROLE MARGINALS:"
        )

        for r in sorted(
            CANDIDATE_ROLES,
            key=lambda x:
                rp_after[x],
            reverse=True,
        ):
            print(
                f"  {r:<28} "
                f"P(installed)="
                f"{rp_after[r]:.4f}"
            )

        print(
            "\nCURRENT MAP DECISION:",
            decision["roles"],
        )
        print(
            "MAP composition mass:",
            f"{decision['mass']:.4f}",
        )

        print(
            "\nROUTER TRACE [EVAL ONLY]:",
            actual_eval,
        )

        records.append({
            "round": rnd,
            "candidates": candidates,
            "selector": {
                "mi_z":
                    mi_z.tolist(),
                "mi_identity":
                    mi_identity.tolist(),
                "predicted_exposure":
                    predicted_exposure.tolist(),
                "chosen_index":
                    chosen,
                "chosen_query":
                    query,
            },
            "final_output":
                output,
            "sensor_loglr_six":
                loglr6,
            "posterior_role_probs": {
                r: rp_after[r]
                for r in CANDIDATE_ROLES
            },
            "map_decision":
                decision,
            "actual_route_eval_only":
                actual_eval,
        })

    # ========================================================
    # Final decision FIRST; truth comparison only afterwards.
    # ========================================================

    final_decision = map_decision(
        W,
        hypothesis_names,
    )

    final_rp = role_probs(
        W,
        Z,
        roles,
    )

    predicted_set = set(
        final_decision["roles"]
    )

    true_set = set(TRUE_ROLES)

    exact = (
        predicted_set
        == true_set
    )

    print("\n" + "#" * 100)
    print("FINAL END-TO-END DECISION")
    print("#" * 100)

    print(
        "PREDICTED INSTALLED:",
        final_decision["roles"],
    )
    print(
        "POSTERIOR MASS OF MAP SET:",
        f"{final_decision['mass']:.6f}",
    )

    print("\nFINAL MARGINALS:")

    for r in sorted(
        CANDIDATE_ROLES,
        key=lambda x:
            final_rp[x],
        reverse=True,
    ):
        print(
            f"  {r:<28} "
            f"{final_rp[r]:.6f}"
        )

    print(
        "\nTRUE INSTALLED "
        "[EVAL ONLY, revealed now]:",
        list(TRUE_ROLES),
    )

    print(
        "EXACT SET RECOVERY:",
        exact,
    )

    result = {
        "protocol":
            "cheat_full_adaptive_k3_v1",
        "deliberately_easy": True,
        "known_k": 3,
        "candidate_roles":
            list(CANDIDATE_ROLES),
        "generator":
            str(GENERATOR),
        "sensor":
            str(SENSOR),
        "selector":
            "adaptive generated candidates + MI_Z",
        "posterior":
            "exact 20 K=3 hypotheses + visible sensor generalized Bayes",
        "rounds":
            ROUNDS,
        "api_calls_expected":
            2 * ROUNDS,
        "final_map":
            final_decision,
        "final_role_probs": {
            r: final_rp[r]
            for r in CANDIDATE_ROLES
        },
        "true_roles_eval_only":
            list(TRUE_ROLES),
        "exact_set_recovery":
            bool(exact),
        "records":
            records,
    }

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "\nSAVED:",
        OUT,
    )


if __name__ == "__main__":
    main()
