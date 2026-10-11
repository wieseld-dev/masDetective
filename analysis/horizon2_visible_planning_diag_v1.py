#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path("analysis").resolve()))

import observation_model_v1 as om
import visible_adaptive_bridge_v1 as vb
import expected_information_gain_selector_v1 as eig


def stable_seed(*parts):
    s = "||".join(map(str, parts))
    return int.from_bytes(
        hashlib.sha256(s.encode("utf-8")).digest()[:4],
        "little",
    )


def normw(w):
    w = np.asarray(w, dtype=np.float64)
    w = np.maximum(w, 0.0)
    z = w.sum()
    if z <= 0:
        raise RuntimeError("zero particle weight")
    return w / z


def kprior_from_weights(W, Klabels):
    W = normw(W)
    out = {}
    for k in sorted(set(map(int, Klabels.tolist()))):
        out[int(k)] = float(W[Klabels == k].sum())
    z = sum(out.values())
    return {k: v / z for k, v in out.items()}


def belief_from_weights(W, Z):
    W = normw(W)
    return np.einsum("m,mr->r", W, Z).astype(np.float32)


def kvec_from_weights(W, Klabels):
    kp = kprior_from_weights(W, Klabels)
    return np.asarray(
        [kp.get(k, 0.0) for k in (3, 4, 5, 6)],
        dtype=np.float32,
    )


def score_from_P(P, W, Klabels, Kprior, objective, alpha):
    mi_k, mi_id, mi_z, exposure = eig.mi_scores(
        P, W, Klabels, Kprior
    )

    if objective == "mi_z":
        score = mi_z
    elif objective == "hybrid":
        score = alpha * mi_k + (1.0 - alpha) * mi_id
    elif objective == "mi_k":
        score = mi_k
    elif objective == "mi_identity":
        score = mi_id
    else:
        raise ValueError(objective)

    return score, {
        "mi_k": mi_k,
        "mi_identity": mi_id,
        "mi_z": mi_z,
        "exposure": exposure,
    }


def posterior_metrics(W, Z, Klabels, true_z, true_k):
    W = normw(W)

    rp = np.einsum("m,mr->r", W, Z)
    role_brier = float(np.mean((rp - true_z) ** 2))

    kvals = sorted(set(map(int, Klabels.tolist())))
    pk = np.asarray(
        [W[Klabels == k].sum() for k in kvals],
        dtype=np.float64,
    )
    pk /= pk.sum()

    ktgt = np.asarray(
        [1.0 if k == true_k else 0.0 for k in kvals],
        dtype=np.float64,
    )

    k_brier = float(np.sum((pk - ktgt) ** 2))

    p_true_k = (
        float(pk[kvals.index(true_k)])
        if true_k in kvals
        else 0.0
    )

    true_mask = np.all(
        Z.astype(np.int8) == true_z.astype(np.int8)[None, :],
        axis=1,
    )

    p_true_z = float(W[true_mask].sum()) if np.any(true_mask) else 0.0

    return {
        "role_brier": role_brier,
        "k_brier": k_brier,
        "p_true_k": p_true_k,
        "p_true_z": p_true_z,
    }


@torch.no_grad()
def visible_channel_for_pool(
    model,
    Q,
    idxs,
    belief,
    kvec,
    rnorm,
    Z,
    W,
    Klabels,
    log_posw,
    T,
    tpr,
    fpr,
    objective,
    alpha,
    batch_size,
):
    qg = np.stack([Q[i] for i in idxs]).astype(np.float32)

    L = vb.particle_logits(
        model,
        qg,
        belief,
        kvec,
        rnorm,
        Z,
        "cpu",
        batch_size,
    )

    pY = vb.calibrated_hidden_probs(
        L,
        log_posw,
        T,
    )

    pE = vb.detector_channel_probs(
        pY,
        tpr,
        fpr,
    )

    kp = kprior_from_weights(W, Klabels)

    scores, parts = score_from_P(
        pE,
        W,
        Klabels,
        kp,
        objective,
        alpha,
    )

    return pE, scores, parts


def actual_update(
    W,
    Pe,
    evidence,
    beta,
):
    return vb.update_weights(
        Pe,
        evidence,
        W,
        beta,
    )


def mean(xs):
    return float(np.mean(xs)) if xs else None


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        default="artifacts/adaptive_v11_curriculum_v2",
    )
    ap.add_argument(
        "--obs-dir",
        default="artifacts/observation_model_v1",
    )
    ap.add_argument(
        "--vis-dir",
        default="artifacts/visible_output_fingerprint_v1",
    )
    ap.add_argument(
        "--cal-dir",
        default="artifacts/calibrated_bayes_diag_v1",
    )
    ap.add_argument(
        "--out",
        default="artifacts/horizon2_visible_planning_diag_v1",
    )

    ap.add_argument("--particles-per-k", type=int, default=8)
    ap.add_argument("--branches", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=4096)

    ap.add_argument(
        "--objective",
        choices=["mi_z", "hybrid", "mi_k", "mi_identity"],
        default="mi_z",
    )
    ap.add_argument("--hybrid-alpha", type=float, default=0.5)

    ap.add_argument("--round-only", type=int, default=1)
    ap.add_argument("--max-states", type=int, default=20)
    ap.add_argument("--seed", type=int, default=731)

    args = ap.parse_args()

    root = Path(args.root)
    obsdir = Path(args.obs_dir)
    visdir = Path(args.vis_dir)
    caldir = Path(args.cal_dir)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 92)
    print("HORIZON-2 VISIBLE PLANNING DIAGNOSTIC V1")
    print("=" * 92)

    print("[1/6] loading examples / split")

    examples, roles, _, _ = om.collect_examples(root)

    split = json.loads(
        (obsdir / "split_systems.json").read_text()
    )

    train_systems = set(map(str, split["train"]))
    val_systems = set(map(str, split["val"]))
    test_systems = set(map(str, split["test"]))

    tri = np.asarray(
        [
            i for i, x in enumerate(examples)
            if str(x["system_id"]) in train_systems
        ],
        dtype=int,
    )

    vai = np.asarray(
        [
            i for i, x in enumerate(examples)
            if str(x["system_id"]) in val_systems
        ],
        dtype=int,
    )

    tei = np.asarray(
        [
            i for i, x in enumerate(examples)
            if str(x["system_id"]) in test_systems
        ],
        dtype=int,
    )

    Q = np.load(
        obsdir / "query_embeddings.npy"
    ).astype(np.float32)

    O = np.load(
        visdir / "output_embeddings.npy"
    ).astype(np.float32)

    if len(Q) != len(examples) or len(O) != len(examples):
        raise RuntimeError(
            f"embedding/example mismatch: "
            f"Q={len(Q)} O={len(O)} examples={len(examples)}"
        )

    X = np.concatenate([Q, O], axis=1)

    print("[2/6] fitting frozen visible detector channel")

    Y = vb.labels(
        examples,
        roles,
        "selected_roles",
    )

    fp_models, fp_cals, thresholds, tpr, fpr, _ = (
        vb.fit_visible_detector(
            X,
            Y,
            tri,
            vai,
        )
    )

    Pvisible = vb.visible_probs(
        fp_models,
        fp_cals,
        X,
    )

    print(
        "visible channel mean TPR/FPR:",
        float(np.mean(tpr)),
        float(np.mean(fpr)),
    )

    print("[3/6] loading frozen forward observation model")

    ckpt = torch.load(
        obsdir / "model.pt",
        map_location="cpu",
    )

    if ckpt["role_order"] != roles:
        raise RuntimeError("role order mismatch")

    model = om.ObservationNet(
        qdim=ckpt["query_embedding_dim"],
        n_roles=len(roles),
        hidden=ckpt["hidden"],
        dropout=ckpt["dropout"],
    )

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ri = {r: i for i, r in enumerate(roles)}

    pos = np.zeros(len(roles), dtype=np.float64)

    for i in tri:
        for r in examples[i]["selected_roles"]:
            pos[ri[r]] += 1

    posw = np.clip(
        (len(tri) - pos) / np.maximum(pos, 1.0),
        1.0,
        20.0,
    )

    log_posw = np.log(posw)

    cs = json.loads(
        (caldir / "summary.json").read_text()
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

    print("T:", T)
    print("beta:", beta)

    print("[4/6] grouping clean candidate states")

    traj = defaultdict(list)

    for i in tei:
        x = examples[i]

        if int(x["round"]) != args.round_only:
            continue

        key = (
            str(x.get("parent", "")),
            str(x["system_id"]),
            int(x["round"]),
        )

        traj[key].append((i, x))

    groups = []

    for key, rows in sorted(traj.items()):
        rows = sorted(
            rows,
            key=lambda ix: int(ix[1]["candidate_index"]),
        )

        if len(rows) >= 3:
            groups.append((key, rows))

    if args.max_states > 0:
        groups = groups[:args.max_states]

    print(
        f"states={len(groups)} "
        f"round={args.round_only} "
        f"particles_per_k={args.particles_per_k} "
        f"branches={args.branches}"
    )

    if not groups:
        raise RuntimeError("no usable states")

    max_round = max(
        int(x["round"]) for x in examples
    )

    per_state = []

    print("[5/6] horizon-2 planning")

    for gi, (key, rows) in enumerate(groups, 1):
        parent, sid, rnd = key

        idxs = [i for i, _ in rows]
        x0 = rows[0][1]

        belief0 = np.asarray(
            [x0["role_probs"][r] for r in roles],
            dtype=np.float32,
        )

        kvec0 = np.asarray(
            [
                x0["k_probs"].get(str(k), 0.0)
                for k in (3, 4, 5, 6)
            ],
            dtype=np.float32,
        )

        rng_particles = np.random.default_rng(
            stable_seed(
                sid,
                rnd,
                args.seed,
                "particles",
            )
        )

        Z, W0, Klabels, Kprior0 = eig.stratified_particles(
            x0["k_probs"],
            x0["role_probs"],
            roles,
            args.particles_per_k,
            rng_particles,
        )

        W0 = normw(W0)

        rnorm0 = float(
            rnd / max(max_round, 1)
        )

        rnorm1 = float(
            min(rnd + 1, max_round)
            / max(max_round, 1)
        )

        Pe0, immediate, immediate_parts = (
            visible_channel_for_pool(
                model,
                Q,
                idxs,
                belief0,
                kvec0,
                rnorm0,
                Z,
                W0,
                Klabels,
                log_posw,
                T,
                tpr,
                fpr,
                args.objective,
                args.hybrid_alpha,
                args.batch_size,
            )
        )

        C = len(rows)

        h2_scores = np.full(
            C,
            -np.inf,
            dtype=np.float64,
        )

        future_values = np.zeros(
            C,
            dtype=np.float64,
        )

        for j in range(C):
            rem_local = [
                k for k in range(C)
                if k != j
            ]

            rem_idxs = [
                idxs[k]
                for k in rem_local
            ]

            branch_vals = []

            rng_branch = np.random.default_rng(
                stable_seed(
                    sid,
                    rnd,
                    args.seed,
                    "q1",
                    j,
                )
            )

            for b in range(args.branches):
                # Sample one latent posterior particle, then
                # visible detector evidence conditional on it.
                m = int(
                    rng_branch.choice(
                        np.arange(len(W0)),
                        p=W0,
                    )
                )

                evidence_sim = (
                    rng_branch.random(len(roles))
                    < Pe0[j, m]
                ).astype(np.float64)

                W1_sim = vb.update_weights(
                    Pe0[j],
                    evidence_sim,
                    W0,
                    beta,
                )

                belief1_sim = belief_from_weights(
                    W1_sim,
                    Z,
                )

                kvec1_sim = kvec_from_weights(
                    W1_sim,
                    Klabels,
                )

                _, score2, _ = visible_channel_for_pool(
                    model,
                    Q,
                    rem_idxs,
                    belief1_sim,
                    kvec1_sim,
                    rnorm1,
                    Z,
                    W1_sim,
                    Klabels,
                    log_posw,
                    T,
                    tpr,
                    fpr,
                    args.objective,
                    args.hybrid_alpha,
                    args.batch_size,
                )

                branch_vals.append(
                    float(np.max(score2))
                )

            future_values[j] = float(
                np.mean(branch_vals)
            )

            h2_scores[j] = (
                float(immediate[j])
                + args.gamma * future_values[j]
            )

        j_greedy = int(np.argmax(immediate))
        j_h2 = int(np.argmax(h2_scores))

        evidence_actual = {
            j: (
                Pvisible[idxs[j]]
                >= thresholds
            ).astype(np.float64)
            for j in range(C)
        }

        true_roles = set(
            map(str, x0["truth_roles"])
        )

        true_z = np.zeros(
            len(roles),
            dtype=np.float64,
        )

        for r in true_roles:
            if r in ri:
                true_z[ri[r]] = 1.0

        true_k = len(true_roles)

        prior_metrics = posterior_metrics(
            W0,
            Z,
            Klabels,
            true_z,
            true_k,
        )

        # Cache actual first-step branches. They are also
        # useful for oracle upper bounds below.
        branch_cache = {}

        def actual_branch(j1):
            if j1 in branch_cache:
                return branch_cache[j1]

            W1 = actual_update(
                W0,
                Pe0[j1],
                evidence_actual[j1],
                beta,
            )

            belief1 = belief_from_weights(
                W1,
                Z,
            )

            kvec1 = kvec_from_weights(
                W1,
                Klabels,
            )

            rem = [
                k for k in range(C)
                if k != j1
            ]

            rem_idxs = [
                idxs[k] for k in rem
            ]

            Pe2, score2, _ = visible_channel_for_pool(
                model,
                Q,
                rem_idxs,
                belief1,
                kvec1,
                rnorm1,
                Z,
                W1,
                Klabels,
                log_posw,
                T,
                tpr,
                fpr,
                args.objective,
                args.hybrid_alpha,
                args.batch_size,
            )

            branch_cache[j1] = (
                W1,
                rem,
                Pe2,
                score2,
            )

            return branch_cache[j1]

        def run_receding_policy(j1):
            W1, rem, Pe2, score2 = actual_branch(j1)

            jj = int(np.argmax(score2))
            j2 = rem[jj]

            W2 = actual_update(
                W1,
                Pe2[jj],
                evidence_actual[j2],
                beta,
            )

            post = posterior_metrics(
                W2,
                Z,
                Klabels,
                true_z,
                true_k,
            )

            return {
                "q1_local": int(j1),
                "q1_candidate_index": int(
                    rows[j1][1]["candidate_index"]
                ),
                "q2_local": int(j2),
                "q2_candidate_index": int(
                    rows[j2][1]["candidate_index"]
                ),
                "role_brier_gain":
                    prior_metrics["role_brier"]
                    - post["role_brier"],
                "k_brier_gain":
                    prior_metrics["k_brier"]
                    - post["k_brier"],
                "p_true_k_gain":
                    post["p_true_k"]
                    - prior_metrics["p_true_k"],
                "p_true_z_gain":
                    post["p_true_z"]
                    - prior_metrics["p_true_z"],
                "post": post,
            }

        greedy_result = run_receding_policy(
            j_greedy
        )

        h2_result = run_receding_policy(
            j_h2
        )

        # Retrospective fixed-bank oracle:
        # every ordered q1,q2 pair is scored using recorded
        # visible evidence. Truth is used ONLY for evaluation.
        pair_results = []

        for j1 in range(C):
            W1, rem, Pe2, score2 = actual_branch(j1)

            for jj, j2 in enumerate(rem):
                W2 = actual_update(
                    W1,
                    Pe2[jj],
                    evidence_actual[j2],
                    beta,
                )

                post = posterior_metrics(
                    W2,
                    Z,
                    Klabels,
                    true_z,
                    true_k,
                )

                pair_results.append({
                    "j1": j1,
                    "j2": j2,
                    "role_brier_gain":
                        prior_metrics["role_brier"]
                        - post["role_brier"],
                    "k_brier_gain":
                        prior_metrics["k_brier"]
                        - post["k_brier"],
                    "p_true_k_gain":
                        post["p_true_k"]
                        - prior_metrics["p_true_k"],
                    "p_true_z_gain":
                        post["p_true_z"]
                        - prior_metrics["p_true_z"],
                })

        oracle_role = max(
            pair_results,
            key=lambda x: x["role_brier_gain"],
        )

        oracle_k = max(
            pair_results,
            key=lambda x: x["k_brier_gain"],
        )

        rec = {
            "parent": parent,
            "system_id": sid,
            "round": rnd,
            "n_candidates": C,
            "prior": prior_metrics,
            "greedy_q1": {
                "local": j_greedy,
                "candidate_index": int(
                    rows[j_greedy][1]["candidate_index"]
                ),
                "immediate_score": float(
                    immediate[j_greedy]
                ),
                "h2_score": float(
                    h2_scores[j_greedy]
                ),
            },
            "horizon2_q1": {
                "local": j_h2,
                "candidate_index": int(
                    rows[j_h2][1]["candidate_index"]
                ),
                "immediate_score": float(
                    immediate[j_h2]
                ),
                "future_value": float(
                    future_values[j_h2]
                ),
                "h2_score": float(
                    h2_scores[j_h2]
                ),
            },
            "q1_differs": bool(
                j_greedy != j_h2
            ),
            "greedy_terminal": greedy_result,
            "horizon2_terminal": h2_result,
            "oracle_role_pair_eval_only": oracle_role,
            "oracle_k_pair_eval_only": oracle_k,
        }

        per_state.append(rec)

        print(
            f"[{gi:03d}/{len(groups):03d}] "
            f"{sid} "
            f"C={C} "
            f"q1 greedy={j_greedy} "
            f"h2={j_h2} "
            f"diff={int(j_greedy != j_h2)} "
            f"roleGain "
            f"{greedy_result['role_brier_gain']:+.5f}"
            f"->{h2_result['role_brier_gain']:+.5f} "
            f"kGain "
            f"{greedy_result['k_brier_gain']:+.5f}"
            f"->{h2_result['k_brier_gain']:+.5f}",
            flush=True,
        )

    print("[6/6] summary")

    summary = {
        "setting": {
            "round_only": args.round_only,
            "states": len(per_state),
            "particles_per_k": args.particles_per_k,
            "branches": args.branches,
            "gamma": args.gamma,
            "objective": args.objective,
            "hybrid_alpha": args.hybrid_alpha,
            "real_belief_mutated_during_planning": False,
            "simulated_beliefs_discarded_after_scoring": True,
            "q2_bank": "remaining candidates from same pre-execution state",
            "counterfactual_limitation":
                "Recorded q2 visible evidence was generated from the original "
                "state, not after counterfactual q1. This is a fixed-bank "
                "offline planning diagnostic, not final sequential deployment.",
            "truth_used_for_candidate_scoring": False,
            "truth_used_for_retrospective_evaluation_only": True,
        },
        "q1_difference_rate": mean([
            float(x["q1_differs"])
            for x in per_state
        ]),
        "role_brier_gain": {
            "greedy": mean([
                x["greedy_terminal"]["role_brier_gain"]
                for x in per_state
            ]),
            "horizon2": mean([
                x["horizon2_terminal"]["role_brier_gain"]
                for x in per_state
            ]),
            "oracle_fixed_bank": mean([
                x["oracle_role_pair_eval_only"]["role_brier_gain"]
                for x in per_state
            ]),
        },
        "k_brier_gain": {
            "greedy": mean([
                x["greedy_terminal"]["k_brier_gain"]
                for x in per_state
            ]),
            "horizon2": mean([
                x["horizon2_terminal"]["k_brier_gain"]
                for x in per_state
            ]),
            "oracle_fixed_bank": mean([
                x["oracle_k_pair_eval_only"]["k_brier_gain"]
                for x in per_state
            ]),
        },
        "p_true_k_gain": {
            "greedy": mean([
                x["greedy_terminal"]["p_true_k_gain"]
                for x in per_state
            ]),
            "horizon2": mean([
                x["horizon2_terminal"]["p_true_k_gain"]
                for x in per_state
            ]),
        },
        "p_true_z_gain": {
            "greedy": mean([
                x["greedy_terminal"]["p_true_z_gain"]
                for x in per_state
            ]),
            "horizon2": mean([
                x["horizon2_terminal"]["p_true_z_gain"]
                for x in per_state
            ]),
        },
    }

    (out / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    with (out / "per_state.jsonl").open(
        "w",
        encoding="utf-8",
    ) as f:
        for x in per_state:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print("=" * 92)
    print("RESULT")
    print("=" * 92)
    print(json.dumps(summary, indent=2))
    print()
    print("wrote:", out)


if __name__ == "__main__":
    main()
