from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("analysis").resolve()))

import observation_model_v1 as om
import visible_adaptive_bridge_v1 as vb
from v11.role_universe import TRAIN_ROLES


ROOT = Path("artifacts/adaptive_v11_curriculum_v2")
OBS = Path("artifacts/observation_model_v1")
CAL = Path("artifacts/calibrated_bayes_diag_v1")
TRAJ = Path("artifacts/one_system_k3_demo/trajectory.json")
OUT = Path("artifacts/oracle_activation_exact_replay")

ROLES = list(TRAIN_ROLES)
KS = (3, 4, 5, 6)
TRUE = {"code", "planning", "research"}
BETAS = (0.10, 0.25, 0.50, 1.00)


def normalize(w):
    w = np.asarray(w, dtype=np.float64)
    return w / max(w.sum(), 1e-300)


def build_exact_hypotheses():
    zs = []
    ks = []
    weights = []

    for k in KS:
        combs = list(itertools.combinations(range(len(ROLES)), k))
        prior_each = (1.0 / len(KS)) / len(combs)

        for c in combs:
            z = np.zeros(len(ROLES), dtype=np.float32)
            z[list(c)] = 1.0
            zs.append(z)
            ks.append(k)
            weights.append(prior_each)

    Z = np.stack(zs)
    K = np.asarray(ks, dtype=np.int64)
    W = normalize(weights)

    return Z, K, W


def k_probs(W, K):
    return {
        str(k): float(W[K == k].sum())
        for k in KS
    }


def role_probs(W, Z):
    p = W @ Z
    return {
        r: float(p[i])
        for i, r in enumerate(ROLES)
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    trajectory = json.loads(TRAJ.read_text())

    print("ROLES:", len(ROLES))
    print("TRUE:", sorted(TRUE))
    print("ROUNDS:", len(trajectory))

    # ------------------------------------------------------------------
    # Exact hypothesis space: all subsets of size 3..6.
    # ------------------------------------------------------------------
    Z, Klabels, W0 = build_exact_hypotheses()

    print("\nEXACT HYPOTHESIS COUNT:", len(Z))
    for k in KS:
        print(f"K={k}: {(Klabels == k).sum()} hypotheses")

    true_vec = np.asarray(
        [1.0 if r in TRUE else 0.0 for r in ROLES],
        dtype=np.float32,
    )

    exact_truth_mask = np.all(Z == true_vec[None, :], axis=1)
    print("EXACT TRUE COMPOSITION PRESENT:", bool(exact_truth_mask.any()))
    print("TRUE PRIOR MASS:", float(W0[exact_truth_mask].sum()))

    # ------------------------------------------------------------------
    # Load same calibrated hidden activation model used by the demo.
    # ------------------------------------------------------------------
    examples, obs_roles, _, _ = om.collect_examples(ROOT)

    if list(obs_roles) != ROLES:
        raise RuntimeError(
            f"role ordering mismatch\nobs={obs_roles}\nroles={ROLES}"
        )

    split = json.loads((OBS / "split_systems.json").read_text())
    train_systems = set(map(str, split["train"]))

    train_idx = np.asarray([
        i for i, x in enumerate(examples)
        if str(x["system_id"]) in train_systems
    ], dtype=int)

    ckpt = torch.load(OBS / "model.pt", map_location="cpu")

    model = om.ObservationNet(
        qdim=ckpt["query_embedding_dim"],
        n_roles=len(ROLES),
        hidden=ckpt["hidden"],
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Inverse positive-class weighting correction.
    ri = {r: i for i, r in enumerate(ROLES)}
    pos = np.zeros(len(ROLES), dtype=np.float64)

    for i in train_idx:
        for r in examples[i]["selected_roles"]:
            pos[ri[r]] += 1

    posweight = np.clip(
        (len(train_idx) - pos) / np.maximum(pos, 1.0),
        1.0,
        20.0,
    )
    log_posweight = np.log(posweight)

    cal = json.loads((CAL / "summary.json").read_text())
    T = float(
        cal["true_Z_probability_calibration"]
           ["corrected_plus_temperature"]["best_T"]
    )

    print("\nCALIBRATION T:", T)

    # Embed the exact already-executed queries once.
    embedder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    queries = [r["query"] for r in trajectory]
    Q = np.asarray(
        embedder.encode(queries, convert_to_numpy=True),
        dtype=np.float32,
    )

    max_round = max(int(x["round"]) for x in examples)

    union_selected = set()
    for r in trajectory:
        union_selected |= set(r["selected_roles_eval_only"])

    print("\nOBSERVED ACTIVATION UNION:", sorted(union_selected))
    print("UNION SIZE:", len(union_selected))
    print(
        "NOTE: activation union alone only proves K >=",
        len(union_selected),
    )

    all_results = {}

    # ------------------------------------------------------------------
    # Replay separately for several beta values.
    # Each replay starts from the SAME exact prior.
    # ------------------------------------------------------------------
    for beta in BETAS:

        print("\n" + "#" * 92)
        print("BETA =", beta)
        print("#" * 92)

        W = W0.copy()
        rows = []

        for t, row in enumerate(trajectory, start=1):

            kp_before = k_probs(W, Klabels)
            rp_before = role_probs(W, Z)

            belief = np.asarray(
                [rp_before[r] for r in ROLES],
                dtype=np.float32,
            )

            kvec = np.asarray(
                [kp_before[str(k)] for k in KS],
                dtype=np.float32,
            )

            rnorm = float(t / max(max_round, 1))

            logits = vb.particle_logits(
                model,
                Q[t - 1:t],
                belief,
                kvec,
                rnorm,
                Z,
                "cpu",
                4096,
            )

            pY = vb.calibrated_hidden_probs(
                logits,
                log_posweight,
                T,
            )[0]

            selected = set(row["selected_roles_eval_only"])

            Y = np.asarray(
                [1.0 if r in selected else 0.0 for r in ROLES],
                dtype=np.float64,
            )

            # IMPORTANT:
            # Oracle/EVAL diagnostic only.
            # Update from TRUE latent activation, not visible output detector.
            W = vb.update_weights(
                pY,
                Y,
                W,
                beta,
            )
            W = normalize(W)

            kp_after = k_probs(W, Klabels)
            rp_after = role_probs(W, Z)

            truth_mass = float(W[exact_truth_mask].sum())

            top = sorted(
                rp_after.items(),
                key=lambda x: -x[1],
            )[:8]

            print("\nROUND", t)
            print("QUERY:", row["query"])
            print("TRUE SELECTED:", sorted(selected))
            print("P(K):", {
                k: round(v, 6)
                for k, v in kp_after.items()
            })
            print("TRUE-COMPOSITION MASS:", truth_mass)

            print("TOP ROLES:")
            for r, p in top:
                flag = " <-- TRUE" if r in TRUE else ""
                print(f"  {r:<28} {p:.4f}{flag}")

            rows.append({
                "round": t,
                "query": row["query"],
                "selected_roles_eval_only": sorted(selected),
                "k_probs": kp_after,
                "role_probs": rp_after,
                "true_composition_mass": truth_mass,
            })

        kp = k_probs(W, Klabels)
        rp = role_probs(W, Z)

        pred_k = int(max(kp, key=kp.get))
        top3 = [
            r for r, _ in
            sorted(rp.items(), key=lambda x: -x[1])[:3]
        ]

        result = {
            "beta": beta,
            "pred_k": pred_k,
            "p_k3": kp["3"],
            "k_correct": pred_k == 3,
            "top3": top3,
            "top3_exact": set(top3) == TRUE,
            "true_composition_mass":
                float(W[exact_truth_mask].sum()),
            "final_k_probs": kp,
            "final_role_probs": rp,
            "rounds": rows,
        }

        all_results[str(beta)] = result

        print("\nFINAL beta =", beta)
        print("PRED K:", pred_k)
        print("P(K=3):", kp["3"])
        print("TOP3:", top3)
        print("TOP3 EXACT:", result["top3_exact"])
        print(
            "TRUE COMPOSITION MASS:",
            result["true_composition_mass"],
        )

    summary = {
        "hypotheses": len(Z),
        "true_roles": sorted(TRUE),
        "activation_union": sorted(union_selected),
        "activation_union_size": len(union_selected),
        "results": all_results,
        "note":
            "EVALUATION-ONLY replay using true selected roles. "
            "No visible outputs or traces are permitted in deployment inference.",
    }

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n\nSAVED:", OUT / "summary.json")


if __name__ == "__main__":
    main()
