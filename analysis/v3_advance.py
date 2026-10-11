#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("analysis").resolve()))

import observation_model_v1 as om
import visible_adaptive_bridge_v1 as vb

from adaptive_v11_active.schema import EvidenceFrame, PublicBeliefState
from adaptive_v11_curriculum.reward import group_relative_advantages
from v11.role_universe import TRAIN_ROLES

from v3_common import (
    diagnosticity,
    k_probs,
    largest_gap_k,
    load_matrix_npz,
    map_k,
    normalize_weights,
    reward_visible_truth_gain,
    role_probs,
    save_matrix_npz,
    topk_roles,
    true_mask,
)

OBS = Path("artifacts/observation_model_v1")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--split", choices=["train", "dev"], default="train")
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    root = Path(args.root)
    roles = list(TRAIN_ROLES)
    n_roles = len(roles)

    Z = np.load(root / "hypotheses.npy").astype(np.float32)
    Klabels = np.load(root / "klabels.npy")
    if len(Z) != 60249:
        raise RuntimeError(f"bad hypothesis count: {len(Z)}")

    p_ids, P = load_matrix_npz(
        root / f"posterior_{args.split}_round_{args.round-1:02d}.npz",
        "weights",
    )
    c_ids, C = load_matrix_npz(
        root / f"coverage_{args.split}_round_{args.round-1:02d}.npz",
        "coverage",
    )
    if p_ids != c_ids:
        raise RuntimeError("posterior/coverage id order mismatch")
    row_of = {sid:i for i,sid in enumerate(p_ids)}

    states = {
        x["session_id"]: PublicBeliefState.from_public_dict(x)
        for x in read_jsonl(root / f"states_{args.split}_round_{args.round-1:02d}.jsonl")
    }
    cand = read_jsonl(root / f"candidates_{args.split}_round_{args.round:02d}.jsonl")
    resp = read_jsonl(root / f"responses_{args.split}_round_{args.round:02d}.jsonl")
    resp_by = {
        (x["session_id"], int(x["candidate_index"])): x
        for x in resp
    }

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    truth = {
        x["session_id"]: x
        for x in manifest["systems"]
        if x["split"] == args.split
    }

    asset = joblib.load(root / "visible_bridge.joblib")
    if list(asset["roles"]) != roles:
        raise RuntimeError("visible bridge role order mismatch")

    fp_models = asset["models"]
    fp_cals = asset["cals"]
    thresholds = np.asarray(asset["thresholds"], dtype=np.float64)
    tpr = np.asarray(asset["tpr"], dtype=np.float64)
    fpr = np.asarray(asset["fpr"], dtype=np.float64)
    log_posw = np.asarray(asset["log_posweight"], dtype=np.float64)
    T = float(asset["temperature"])
    beta = float(asset["beta"])

    ckpt = torch.load(OBS / "model.pt", map_location="cpu")
    fwd = om.ObservationNet(
        qdim=ckpt["query_embedding_dim"],
        n_roles=n_roles,
        hidden=ckpt["hidden"],
        dropout=ckpt["dropout"],
    )
    fwd.load_state_dict(ckpt["state_dict"])
    fwd.eval()

    # Embed every executed candidate once.
    valid_pairs = []
    for c in cand:
        rr = resp_by.get((c["session_id"], int(c["candidate_index"])))
        if c.get("valid") and rr and rr.get("execution_status") == "ok":
            valid_pairs.append((c, rr))

    embedder = SentenceTransformer(args.embedding_model)
    q_texts = [c["query"] for c, _ in valid_pairs]
    o_texts = [r.get("final_output", "") for _, r in valid_pairs]
    Q = np.asarray(embedder.encode(q_texts, convert_to_numpy=True), dtype=np.float32)
    O = np.asarray(embedder.encode(o_texts, convert_to_numpy=True), dtype=np.float32)
    XV = np.concatenate([Q, O], axis=1)
    visible_soft = vb.visible_probs(fp_models, fp_cals, XV)
    emb_by = {}
    for i, (c, rr) in enumerate(valid_pairs):
        emb_by[(c["session_id"], int(c["candidate_index"]))] = (
            Q[i:i+1],
            np.asarray(visible_soft[i], dtype=np.float64),
        )

    grouped = defaultdict(list)
    for c in cand:
        grouped[c["session_id"]].append(c)

    next_states = []
    next_P = np.zeros_like(P, dtype=np.float64)
    next_C = np.zeros_like(C, dtype=np.float32)
    train_rows = []
    metrics = []
    candidate_diag_rows = []

    for sid, cs in sorted(grouped.items()):
        if sid not in states or sid not in truth or sid not in row_of:
            raise RuntimeError(f"missing state/truth/posterior for {sid}")

        st = states[sid]
        ii = row_of[sid]
        W = normalize_weights(P[ii])
        cov = np.asarray(C[ii], dtype=np.float64)
        tr = truth[sid]
        true_roles = tuple(map(str, tr["roles"]))
        true_k = len(true_roles)

        kp_before = k_probs(W, Klabels)
        rp_before = role_probs(W, Z, roles)

        # Consistency audit: policy-visible state should be the exact posterior marginals.
        max_rerr = max(abs(float(st.role_probs[r]) - rp_before[r]) for r in roles)
        max_kerr = max(abs(float(st.k_probs[k]) - kp_before[int(k)]) for k in st.k_probs)
        if max_rerr > 2e-4 or max_kerr > 2e-4:
            raise RuntimeError(
                f"state/posterior mismatch sid={sid} role_err={max_rerr} k_err={max_kerr}"
            )

        kvec = np.asarray([kp_before[k] for k in (3,4,5,6)], dtype=np.float32)
        belief = np.asarray([rp_before[r] for r in roles], dtype=np.float32)
        rnorm = float(min(args.round, 8) / 8.0)
        prev_queries = [h.query for h in st.history]

        evals = []

        for c in sorted(cs, key=lambda x: int(x["candidate_index"])):
            idx = int(c["candidate_index"])
            rr = resp_by.get((sid, idx))
            valid = bool(
                c.get("valid")
                and rr is not None
                and rr.get("execution_status") == "ok"
                and (sid, idx) in emb_by
            )

            if not valid:
                rb = {
                    "total": -1.5,
                    "invalid_penalty": 1.5,
                    "valid": False,
                    "invalid_reason": c.get("invalid_reason") or (
                        rr.get("execution_error", "") if rr else "missing_response"
                    ),
                }
                evals.append((c, rr, None, np.zeros(n_roles), None, rb, None))
                continue

            qemb, pvis = emb_by[(sid, idx)]
            logits = vb.particle_logits(
                fwd,
                qemb,
                belief,
                kvec,
                rnorm,
                Z,
                "cpu",
                args.batch_size,
            )
            # The bridge helpers may preserve singleton candidate/batch axes.
            # For one candidate we require exactly:
            #   particles x roles == (60249, 20)
            pY = np.asarray(
                vb.calibrated_hidden_probs(logits, log_posw, T),
                dtype=np.float64,
            )
            pY = np.squeeze(pY)

            if pY.shape != Z.shape:
                raise RuntimeError(
                    f"hidden-prob shape mismatch: "
                    f"pY={pY.shape} expected={Z.shape} "
                    f"logits={np.asarray(logits).shape}"
                )

            pE = np.asarray(
                vb.detector_channel_probs(pY, tpr, fpr),
                dtype=np.float64,
            )
            pE = np.squeeze(pE)

            if pE.shape != Z.shape:
                raise RuntimeError(
                    f"detector-channel shape mismatch: "
                    f"pE={pE.shape} expected={Z.shape}"
                )

            evidence = np.asarray(
                pvis >= thresholds,
                dtype=np.float64,
            ).reshape(-1)

            if evidence.shape != (n_roles,):
                raise RuntimeError(
                    f"visible evidence shape mismatch: "
                    f"{evidence.shape} expected={(n_roles,)}"
                )

            Wnew = normalize_weights(
                vb.update_weights(pE, evidence, W, beta)
            )
            diag = diagnosticity(pE, W, Z)
            rb = reward_visible_truth_gain(
                before_w=W,
                after_w=Wnew,
                klabels=Klabels,
                Z=Z,
                roles=roles,
                truth_roles=true_roles,
                coverage=cov,
                diag=diag,
                query=str(c["query"]),
                previous_queries=prev_queries,
            )
            rb["valid"] = True
            rb["visible_evidence_cardinality"] = int(evidence.sum())
            rb["visible_soft_cardinality"] = float(pvis.sum())
            evals.append((c, rr, Wnew, diag, pvis, rb, evidence))

        rewards = [float(x[5]["total"]) for x in evals]
        advs = group_relative_advantages(rewards)

        chosen_item = None
        for item, adv in zip(evals, advs):
            c, rr, Wnew, diag, pvis, rb, evidence = item
            chosen = bool(c.get("chosen_for_continuation")) and Wnew is not None
            if chosen:
                chosen_item = item

            if args.split == "train" and c.get("completion"):
                train_rows.append({
                    "session_id": sid,
                    "round": args.round,
                    "candidate_index": int(c["candidate_index"]),
                    "group_id": c["group_id"],
                    "prompt": c["prompt"],
                    "completion": c["completion"],
                    "old_logp": float(c["old_logp"]),
                    "ref_logp": float(c["ref_logp"]),
                    "reward": float(rb["total"]),
                    "local_advantage": float(adv),
                    "chosen_for_continuation": bool(chosen),
                    "reward_breakdown_training_only": rb,
                })

            candidate_diag_rows.append({
                "session_id": sid,
                "round": args.round,
                "candidate_index": int(c["candidate_index"]),
                "chosen_for_continuation": bool(chosen),
                "query": c.get("query", ""),
                "reward": float(rb["total"]),
                "diagnosticity": {
                    r: float(diag[i]) for i,r in enumerate(roles)
                } if Wnew is not None else {},
                "visible_probs": {
                    r: float(pvis[i]) for i,r in enumerate(roles)
                } if pvis is not None else {},
            })

        if chosen_item is None:
            # No successful chosen execution: carry state unchanged.
            Wnext = W.copy()
            cov_next = cov.copy()
            history = list(st.history)
            chosen_reward = None
            chosen_query = None
            status = "carry_no_valid_chosen"
        else:
            c, rr, Wnext, diag, pvis, rb, evidence = chosen_item
            cov_next = np.maximum(cov, diag)
            frame = EvidenceFrame(
                query=str(c["query"]),
                answer=str(rr.get("final_output", "")),
                v135={r: float(pvis[i]) for i,r in enumerate(roles)},
                v134={},
            )
            history = [*st.history, frame]
            chosen_reward = float(rb["total"])
            chosen_query = str(c["query"])
            status = "ok"

        kp = k_probs(Wnext, Klabels)
        rp = role_probs(Wnext, Z, roles)
        kgap, gap = largest_gap_k(rp)
        tm = true_mask(Z, roles, true_roles)
        pz = float(Wnext[tm].sum())
        top_true = set(topk_roles(rp, true_k))

        ns = PublicBeliefState(
            session_id=sid,
            split=args.split,
            round_index=args.round,
            k_probs=dict(kp),
            role_probs=dict(rp),
            history=history,
        )
        ns.validate(roles, (3,4,5,6))
        next_states.append(ns.to_public_dict())
        next_P[ii] = Wnext
        next_C[ii] = cov_next.astype(np.float32)

        metrics.append({
            "session_id": sid,
            "split": args.split,
            "round": args.round,
            "status": status,
            "true_k_eval_only": true_k,
            "pred_k_map": map_k(kp),
            "k_map_correct": int(map_k(kp) == true_k),
            "gap_k": kgap,
            "gap_k_correct": int(kgap == true_k),
            "gap_size": gap,
            "top_true_k_exact": int(top_true == set(true_roles)),
            "p_true_k": float(kp[true_k]),
            "p_true_composition": pz,
            "mean_coverage": float(cov_next.mean()),
            "min_coverage": float(cov_next.min()),
            "chosen_reward_training_only": chosen_reward,
            "chosen_query": chosen_query,
        })

    # Preserve original posterior id order.
    state_by = {x["session_id"]: x for x in next_states}
    ordered_states = [state_by[sid] for sid in p_ids]
    write_jsonl(root / f"states_{args.split}_round_{args.round:02d}.jsonl", ordered_states)
    save_matrix_npz(
        root / f"posterior_{args.split}_round_{args.round:02d}.npz",
        p_ids, next_P, "weights"
    )
    save_matrix_npz(
        root / f"coverage_{args.split}_round_{args.round:02d}.npz",
        p_ids, next_C, "coverage"
    )
    write_jsonl(root / f"metrics_{args.split}_round_{args.round:02d}.jsonl", metrics)
    write_jsonl(root / f"candidate_diag_{args.split}_round_{args.round:02d}.jsonl", candidate_diag_rows)
    if args.split == "train":
        write_jsonl(root / f"train_rows_round_{args.round:02d}.jsonl", train_rows)

    summary = {
        "split": args.split,
        "round": args.round,
        "systems": len(metrics),
        "valid_state_updates": sum(x["status"] == "ok" for x in metrics),
        "mean_k_map_acc": float(np.mean([x["k_map_correct"] for x in metrics])),
        "mean_gap_k_acc": float(np.mean([x["gap_k_correct"] for x in metrics])),
        "mean_top_true_k_exact": float(np.mean([x["top_true_k_exact"] for x in metrics])),
        "mean_p_true_k": float(np.mean([x["p_true_k"] for x in metrics])),
        "mean_p_true_composition": float(np.mean([x["p_true_composition"] for x in metrics])),
        "mean_coverage": float(np.mean([x["mean_coverage"] for x in metrics])),
        "mean_chosen_reward": float(np.mean([
            x["chosen_reward_training_only"]
            for x in metrics
            if x["chosen_reward_training_only"] is not None
        ])) if any(x["chosen_reward_training_only"] is not None for x in metrics) else None,
        "visible_beta": beta,
        "truth_boundary": "truth only used in reward/metrics; state update uses visible query+final_output fingerprint",
    }
    (root / f"advance_{args.split}_round_{args.round:02d}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
