#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("analysis").resolve()))

import observation_model_v1 as om
import visible_adaptive_bridge_v1 as vb

ROOT = Path("artifacts/adaptive_v11_curriculum_v2")
OBS = Path("artifacts/observation_model_v1")
VIS = Path("artifacts/visible_output_fingerprint_v1")
CAL = Path("artifacts/calibrated_bayes_diag_v1")
OUT = Path("artifacts/k_gap_exact_current_v1")

KS = (3, 4, 5, 6)
THRESHOLDS = tuple(round(x, 2) for x in np.arange(0.20, 0.801, 0.05))


def is_true(v):
    if v is True or v == 1:
        return True
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes"}
    return False


def read_jsonl(path):
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def norm(w):
    w = np.asarray(w, dtype=np.float64)
    s = float(w.sum())
    if not np.isfinite(s) or s <= 0:
        raise RuntimeError("invalid posterior normalization")
    return w / s


def exact_hypotheses(n_roles):
    zs, ks, ws = [], [], []
    for k in KS:
        combs = itertools.combinations(range(n_roles), k)
        n = math.comb(n_roles, k)
        per = (1.0 / len(KS)) / n
        for c in combs:
            z = np.zeros(n_roles, dtype=np.float32)
            z[list(c)] = 1.0
            zs.append(z)
            ks.append(k)
            ws.append(per)
    return np.stack(zs), np.asarray(ks, dtype=np.int64), norm(ws)


def k_probs(W, K):
    return {str(k): float(W[K == k].sum()) for k in KS}


def role_probs(W, Z, roles):
    p = W @ Z
    return {r: float(p[i]) for i, r in enumerate(roles)}


def k_map(kp):
    return int(max(kp, key=kp.get))


def sorted_role_values(rp):
    return sorted(((r, float(p)) for r, p in rp.items()),
                  key=lambda x: (-x[1], x[0]))


def largest_gap_k(rp):
    vals = [p for _, p in sorted_role_values(rp)]
    choices = []
    for k in KS:
        gap = float(vals[k - 1] - vals[k])
        choices.append((gap, -k, k))
    gap, _, k = max(choices)
    return int(k), float(gap)


def threshold_k(rp, tau):
    c = sum(float(p) >= float(tau) for p in rp.values())
    return int(min(max(c, min(KS)), max(KS)))


def round_sum_k(rp):
    return int(min(max(round(sum(map(float, rp.values()))), min(KS)), max(KS)))


def two_cluster_k(rp):
    x = np.asarray(list(map(float, rp.values())), dtype=np.float64)
    lo = float(np.quantile(x, 0.25))
    hi = float(np.quantile(x, 0.75))
    high = x >= (lo + hi) / 2.0
    for _ in range(100):
        if high.all() or (~high).all():
            break
        nlo = float(x[~high].mean())
        nhi = float(x[high].mean())
        nhigh = np.abs(x - nhi) < np.abs(x - nlo)
        if np.array_equal(nhigh, high):
            high = nhigh
            break
        lo, hi, high = nlo, nhi, nhigh
    count = int(high.sum())
    return int(min(max(count, min(KS)), max(KS)))


def role_structure(rp, truth):
    ordered = sorted_role_values(rp)
    truth = set(truth)
    true_ps = [rp[r] for r in truth]
    false_ps = [p for r, p in rp.items() if r not in truth]
    margin = float(min(true_ps) - max(false_ps))
    top_true_k = {r for r, _ in ordered[:len(truth)]}
    kg, gap = largest_gap_k(rp)
    gap_set = {r for r, _ in ordered[:kg]}
    return {
        "true_false_margin": margin,
        "perfect_role_ranking": bool(margin > 0),
        "top_true_k_exact": bool(top_true_k == truth),
        "gap_k": int(kg),
        "gap_size": float(gap),
        "gap_set_exact": bool(gap_set == truth),
    }


def get_text(x, *keys):
    for k in keys:
        v = x.get(k)
        if v not in (None, ""):
            return str(v)
    return ""


def load_chosen_rows():
    """
    Load only the one candidate that actually continued each recorded trajectory.
    Key by (system_id, round). We do not mix counterfactual siblings.
    """
    chosen = {}
    truth_by_system = {}
    conflicts = []

    files = sorted(ROOT.rglob("responses_*_round_*.jsonl"))
    for p in files:
        for x in read_jsonl(p):
            if not is_true(x.get("chosen_for_continuation")):
                continue
            sid = str(x.get("system_id") or x.get("session_id") or "")
            rnd = x.get("round")
            truth = tuple(sorted(map(str, x.get("truth_roles_training_only") or [])))
            if not sid or rnd is None or len(truth) not in KS:
                continue

            if sid in truth_by_system and truth_by_system[sid] != truth:
                conflicts.append(("truth", sid))
                continue
            truth_by_system[sid] = truth

            key = (sid, int(rnd))
            rec = {
                "system_id": sid,
                "round": int(rnd),
                "query": get_text(x, "query"),
                "output": get_text(x, "final_output", "output", "answer"),
                "selected_roles": tuple(sorted(map(
                    str, x.get("selected_roles_training_only") or []
                ))),
                "truth_roles": truth,
                "source": str(p),
            }
            if key in chosen:
                # This should not occur if system ids are globally unique.
                prev = chosen[key]
                if (prev["query"], prev["output"]) != (rec["query"], rec["output"]):
                    conflicts.append(("duplicate_chosen", key))
                    continue
            chosen[key] = rec

    if conflicts:
        print("[audit] conflicts:", conflicts[:10], "count=", len(conflicts))
    return chosen, truth_by_system


def build_example_index(examples):
    """
    Map chosen raw rows back to the exact example/embedding row.
    """
    d = defaultdict(list)
    for i, x in enumerate(examples):
        sid = str(x.get("system_id") or x.get("session_id") or "")
        rnd = int(x.get("round", -1))
        q = get_text(x, "query")
        d[(sid, rnd, q)].append(i)
    return d


def pick_example_idx(rec, ex_index, examples):
    cand = ex_index.get((rec["system_id"], rec["round"], rec["query"]), [])
    if not cand:
        return None
    if len(cand) == 1:
        return cand[0]
    # Prefer exact output match if duplicates exist.
    for i in cand:
        if get_text(examples[i], "final_output", "output", "answer") == rec["output"]:
            return i
    return cand[0]


def aggregate(rows):
    if not rows:
        return {}
    keys = [
        "k_map_correct", "gap_correct", "two_cluster_correct",
        "round_sum_correct", "top_true_k_exact", "gap_set_exact",
        "perfect_role_ranking",
    ]
    out = {"n": len(rows)}
    for k in keys:
        out[k] = float(np.mean([float(r[k]) for r in rows]))
    out["mean_gap_size"] = float(np.mean([r["gap_size"] for r in rows]))
    out["median_gap_size"] = float(np.median([r["gap_size"] for r in rows]))
    out["mean_true_false_margin"] = float(np.mean([r["true_false_margin"] for r in rows]))
    for tau in THRESHOLDS:
        name = f"threshold_{tau:.2f}_correct"
        out[name] = float(np.mean([float(r[name]) for r in rows]))
    return out


def choose_threshold(val_final):
    if not val_final:
        return 0.50
    scored = []
    for tau in THRESHOLDS:
        key = f"threshold_{tau:.2f}_correct"
        acc = float(np.mean([float(r[key]) for r in val_final]))
        scored.append((acc, -abs(tau - 0.5), tau))
    return float(max(scored)[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--oracle-beta", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=4096)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    print("loading examples + embeddings...")
    examples, roles, _, _ = om.collect_examples(ROOT)
    roles = list(roles)
    n_roles = len(roles)
    print("examples:", len(examples), "roles:", n_roles)

    split = json.loads((OBS / "split_systems.json").read_text())
    split_sets = {
        "train": set(map(str, split["train"])),
        "val": set(map(str, split["val"])),
        "test": set(map(str, split["test"])),
    }

    Qhist = np.load(OBS / "query_embeddings.npy").astype(np.float32)
    Ohist = np.load(VIS / "output_embeddings.npy").astype(np.float32)
    if len(Qhist) != len(examples) or len(Ohist) != len(examples):
        raise RuntimeError(
            f"embedding/example mismatch Q={len(Qhist)} O={len(Ohist)} ex={len(examples)}"
        )

    print("fitting visible fingerprint detector on TRAIN, calibrating VAL...")
    tri = np.asarray([i for i,x in enumerate(examples)
                      if str(x["system_id"]) in split_sets["train"]], dtype=int)
    vai = np.asarray([i for i,x in enumerate(examples)
                      if str(x["system_id"]) in split_sets["val"]], dtype=int)
    Xhist = np.concatenate([Qhist, Ohist], axis=1)
    Yhist = vb.labels(examples, roles, "selected_roles")
    fp_models, fp_cals, thr, tpr, fpr, _ = vb.fit_visible_detector(
        Xhist, Yhist, tri, vai
    )

    print("loading calibrated hidden activation model...")
    ckpt = torch.load(OBS / "model.pt", map_location="cpu")
    fwd = om.ObservationNet(
        qdim=ckpt["query_embedding_dim"],
        n_roles=n_roles,
        hidden=ckpt["hidden"],
        dropout=ckpt["dropout"],
    )
    fwd.load_state_dict(ckpt["state_dict"])
    fwd.eval()

    ri = {r:i for i,r in enumerate(roles)}
    pos = np.zeros(n_roles, dtype=np.float64)
    for i in tri:
        for r in examples[i]["selected_roles"]:
            pos[ri[r]] += 1
    posw = np.clip((len(tri)-pos)/np.maximum(pos,1.0), 1.0, 20.0)
    log_posw = np.log(posw)

    cal = json.loads((CAL / "summary.json").read_text())
    T = float(
        cal["true_Z_probability_calibration"]
           ["corrected_plus_temperature"]["best_T"]
    )
    beta_visible = float(
        cal["validation_posterior_update"]
           ["selected_beta_by_validation_log_gain"]
    )
    beta_oracle = float(args.oracle_beta)

    print("T=", T, "visible_beta=", beta_visible, "oracle_beta=", beta_oracle)

    print("building exact K=3..6 hypothesis space...")
    Z, Klabels, W0 = exact_hypotheses(n_roles)
    print("hypotheses:", len(Z))
    if len(Z) != 60249:
        raise RuntimeError(f"expected 60249 hypotheses, got {len(Z)}")

    chosen, truth_by_system = load_chosen_rows()
    ex_index = build_example_index(examples)

    # Build fixed-budget trajectories from validation and test only.
    trajectories = {}
    audit = defaultdict(int)

    for split_name in ("val", "test"):
        for sid in sorted(split_sets[split_name]):
            truth = truth_by_system.get(sid)
            if truth is None or len(truth) not in KS:
                audit[f"{split_name}_no_truth_or_out_of_range"] += 1
                continue

            recs = []
            for rnd in sorted(r for (s,r) in chosen if s == sid):
                rec = chosen[(sid, rnd)]
                idx = pick_example_idx(rec, ex_index, examples)
                if idx is None:
                    audit[f"{split_name}_chosen_not_in_examples"] += 1
                    continue
                rec = dict(rec)
                rec["example_idx"] = int(idx)
                recs.append(rec)

            if len(recs) < args.budget:
                audit[f"{split_name}_short_trajectory"] += 1
                continue

            trajectories[(split_name, sid)] = recs[:args.budget]

    print("[audit]", dict(audit))
    print("fixed-budget trajectories:",
          {s: sum(1 for k in trajectories if k[0] == s) for s in ("val","test")})

    if not any(k[0] == "test" for k in trajectories):
        raise RuntimeError("no eligible TEST trajectories")

    max_round = max(int(x["round"]) for x in examples)

    rows_by_track_split_round = defaultdict(list)
    per_system = []

    # Replay each system independently with the same exact prior.
    for si, ((split_name, sid), recs) in enumerate(sorted(trajectories.items()), 1):
        truth = tuple(recs[0]["truth_roles"])
        true_k = len(truth)

        W_oracle = W0.copy()
        W_visible = W0.copy()

        for step, rec in enumerate(recs, 1):
            idx = rec["example_idx"]
            qemb = Qhist[idx:idx+1]
            xvis = np.concatenate([Qhist[idx:idx+1], Ohist[idx:idx+1]], axis=1)

            # ---------------- ORACLE track ----------------
            kp_o_before = k_probs(W_oracle, Klabels)
            rp_o_before = role_probs(W_oracle, Z, roles)
            belief_o = np.asarray([rp_o_before[r] for r in roles], np.float32)
            kvec_o = np.asarray([kp_o_before[str(k)] for k in KS], np.float32)
            rnorm = float(min(step, max_round) / max(max_round, 1))

            L_o = vb.particle_logits(
                fwd, qemb, belief_o, kvec_o, rnorm, Z, "cpu", args.batch_size
            )
            pY_o = vb.calibrated_hidden_probs(L_o, log_posw, T)[0]
            sel = set(rec["selected_roles"])
            Y = np.asarray([1.0 if r in sel else 0.0 for r in roles], np.float64)
            W_oracle = norm(vb.update_weights(pY_o, Y, W_oracle, beta_oracle))

            # ---------------- VISIBLE track ----------------
            kp_v_before = k_probs(W_visible, Klabels)
            rp_v_before = role_probs(W_visible, Z, roles)
            belief_v = np.asarray([rp_v_before[r] for r in roles], np.float32)
            kvec_v = np.asarray([kp_v_before[str(k)] for k in KS], np.float32)

            L_v = vb.particle_logits(
                fwd, qemb, belief_v, kvec_v, rnorm, Z, "cpu", args.batch_size
            )
            pY_v = vb.calibrated_hidden_probs(L_v, log_posw, T)[0]
            pE_v = vb.detector_channel_probs(pY_v, tpr, fpr)
            Pvis = vb.visible_probs(fp_models, fp_cals, xvis)[0]
            E = (Pvis >= thr).astype(np.float64)
            W_visible = norm(vb.update_weights(pE_v, E, W_visible, beta_visible))

            for track, W in (("oracle", W_oracle), ("visible", W_visible)):
                kp = k_probs(W, Klabels)
                rp = role_probs(W, Z, roles)
                kg, gap = largest_gap_k(rp)
                struct = role_structure(rp, truth)
                row = {
                    "split": split_name,
                    "system_id": sid,
                    "step": step,
                    "true_k": true_k,
                    "track": track,
                    "k_map": k_map(kp),
                    "k_map_correct": int(k_map(kp) == true_k),
                    "gap_k": kg,
                    "gap_correct": int(kg == true_k),
                    "two_cluster_k": two_cluster_k(rp),
                    "two_cluster_correct": int(two_cluster_k(rp) == true_k),
                    "round_sum_k": round_sum_k(rp),
                    "round_sum_correct": int(round_sum_k(rp) == true_k),
                    **struct,
                }
                for tau in THRESHOLDS:
                    pred = threshold_k(rp, tau)
                    row[f"threshold_{tau:.2f}_k"] = pred
                    row[f"threshold_{tau:.2f}_correct"] = int(pred == true_k)

                rows_by_track_split_round[(track, split_name, step)].append(row)
                if step == args.budget:
                    per_system.append(row)

        if si % 10 == 0:
            print(f"processed systems {si}/{len(trajectories)}", flush=True)

    # Tune threshold ONLY on validation final-budget states.
    selected_threshold = {}
    for track in ("oracle", "visible"):
        val_final = rows_by_track_split_round[(track, "val", args.budget)]
        selected_threshold[track] = choose_threshold(val_final)

    summary = {
        "setting": {
            "budget": args.budget,
            "ks": list(KS),
            "exact_hypotheses": len(Z),
            "roles": roles,
            "visible_beta": beta_visible,
            "oracle_beta": beta_oracle,
            "temperature": T,
            "threshold_selected_on_validation_only": selected_threshold,
            "trajectory_rule": "recorded chosen_for_continuation candidate only",
            "test_is_held_out_by_system": True,
            "new_mas_calls": 0,
        },
        "audit": dict(audit),
        "per_round": {},
        "final_test": {},
    }

    for track in ("oracle", "visible"):
        summary["per_round"][track] = {}
        for split_name in ("val", "test"):
            summary["per_round"][track][split_name] = {}
            for step in range(1, args.budget + 1):
                rr = rows_by_track_split_round[(track, split_name, step)]
                summary["per_round"][track][split_name][str(step)] = aggregate(rr)

        test_final = rows_by_track_split_round[(track, "test", args.budget)]
        agg = aggregate(test_final)
        tau = selected_threshold[track]
        key = f"threshold_{tau:.2f}_correct"
        agg["selected_threshold"] = tau
        agg["selected_threshold_accuracy"] = (
            float(np.mean([r[key] for r in test_final])) if test_final else None
        )
        summary["final_test"][track] = agg

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Save final per-system rows for auditing.
    with (OUT / "final_systems.jsonl").open("w", encoding="utf-8") as f:
        for r in per_system:
            f.write(json.dumps(r) + "\n")

    print("\n" + "="*100)
    print("FINAL HELD-OUT TEST @ FIXED BUDGET", args.budget)
    print("="*100)

    for track in ("oracle", "visible"):
        x = summary["final_test"][track]
        print(f"\n[{track.upper()}] n={x.get('n',0)}")
        if not x:
            continue
        print(f"  K posterior MAP      : {x['k_map_correct']:.4f}")
        print(f"  LARGEST GAP         : {x['gap_correct']:.4f}")
        print(f"  TWO CLUSTER         : {x['two_cluster_correct']:.4f}")
        print(f"  ROUND SUM           : {x['round_sum_correct']:.4f}")
        print(f"  VAL-TUNED THRESHOLD : tau={x['selected_threshold']:.2f} "
              f"acc={x['selected_threshold_accuracy']:.4f}")
        print(f"  TOP-TRUE-K SET exact: {x['top_true_k_exact']:.4f}")
        print(f"  GAP SET exact       : {x['gap_set_exact']:.4f}")
        print(f"  perfect role ranking: {x['perfect_role_ranking']:.4f}")
        print(f"  mean true/false margin: {x['mean_true_false_margin']:.4f}")
        print(f"  median gap size       : {x['median_gap_size']:.4f}")

    print("\nSAVED:", OUT / "summary.json")
    print("SAVED:", OUT / "final_systems.jsonl")


if __name__ == "__main__":
    main()
