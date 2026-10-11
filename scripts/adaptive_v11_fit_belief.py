from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from adaptive_v11_active.belief import FrozenBeliefBundle, systems_from_replay_rows
from adaptive_v11_active.reward import map_k, predicted_set
from adaptive_v11_active.io import read_jsonl, write_json
from v11.role_universe import TRAIN_ROLES

EPS = 1e-6


def find_bank(root: Path) -> Path:
    candidates = []
    for p in sorted(root.glob("*.jsonl")):
        try:
            rows = read_jsonl(p)
        except Exception:
            continue
        if not rows:
            continue
        x = rows[0]
        if isinstance(x.get("evidence"), dict) and isinstance(x.get("metadata", {}).get("evidence_v134"), dict) and (
            "truth_roles" in x or "truth_roles_training_only" in x
        ):
            candidates.append(p)
    if len(candidates) != 1:
        raise SystemExit(
            "Could not uniquely identify the dual-scored baseline replay bank. "
            f"compatible candidates={list(map(str, candidates))}. Pass --bank explicitly."
        )
    return candidates[0]


def eval_systems(bundle: FrozenBeliefBundle, systems) -> dict:
    if not systems:
        return {"systems": 0}
    k_to_i = {k: i for i, k in enumerate(bundle.ks)}
    k_hit = 0; k_abs = 0.0; exact = 0
    k_nll = []; k_brier = []; role_ll = []; role_brier = []; f1s = []
    all_y = []; all_p = []
    for ds in systems:
        kp, rp = bundle.predict(ds.observations)
        pk = np.asarray([max(EPS, float(kp[k])) for k in bundle.ks], dtype=np.float64)
        pk /= pk.sum()
        yi = k_to_i[ds.k]
        yk = np.zeros(len(bundle.ks)); yk[yi] = 1.0
        k_nll.append(-math.log(max(float(pk[yi]), EPS)))
        k_brier.append(float(np.sum((pk - yk) ** 2)))
        khat = int(map_k(kp)); k_hit += int(khat == ds.k); k_abs += abs(khat - ds.k)

        truth = set(ds.truth_roles); pred = predicted_set(rp, kp)
        exact += int(pred == truth)
        tp = len(pred & truth); pr = tp/max(1,len(pred)); rc = tp/max(1,len(truth))
        f1s.append(0.0 if pr+rc == 0 else 2*pr*rc/(pr+rc))
        for r in bundle.roles:
            y = int(r in truth); p = min(1-EPS, max(EPS, float(rp[r])))
            role_ll.append(-(y*math.log(p)+(1-y)*math.log1p(-p)))
            role_brier.append((p-y)**2)
            all_y.append(y); all_p.append(p)

    return {
        "systems": len(systems),
        "k_exact": k_hit/len(systems),
        "k_mae": k_abs/len(systems),
        "k_nll": float(np.mean(k_nll)),
        "k_brier": float(np.mean(k_brier)),
        "role_log_loss": float(np.mean(role_ll)),
        "role_brier": float(np.mean(role_brier)),
        "role_micro_auc": float(roc_auc_score(all_y, all_p)) if len(set(all_y)) == 2 else None,
        "role_set_f1": float(np.mean(f1s)),
        "exact_set": exact/len(systems),
    }


def main():
    ap = argparse.ArgumentParser(description="Fit calibrated target-free frozen belief v2 for Adaptive-V11")
    ap.add_argument("--bank", default=None)
    ap.add_argument("--search-root", default="artifacts/adaptive_v1")
    ap.add_argument("--output", default="artifacts/adaptive_v11_active_v1/frozen_belief.joblib")
    ap.add_argument("--summary", default="artifacts/adaptive_v11_active_v1/frozen_belief_summary.json")
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--folds", type=int, default=4)
    args = ap.parse_args()

    bank = Path(args.bank) if args.bank else find_bank(Path(args.search_root))
    rows = read_jsonl(bank)
    train = systems_from_replay_rows(rows, TRAIN_ROLES, split="train")
    dev = systems_from_replay_rows(rows, TRAIN_ROLES, split="dev")
    if len(train) < 12:
        raise SystemExit(f"too few train systems in {bank}: {len(train)}")

    bundle = FrozenBeliefBundle(TRAIN_ROLES, ks=(3,4,5,6))
    summary = bundle.fit(train, seed=args.seed, folds=args.folds)
    bundle.save(args.output)

    dev_metrics = eval_systems(bundle, dev)
    prior_p = 4.5 / len(TRAIN_ROLES)
    prior = {
        "k_nll": math.log(4.0),
        "k_brier": 0.75,
        "role_log_loss_reference_p0225": -(prior_p*math.log(prior_p)+(1-prior_p)*math.log(1-prior_p)),
        "role_brier_reference_p0225": prior_p*(1-prior_p),
    }
    summary["baseline_dev"] = {
        **dev_metrics,
        "note": "held-out baseline-bank dev; controller-development data, not final paper test",
    }
    summary["uninformative_prior_reference"] = prior
    if dev_metrics.get("systems", 0):
        summary["go_no_go"] = {
            "k_nll_beats_uniform": bool(dev_metrics["k_nll"] < prior["k_nll"]),
            "k_brier_beats_uniform": bool(dev_metrics["k_brier"] < prior["k_brier"]),
            "role_log_loss_beats_symmetric_prior": bool(dev_metrics["role_log_loss"] < prior["role_log_loss_reference_p0225"]),
            "recommended": bool(
                dev_metrics["k_nll"] < prior["k_nll"]
                and dev_metrics["k_brier"] < prior["k_brier"]
                and dev_metrics["role_log_loss"] < prior["role_log_loss_reference_p0225"]
            ),
            "note": "recommended only means the frozen reward model beats simple priors on all three proper-score checks; inspect effect size because dev has only 8 systems",
        }
    summary = {**summary, "source_bank": str(bank), "roles": list(TRAIN_ROLES), "ks": [3,4,5,6]}
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[done] frozen belief v2={args.output}", flush=True)


if __name__ == "__main__":
    main()
