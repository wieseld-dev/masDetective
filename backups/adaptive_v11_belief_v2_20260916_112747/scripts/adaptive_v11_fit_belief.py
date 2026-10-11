from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_v11_active.belief import FrozenBeliefBundle, systems_from_replay_rows
from adaptive_v11_active.reward import map_k, predicted_set
from adaptive_v11_active.io import read_jsonl, write_json
from v11.role_universe import TRAIN_ROLES


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


def main():
    ap = argparse.ArgumentParser(description="Fit action-target-free frozen belief model for Adaptive-V11")
    ap.add_argument("--bank", default=None)
    ap.add_argument("--search-root", default="artifacts/adaptive_v1")
    ap.add_argument("--output", default="artifacts/adaptive_v11_active_v1/frozen_belief.joblib")
    ap.add_argument("--summary", default="artifacts/adaptive_v11_active_v1/frozen_belief_summary.json")
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--folds", type=int, default=4)
    args = ap.parse_args()

    bank = Path(args.bank) if args.bank else find_bank(Path(args.search_root))
    rows = read_jsonl(bank)
    systems = systems_from_replay_rows(rows, TRAIN_ROLES, split="train")
    dev_systems = systems_from_replay_rows(rows, TRAIN_ROLES, split="dev")
    if len(systems) < 12:
        raise SystemExit(f"too few train systems in {bank}: {len(systems)}")

    bundle = FrozenBeliefBundle(TRAIN_ROLES, ks=(3, 4, 5, 6))
    summary = bundle.fit(systems, seed=args.seed, folds=args.folds)
    bundle.save(args.output)

    if dev_systems:
        k_hit = 0; k_abs = 0.0; f1s = []; exact = 0
        for ds in dev_systems:
            kp, rp = bundle.predict(ds.observations)
            pk = int(map_k(kp)); truth = set(ds.truth_roles); pred = predicted_set(rp, kp)
            k_hit += int(pk == ds.k); k_abs += abs(pk - ds.k); exact += int(pred == truth)
            tp = len(pred & truth); pr = tp/max(1,len(pred)); rc = tp/max(1,len(truth))
            f1s.append(0.0 if pr+rc == 0 else 2*pr*rc/(pr+rc))
        summary["baseline_dev"] = {
            "systems": len(dev_systems),
            "k_exact": k_hit/len(dev_systems),
            "k_mae": k_abs/len(dev_systems),
            "role_set_f1": sum(f1s)/len(f1s),
            "exact_set": exact/len(dev_systems),
            "note": "held-out baseline-bank dev; still controller-development data, not final paper test",
        }
    else:
        summary["baseline_dev"] = {"systems": 0, "warning": "no dev systems detected in baseline replay bank"}
    summary = {**summary, "source_bank": str(bank), "roles": list(TRAIN_ROLES), "ks": [3,4,5,6]}
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[done] frozen belief={args.output}", flush=True)


if __name__ == "__main__":
    main()
