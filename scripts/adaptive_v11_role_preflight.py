from __future__ import annotations

import json
from pathlib import Path

from adaptive_v11_active.belief import FrozenBeliefBundle
from adaptive_v11_active.role_belief import role_only_predict
from adaptive_v11_active.role_reward import adaptive_role_reward
from v11.role_universe import TRAIN_ROLES
from v11_2.hashcheck import assert_exact_v11_start


def main():
    adapter = "artifacts/v11_policy"
    summary = "artifacts/v11_summary.json"
    belief_path = Path("artifacts/adaptive_v11_active_v1/frozen_belief.joblib")
    belief_summary = Path("artifacts/adaptive_v11_active_v1/frozen_belief_summary.json")
    manifest = Path("artifacts/adaptive_v11_active_v1/manifest_privileged.json")

    fp = assert_exact_v11_start(adapter, summary, strict=True)
    if not (fp.get("policy_match") and fp.get("summary_match")):
        raise RuntimeError(fp)
    print(f"[ok] exact original V11 hash={fp['policy_sha256']}", flush=True)

    if not belief_path.exists():
        raise RuntimeError(f"missing {belief_path}")
    b = FrozenBeliefBundle.load(belief_path)
    if getattr(b, "role_platt", None) is None:
        raise RuntimeError("belief is not v2/calibrated")
    kp, rp = role_only_predict(b, [])
    if any(abs(float(v) - 0.25) > 1e-9 for v in kp.values()):
        raise RuntimeError(f"role-only K prior is not uniform: {kp}")
    if set(rp) != set(TRAIN_ROLES):
        raise RuntimeError("role universe mismatch")
    print(f"[ok] frozen calibrated role belief roles={len(rp)}; K intentionally uniform", flush=True)

    if belief_summary.exists():
        s = json.loads(belief_summary.read_text())
        print(
            "[belief-dev] role_auc={} role_log_loss={} k_nll={} (K ignored in phase 1)".format(
                s.get("baseline_dev", {}).get("role_micro_auc"),
                s.get("baseline_dev", {}).get("role_log_loss"),
                s.get("baseline_dev", {}).get("k_nll"),
            ),
            flush=True,
        )

    obj = json.loads(manifest.read_text())
    tr = [x for x in obj["systems"] if x["split"] == "train"]
    dv = [x for x in obj["systems"] if x["split"] == "dev"]
    if (len(tr), len(dv)) != (24, 8):
        raise RuntimeError(f"unexpected manifest split train={len(tr)} dev={len(dv)}")
    print(f"[ok] manifest train={len(tr)} dev={len(dv)}", flush=True)

    rr = adaptive_role_reward(
        old_role_probs=rp,
        new_role_probs=rp,
        true_roles=TRAIN_ROLES[:4],
        query="Can you compare two short implementation options and identify one concrete test that would distinguish their behavior?",
        generator_valid=True,
    )
    if abs(rr.role_log_score_improvement) > 1e-12:
        raise RuntimeError("reward self-check failed")
    print("PASS ROLE-ADAPTIVE V11 PREFLIGHT — ZERO API CALLS", flush=True)


if __name__ == "__main__":
    main()
