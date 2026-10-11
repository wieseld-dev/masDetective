from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from adaptive_v11_active.compat import discover_dual_scorer_cli
from adaptive_v11_active.hypotheses import top_role_set_hypotheses
from adaptive_v11_active.reward import adaptive_reward, group_relative_advantages
from adaptive_v11_active.schema import PublicBeliefState, assert_policy_safe


def main():
    ap = argparse.ArgumentParser(description="Zero-API preflight for Adaptive-V11 interfaces")
    ap.add_argument("--v11-adapter", default="artifacts/v11_policy")
    ap.add_argument("--belief", default="artifacts/adaptive_v11_active_v1/frozen_belief.joblib")
    ap.add_argument("--require-belief", action="store_true")
    args = ap.parse_args()

    from v10.policy import SeededProbePolicyV10
    from v11.role_universe import TRAIN_ROLES
    from v11_2.hashcheck import assert_exact_v11_start
    from targets.single_router_zoo_v11 import validate_v11_design
    from targets.hierarchical_mas import ROLE_PROMPTS
    from targets.single_router_fanout_mas import SingleRouterFanoutMAS, SingleRouterSystemSpec

    validate_v11_design()
    missing = sorted(set(TRAIN_ROLES) - set(ROLE_PROMPTS))
    if missing:
        raise RuntimeError(f"runtime role registry incomplete: {missing}")
    if not Path(args.v11_adapter).is_dir():
        raise RuntimeError(f"missing V11 adapter: {args.v11_adapter}")
    fp = assert_exact_v11_start(args.v11_adapter, "artifacts/v11_summary.json", strict=True)
    if not (fp.get("policy_match") and fp.get("summary_match")):
        raise RuntimeError("original V11 checkpoint hash verification failed")
    print("[ok] exact frozen original V11 checkpoint verified")
    for name in ("sample_probe", "mean_logprob", "trainable_parameters"):
        if not hasattr(SeededProbePolicyV10, name):
            raise RuntimeError(f"V11 policy interface missing {name}()")

    # Constructor-only target validation: dummy client means zero API calls.
    spec = SingleRouterSystemSpec(
        system_id="preflight", split="preflight", execution_mode="single", style="s1",
        worker_roles=tuple(TRAIN_ROLES[:3]),
    )
    SingleRouterFanoutMAS(spec, client=object())

    inp, out, _ = discover_dual_scorer_cli("adaptive_score_v135.py")
    print(f"[ok] dual scorer flags input={inp} output={out}")

    kp = {3:.25,4:.25,5:.25,6:.25}
    rp = {r:4.5/len(TRAIN_ROLES) for r in TRAIN_ROLES}
    s = PublicBeliefState("preflight", "train", 0, kp, rp, [])
    s.validate(TRAIN_ROLES, (3,4,5,6)); assert_policy_safe(s.to_public_dict())
    hyps = top_role_set_hypotheses(rp, kp, top_n=6)
    if len(hyps) != 6:
        raise RuntimeError("hypothesis enumeration failed")
    adv = group_relative_advantages([0.1,0.2,0.3,0.4])
    if abs(sum(adv)) > 1e-4:
        raise RuntimeError("group advantages are not centered")

    if args.require_belief:
        from adaptive_v11_active.belief import FrozenBeliefBundle
        b = FrozenBeliefBundle.load(args.belief)
        b.initial()
        print(f"[ok] belief={args.belief}")

    print(f"PASS: roles={len(TRAIN_ROLES)} V11 interface ok; target registry ok; scorer bridge discoverable; pure logic ok")
    print("ZERO API calls were made.")

if __name__ == "__main__": main()
