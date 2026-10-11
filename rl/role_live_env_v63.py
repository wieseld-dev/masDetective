from __future__ import annotations

from typing import Dict, List, Optional

from detective.benign_gate import check_benign
from estimator.role_presence_estimator_v63 import RolePresenceEstimatorV63
from probes.role_probe_bank import ANCHOR_PROBE
from rl.role_reward_v63 import RoleWhiteBoxRewardV63
from targets.single_router_fanout_mas import SingleRouterFanoutMAS


class RoleAdaptiveLiveEnvV63:
    """Black-box observation; privileged trace is consumed only inside reward/logging."""

    def __init__(
        self,
        target: SingleRouterFanoutMAS,
        estimator: RolePresenceEstimatorV63,
        profile: Dict,
        *,
        max_steps: int = 4,
        rewarder: Optional[RoleWhiteBoxRewardV63] = None,
        use_anchor: bool = True,
    ):
        self.target = target
        self.estimator = estimator
        self.profile = profile
        self.candidate_roles = tuple(profile["candidate_roles"])
        self.max_steps = int(max_steps)
        self.rewarder = rewarder or RoleWhiteBoxRewardV63(self.candidate_roles)
        self.use_anchor = bool(use_anchor)
        self.history: List[Dict[str, str]] = []
        self.tested_roles: List[str] = []
        self.step_count = 0

    def _observation(self) -> Dict:
        belief = self.estimator.predict_proba(self.history)
        return {
            "app_name": self.profile.get("app_name", "application"),
            "app_description": self.profile.get("description", ""),
            "candidate_roles": list(self.candidate_roles),
            "tested_roles": list(self.tested_roles),
            "role_descriptions": dict(self.profile.get("role_descriptions", {})),
            "role_belief": belief,
            "expected_k": float(sum(belief.values())),
            "history": list(self.history),
            "step": self.step_count,
            "remaining_budget": max(self.max_steps - self.step_count, 0),
        }

    def reset(self) -> Dict:
        self.history = []
        self.tested_roles = []
        self.step_count = 0
        self.rewarder.reset()
        if self.use_anchor:
            output = self.target.query(ANCHOR_PROBE)
            trace = self.target.privileged_last_trace()
            self.history.append({
                "query": ANCHOR_PROBE,
                "output": output,
                "focus_role": "__anchor__",
            })
            self.rewarder.prime(trace)
        return self._observation()

    def step(self, focus_role: str, query: str):
        if self.step_count >= self.max_steps:
            raise RuntimeError("episode budget exhausted")
        if focus_role not in self.candidate_roles or focus_role in self.tested_roles:
            raise ValueError(f"invalid/repeated fixed role action: {focus_role}")
        gate = check_benign(query)
        if not gate.allowed:
            raise RuntimeError(f"pre-approved fixed probe unexpectedly blocked: {gate.reason}: {query}")

        prev_belief = self.estimator.predict_proba(self.history)
        output = self.target.query(query)
        trace = self.target.privileged_last_trace()
        self.history.append({"query": query, "output": output, "focus_role": focus_role})
        self.tested_roles.append(focus_role)
        self.step_count += 1
        new_belief = self.estimator.predict_proba(self.history)

        reward, diag = self.rewarder.step(
            trace,
            focus_role=focus_role,
            prev_role_belief=prev_belief,
            new_role_belief=new_belief,
            query=query,
            output=output,
        )
        done = self.step_count >= self.max_steps
        return self._observation(), reward, done, {
            **diag,
            "blocked": 0.0,
            "execution_mode": trace.execution_mode,
            "k_true": trace.k_true,
            "true_roles": list(trace.true_roles),
            "selected_workers": list(trace.selected_workers),
            "selected_roles": list(trace.selected_roles),
            "activated_role_count": len(trace.selected_roles),
        }
