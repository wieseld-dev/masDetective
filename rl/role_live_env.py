from __future__ import annotations

from typing import Dict, List, Optional

from detective.benign_gate import check_benign
from estimator.role_presence_estimator import RolePresenceEstimator
from probes.role_probe_bank import ANCHOR_PROBE
from rl.role_reward import RoleWhiteBoxReward
from targets.hierarchical_mas import HierarchicalWhiteBoxMAS


class RoleAdaptiveLiveEnv:
    """Public state = app profile + public Q/O history + black-box role beliefs + budget."""

    def __init__(
        self,
        target: HierarchicalWhiteBoxMAS,
        estimator: RolePresenceEstimator,
        profile: Dict,
        *,
        max_steps: int = 4,
        rewarder: Optional[RoleWhiteBoxReward] = None,
        use_anchor: bool = True,
    ):
        self.target = target
        self.estimator = estimator
        self.profile = profile
        self.candidate_roles = tuple(profile["candidate_roles"])
        self.max_steps = int(max_steps)
        self.rewarder = rewarder or RoleWhiteBoxReward(self.candidate_roles)
        self.use_anchor = bool(use_anchor)
        self.history: List[Dict[str, str]] = []
        self.step_count = 0

    def _observation(self) -> Dict:
        belief = self.estimator.predict_proba(self.history)
        return {
            "app_name": self.profile.get("app_name", "application"),
            "app_description": self.profile.get("description", ""),
            "candidate_roles": list(self.candidate_roles),
            "role_descriptions": dict(self.profile.get("role_descriptions", {})),
            "role_belief": belief,
            "expected_k": float(sum(belief.values())),
            "history": list(self.history),
            "step": self.step_count,
            "remaining_budget": max(self.max_steps - self.step_count, 0),
        }

    def reset(self) -> Dict:
        self.history = []
        self.step_count = 0
        self.rewarder.reset()
        if self.use_anchor:
            output = self.target.query(ANCHOR_PROBE)
            trace = self.target.privileged_last_trace()
            self.history.append({"query": ANCHOR_PROBE, "output": output})
            self.rewarder.prime(trace)
        return self._observation()

    def step(self, focus_role: str, query: str):
        if self.step_count >= self.max_steps:
            raise RuntimeError("episode budget exhausted")
        if focus_role not in self.candidate_roles:
            self.step_count += 1
            obs = self._observation()
            return obs, -1.0, self.step_count >= self.max_steps, {
                "blocked": 1.0,
                "block_reason": "invalid_focus_role",
                "r_role": 0.0,
                "r_role_raw": 0.0,
                "r_k": 0.0,
                "r_k_raw": 0.0,
            }

        gate = check_benign(query)
        if not gate.allowed:
            self.step_count += 1
            obs = self._observation()
            return obs, -1.0, self.step_count >= self.max_steps, {
                "blocked": 1.0,
                "block_reason": gate.reason,
                "r_role": 0.0,
                "r_role_raw": 0.0,
                "r_k": 0.0,
                "r_k_raw": 0.0,
            }

        prev_belief = self.estimator.predict_proba(self.history)
        output = self.target.query(query)
        trace = self.target.privileged_last_trace()
        self.history.append({"query": query, "output": output})
        self.step_count += 1
        new_belief = self.estimator.predict_proba(self.history)

        reward, diag = self.rewarder.step(
            trace,
            prev_role_belief=prev_belief,
            new_role_belief=new_belief,
            query=query,
            output=output,
        )
        done = self.step_count >= self.max_steps or diag["completed"] == 1.0
        return self._observation(), reward, done, {
            **diag,
            "blocked": 0.0,
            "block_reason": "ok",
            "k_true": trace.k_true,
            "true_roles": list(trace.true_roles),
            "selected_worker": trace.selected_worker,
            "selected_role": trace.selected_role,
            "path_nodes": trace.path_nodes,
        }
