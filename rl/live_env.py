from __future__ import annotations

from typing import Dict, List, Optional

from detective.benign_gate import check_benign
from estimator.k_estimator import KBeliefEstimator
from rl.reward import FullWhiteBoxReward
from targets.hierarchical_mas import HierarchicalWhiteBoxMAS


class LiveReconstructionEnv:
    """Policy-visible state = external interaction history + budget only."""

    def __init__(self, target: HierarchicalWhiteBoxMAS, estimator: KBeliefEstimator, *, max_steps: int = 5,
                 rewarder: Optional[FullWhiteBoxReward] = None):
        self.target = target
        self.estimator = estimator
        self.max_steps = int(max_steps)
        self.rewarder = rewarder or FullWhiteBoxReward()
        self.history: List[Dict[str, str]] = []
        self.step_count = 0

    def _observation(self) -> Dict:
        # k_belief remains internal to the environment/reward path. The policy prompt in V5.2
        # intentionally does not expose it.
        return {
            "history": list(self.history),
            "step": self.step_count,
            "remaining_budget": max(self.max_steps - self.step_count, 0),
            "k_belief": self.estimator.predict_proba(self.history),
        }

    def reset(self) -> Dict:
        self.history = []
        self.step_count = 0
        self.rewarder.reset()
        return self._observation()

    def step(self, query: str):
        if self.step_count >= self.max_steps:
            raise RuntimeError("episode budget exhausted")

        gate = check_benign(query)
        if not gate.allowed:
            # Hard constraint: never send the invalid action to the target MAS.
            # It still consumes one action and receives a clear negative reward so PPO learns.
            self.step_count += 1
            obs = self._observation()
            done = self.step_count >= self.max_steps
            return obs, -1.0, done, {
                "blocked": 1.0,
                "block_reason": gate.reason,
                "r_k": 0.0,
                "r_k_raw": 0.0,
                "worker_coverage": self.rewarder.seen_workers and 0.0 or 0.0,
            }

        prev_belief = self.estimator.predict_proba(self.history)
        output = self.target.query(query)
        trace = self.target.privileged_last_trace()

        self.history.append({"query": query, "output": output})
        self.step_count += 1
        new_belief = self.estimator.predict_proba(self.history)

        reward, diag = self.rewarder.step(
            trace,
            prev_k_belief=prev_belief,
            new_k_belief=new_belief,
            query=query,
            output=output,
        )
        done = self.step_count >= self.max_steps or diag["completed"] == 1.0

        info = {
            **diag,
            "blocked": 0.0,
            "block_reason": "ok",
            "k_true": trace.k_true,
            "selected_worker": trace.selected_worker,
            "selected_role": trace.selected_role,
            "path_nodes": trace.path_nodes,
        }
        return self._observation(), reward, done, info
