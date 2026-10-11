from __future__ import annotations

from typing import Dict, List

from rl.whitebox_reward import WhiteBoxCoverageReward
from targets.role_presence_mas import RolePresenceWhiteBoxMAS


class LiveWhiteBoxEnv:
    """
    Live training environment.

    Policy-visible observation:
        history of benign queries and external MAS outputs
        remaining budget

    White-box trace is consumed internally by the reward function only.
    """

    def __init__(
        self,
        target: RolePresenceWhiteBoxMAS,
        *,
        max_steps: int = 6,
        rewarder: WhiteBoxCoverageReward | None = None,
    ):
        self.target = target
        self.max_steps = int(max_steps)
        self.rewarder = rewarder or WhiteBoxCoverageReward()

        self.history: List[Dict[str, str]] = []
        self.step_count = 0

    def _observation(self) -> Dict:
        return {
            "history": list(self.history),
            "step": self.step_count,
            "remaining_budget": self.max_steps - self.step_count,
        }

    def reset(self) -> Dict:
        self.history = []
        self.step_count = 0
        self.rewarder.reset(self.target.system.k_true)
        return self._observation()

    def step(self, query: str):
        if self.step_count >= self.max_steps:
            raise RuntimeError("Episode budget already exhausted.")

        output = self.target.query(query)
        trace = self.target.privileged_last_trace()

        reward, reward_diag = self.rewarder.step(trace)

        self.history.append({
            "query": query,
            "output": output,
        })
        self.step_count += 1

        done = (
            self.step_count >= self.max_steps
            or reward_diag["completed"] == 1.0
        )

        # Debug/training logs only; never feed this to the policy.
        info = {
            **reward_diag,
            "k_true": trace.k_true,
            "selected_worker": trace.worker_id,
            "selected_role": trace.selected_role,
        }

        return self._observation(), reward, done, info
