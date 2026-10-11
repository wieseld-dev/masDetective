from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


# These fields are forbidden in anything that the generator can read.
FORBIDDEN_POLICY_KEYS = {
    "truth_roles",
    "truth_roles_training_only",
    "true_roles",
    "true_roles_eval_only",
    "k_true",
    "k_true_workers_eval_only",
    "selected_roles",
    "selected_roles_training_only",
    "selected_workers",
    "routing",
    "route",
    "trace",
    "path_nodes",
    "path_edges",
    "activated_nodes",
    "all_nodes",
    "all_edges",
    "worker_roles",
    "system_prompt",
    "hidden_prompt",
}


def _walk_forbidden(x: object, path: str = "root") -> list[str]:
    bad: list[str] = []
    if isinstance(x, Mapping):
        for k, v in x.items():
            key = str(k)
            if key in FORBIDDEN_POLICY_KEYS:
                bad.append(f"{path}.{key}")
            bad.extend(_walk_forbidden(v, f"{path}.{key}"))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            bad.extend(_walk_forbidden(v, f"{path}[{i}]"))
    return bad


def assert_policy_safe(obj: object) -> None:
    bad = _walk_forbidden(obj)
    if bad:
        raise RuntimeError(f"policy-visible state contains privileged fields: {bad[:12]}")


@dataclass(frozen=True)
class EvidenceFrame:
    query: str
    answer: str
    v135: Mapping[str, float]
    v134: Mapping[str, float]

    def to_public_dict(self, *, max_answer_chars: int = 1600) -> dict:
        d = {
            "query": str(self.query),
            "answer": str(self.answer)[:max_answer_chars],
            "v135": {str(k): float(v) for k, v in self.v135.items()},
            "v134": {str(k): float(v) for k, v in self.v134.items()},
        }
        assert_policy_safe(d)
        return d

    @classmethod
    def from_public_dict(cls, x: Mapping) -> "EvidenceFrame":
        assert_policy_safe(x)
        return cls(
            query=str(x.get("query", "")),
            answer=str(x.get("answer", "")),
            v135={str(k): float(v) for k, v in dict(x.get("v135", {})).items()},
            v134={str(k): float(v) for k, v in dict(x.get("v134", {})).items()},
        )


@dataclass
class PublicBeliefState:
    session_id: str
    split: str
    round_index: int
    k_probs: dict[int, float]
    role_probs: dict[str, float]
    history: list[EvidenceFrame] = field(default_factory=list)

    def validate(self, roles: Sequence[str], ks: Sequence[int]) -> None:
        if set(self.role_probs) != set(map(str, roles)):
            missing = set(map(str, roles)) - set(self.role_probs)
            extra = set(self.role_probs) - set(map(str, roles))
            raise ValueError(f"role-probability universe mismatch missing={sorted(missing)} extra={sorted(extra)}")
        if set(self.k_probs) != set(map(int, ks)):
            raise ValueError(f"K posterior keys mismatch: {sorted(self.k_probs)} vs {sorted(map(int, ks))}")
        if abs(sum(self.k_probs.values()) - 1.0) > 1e-5:
            raise ValueError("K posterior does not sum to one")
        if any(not (0.0 < float(p) < 1.0) for p in self.role_probs.values()):
            raise ValueError("role probabilities must be strictly inside (0,1)")

    def to_public_dict(self) -> dict:
        d = {
            "session_id": str(self.session_id),
            "split": str(self.split),
            "round_index": int(self.round_index),
            "k_probs": {str(int(k)): float(v) for k, v in self.k_probs.items()},
            "role_probs": {str(k): float(v) for k, v in self.role_probs.items()},
            "history": [h.to_public_dict() for h in self.history],
        }
        assert_policy_safe(d)
        return d

    @classmethod
    def from_public_dict(cls, x: Mapping) -> "PublicBeliefState":
        assert_policy_safe(x)
        return cls(
            session_id=str(x["session_id"]),
            split=str(x.get("split", "train")),
            round_index=int(x.get("round_index", 0)),
            k_probs={int(k): float(v) for k, v in dict(x["k_probs"]).items()},
            role_probs={str(k): float(v) for k, v in dict(x["role_probs"]).items()},
            history=[EvidenceFrame.from_public_dict(v) for v in x.get("history", [])],
        )
