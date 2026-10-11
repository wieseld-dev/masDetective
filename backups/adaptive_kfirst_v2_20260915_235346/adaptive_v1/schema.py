from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True, order=True)
class Action:
    """Controller action.

    `target` is the capability the next benign diagnostic probe should exercise.
    `competitor` is optional training-time/public-capability context for a probe generator;
    it is never sent verbatim to the black-box MAS.
    """

    target: str
    competitor: Optional[str] = None
    variant: int = 0

    @property
    def key(self) -> str:
        c = self.competitor if self.competitor else "_"
        return f"{self.target}::{c}::v{int(self.variant)}"

    def to_dict(self) -> dict:
        return {"target": self.target, "competitor": self.competitor, "variant": int(self.variant)}

    @classmethod
    def from_dict(cls, x: Mapping) -> "Action":
        return cls(str(x["target"]), None if x.get("competitor") in (None, "", "_") else str(x.get("competitor")), int(x.get("variant", 0)))


@dataclass(frozen=True)
class EvidenceObservation:
    """One black-box query plus Judge evidence.

    `evidence` is the V13.5 causal-participation channel. Optional V13.4 absolute-
    composition logits are stored in metadata["evidence_v134"]. Neither channel is
    assumed to be a calibrated probability or independent likelihood ratio.
    """

    action: Action
    query: str
    evidence: Mapping[str, float]
    system_id: str = ""
    response_id: str = ""
    method: str = "v11_2"
    metadata: Mapping = field(default_factory=dict)

    def score(self, role: str) -> float:
        return float(self.evidence.get(role, 0.0))


@dataclass
class EpisodeState:
    roles: Tuple[str, ...]
    history: list[EvidenceObservation] = field(default_factory=list)
    remaining_budget: int = 0
    belief: Dict[str, float] = field(default_factory=dict)

    @property
    def step(self) -> int:
        return len(self.history)

    def used_action_keys(self) -> set[str]:
        return {x.action.key for x in self.history}

    def target_counts(self) -> Dict[str, int]:
        out = {r: 0 for r in self.roles}
        for x in self.history:
            if x.action.target in out:
                out[x.action.target] += 1
        return out
