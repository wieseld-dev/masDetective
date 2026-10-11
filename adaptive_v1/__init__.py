"""Adaptive continuing probing for benign black-box MAS role inference.

Design principles:
- observations are V13.5 all-role evidence vectors, not calibrated likelihood ratios;
- aggregation is learned from grouped prefix data rather than naively adding logits;
- actions are benign diagnostic probes (initially target-role actions, with optional competitor labels);
- STOP is an explicit action driven by expected value of information (VOI) vs query cost;
- truth/selected-role traces are used only in offline training/evaluation, never controller observations.
"""

from .schema import Action, EvidenceObservation, EpisodeState
from .belief import PrefixBeliefModel
from .controllers import RandomController, MaxUncertaintyController, EmpiricalVOIController

__all__ = [
    "Action",
    "EvidenceObservation",
    "EpisodeState",
    "PrefixBeliefModel",
    "RandomController",
    "MaxUncertaintyController",
    "EmpiricalVOIController",
]
