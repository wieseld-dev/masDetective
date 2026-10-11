from adaptive_v1.schema import Action, EvidenceObservation
from adaptive_v1.features import (
    BASE_FEATURE_NAMES,
    V134_FEATURE_NAMES,
    V135_FEATURE_NAMES,
    role_history_features,
)

ROLES = ("planning", "operations", "code", "research")


def test_dual_judge_features_are_used():
    obs = EvidenceObservation(
        action=Action("planning", "operations"),
        query="Create a practical project plan with dependencies and sequencing.",
        evidence={
            "planning": 2.1,
            "operations": 0.2,
            "code": -1.0,
            "research": -0.4,
        },
        metadata={
            "evidence_v134": {
                "planning": 4.0,
                "operations": 2.8,
                "code": -0.5,
                "research": -0.2,
            }
        },
    )

    x = role_history_features([obs], "planning", ROLES)

    assert len(x) == len(BASE_FEATURE_NAMES)
    assert len(BASE_FEATURE_NAMES) == len(V135_FEATURE_NAMES) + len(V134_FEATURE_NAMES)
    assert "v134_available_frac" in BASE_FEATURE_NAMES
    assert "dual_last_z_gap" in BASE_FEATURE_NAMES

    idx = BASE_FEATURE_NAMES.index("v134_available_frac")
    assert x[idx] == 1.0


def test_legacy_v135_only_still_works():
    obs = EvidenceObservation(
        action=Action("code"),
        query="Implement a small parser.",
        evidence={
            "planning": -0.2,
            "operations": -0.3,
            "code": 2.5,
            "research": -0.7,
        },
    )

    x = role_history_features([obs], "code", ROLES)

    assert len(x) == len(BASE_FEATURE_NAMES)

    idx = BASE_FEATURE_NAMES.index("v134_available_frac")
    assert x[idx] == 0.0
