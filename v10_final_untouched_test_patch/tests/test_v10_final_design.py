from collections import Counter

from targets.single_router_zoo_v10_final import CANDIDATE_ROLES, FINAL_TEST, get_v10_final_specs, validate_final_design
from v8.counterfactual import build_counterfactual_pairs


def test_final_design_is_balanced_and_fresh():
    validate_final_design()
    specs = get_v10_final_specs()
    assert len(specs) == 16
    counts = Counter(r for s in specs for r in s.worker_roles)
    assert {r: counts[r] for r in CANDIDATE_ROLES} == {r: 9 for r in CANDIDATE_ROLES}


def test_exactly_nine_counterfactual_pairs_per_role():
    specs = get_v10_final_specs()
    pairs = build_counterfactual_pairs(specs, CANDIDATE_ROLES, include_sham=True)
    assert len(pairs) == 72
    counts = Counter(p.target_role for p in pairs)
    assert {r: counts[r] for r in CANDIDATE_ROLES} == {r: 9 for r in CANDIDATE_ROLES}
