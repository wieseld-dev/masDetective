import json
from pathlib import Path

from targets.topology_zoo import get_split


def test_candidate_set_is_overcomplete_and_contains_all_training_truth():
    p = json.loads(Path("config/application_profile.json").read_text())
    candidates = set(p["candidate_roles"])
    specs = get_split("train")
    assert len(candidates) > min(s.k_true for s in specs)
    for spec in specs:
        assert set(spec.worker_roles) <= candidates
        assert len(candidates) >= spec.k_true
