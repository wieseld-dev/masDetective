from __future__ import annotations
from typing import Dict, Sequence, Tuple
import numpy as np
from estimator.role_presence import RolePresenceEstimator
from offline.real_dataset import RealRolePresenceDataset


def evaluate_history(dataset: RealRolePresenceDataset, estimator: RolePresenceEstimator, system_id: str, history: Sequence[Tuple[str, str]]) -> Dict[str, float]:
    truth = dataset.get_truth(system_id)
    belief = estimator.role_belief(history)
    k_dist = estimator.k_distribution(history)
    role_truth = {role: float(role in truth.roles) for role in dataset.roles}
    brier = float(np.mean([(belief[r] - role_truth[r]) ** 2 for r in dataset.roles]))
    expected_k = float(sum(belief.values()))
    map_k = int(max(k_dist, key=k_dist.get))
    return {
        'k_mae': abs(expected_k - truth.k_true),
        'exact_k': float(map_k == truth.k_true),
        'p_k_true': float(k_dist.get(truth.k_true, 0.0)),
        'role_brier': brier,
        'expected_k': expected_k,
        'map_k': float(map_k),
    }
