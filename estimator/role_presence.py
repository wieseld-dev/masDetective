from __future__ import annotations

import csv
from collections import defaultdict
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from offline.real_dataset import RealRolePresenceDataset


class RolePresenceEstimator:
    """
    Pilot-only estimator for the old five-role dataset.

    Inference sees only the probe ID (our own action) and the external MAS output.
    Ground-truth presence labels are used only during fitting.

    A prediction cache makes repeated offline replay fast: each stored output is
    featurized/classified only once per fitted estimator.
    """

    def __init__(
        self,
        dataset: RealRolePresenceDataset,
        models: Mapping[str, Pipeline],
        role_priors: Mapping[str, float],
    ):
        self.dataset = dataset
        self.models = dict(models)
        self.role_priors = dict(role_priors)
        self._prediction_cache: Dict[Tuple[str, str], float] = {}

    @classmethod
    def fit(
        cls,
        dataset: RealRolePresenceDataset,
        train_comp_ids: Iterable[str],
    ) -> "RolePresenceEstimator":
        train_comp_ids = set(train_comp_ids)
        examples = defaultdict(list)
        labels = defaultdict(list)

        # Priors over unique training compositions, not repeated style rows.
        comp_roles = {}
        for sid in dataset.system_ids:
            truth = dataset.get_truth(sid)
            if truth.comp_id in train_comp_ids:
                comp_roles[truth.comp_id] = set(truth.roles)

        if not comp_roles:
            raise ValueError("No training compositions supplied.")

        role_priors = {
            role: sum(role in roles for roles in comp_roles.values()) / len(comp_roles)
            for role in dataset.roles
        }

        with (dataset.root / "truth.csv").open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["comp_id"] not in train_comp_ids:
                    continue
                sid = row["system_id"]
                pid = row["probe_id"]
                role = row["target_role"]
                output = dataset.get_output(sid, pid).output
                examples[role].append(output)
                labels[role].append(int(row["target_present"]))

        models = {}
        for role in dataset.roles:
            y = labels[role]
            if len(set(y)) < 2:
                raise ValueError(
                    f"Training split lacks both presence classes for role={role}"
                )

            model = Pipeline([
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                        max_features=6000,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                    ),
                ),
            ])
            model.fit(examples[role], y)
            models[role] = model

        return cls(dataset=dataset, models=models, role_priors=role_priors)

    def probability_for_output(self, role: str, output: str) -> float:
        key = (role, output)
        cached = self._prediction_cache.get(key)
        if cached is not None:
            return cached
        p = float(self.models[role].predict_proba([output])[0, 1])
        self._prediction_cache[key] = p
        return p

    def role_belief(
        self,
        history: Sequence[Tuple[str, str]],
    ) -> Dict[str, float]:
        """
        history = [(probe_id, external_output), ...]
        """
        evidence = defaultdict(list)

        for probe_id, output in history:
            role = self.dataset.probe_role[probe_id]
            evidence[role].append(self.probability_for_output(role, output))

        belief = {}
        for role in self.dataset.roles:
            if evidence[role]:
                belief[role] = float(np.mean(evidence[role]))
            else:
                belief[role] = float(self.role_priors[role])
        return belief

    @staticmethod
    def k_distribution_from_role_probs(
        role_probs: Sequence[float],
    ) -> Dict[int, float]:
        dist = np.array([1.0], dtype=float)
        for p in role_probs:
            nxt = np.zeros(len(dist) + 1, dtype=float)
            nxt[:-1] += dist * (1.0 - p)
            nxt[1:] += dist * p
            dist = nxt
        return {k: float(v) for k, v in enumerate(dist)}

    def k_distribution(
        self,
        history: Sequence[Tuple[str, str]],
    ) -> Dict[int, float]:
        b = self.role_belief(history)
        return self.k_distribution_from_role_probs(
            [b[r] for r in self.dataset.roles]
        )

    def expected_k(self, history: Sequence[Tuple[str, str]]) -> float:
        return float(sum(self.role_belief(history).values()))

    def map_k(self, history: Sequence[Tuple[str, str]]) -> int:
        dist = self.k_distribution(history)
        return max(dist, key=dist.get)
