from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set

import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier


History = Sequence[Mapping[str, str]]


def history_to_text(history: History) -> str:
    parts: List[str] = []
    total_chars = 0
    for i, row in enumerate(history):
        q = str(row.get("query", ""))
        o = str(row.get("output", ""))
        total_chars += len(q) + len(o)
        parts.append(f"[TURN {i}] [QUERY] {q} [OUTPUT] {o}")
    parts.append(f"__turns_{min(len(history), 12)} __chars_bin_{min(total_chars // 400, 30)}")
    return "\n".join(parts)


def load_profile(path: str | Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class RolePresenceEstimator:
    """Black-box-only multi-label estimator P(role present | public Q/O history)."""

    def __init__(self, candidate_roles: Sequence[str], random_state: int = 73):
        roles = tuple(dict.fromkeys(str(r) for r in candidate_roles))
        if len(roles) < 2:
            raise ValueError("candidate_roles must contain at least two roles")
        self.candidate_roles = roles
        self.random_state = int(random_state)
        self.vectorizer = HashingVectorizer(
            n_features=2**15,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            lowercase=True,
        )
        self.classifiers: Dict[str, SGDClassifier | None] = {}
        self.priors: Dict[str, float] = {r: 0.5 for r in roles}
        self._fitted = False

    def fit(self, histories: Sequence[History], true_role_sets: Sequence[Iterable[str]]) -> "RolePresenceEstimator":
        if not histories or len(histories) != len(true_role_sets):
            raise ValueError("histories and true_role_sets must be non-empty and aligned")
        X = self.vectorizer.transform(history_to_text(h) for h in histories)
        role_sets = [set(map(str, rs)) for rs in true_role_sets]
        for ri, role in enumerate(self.candidate_roles):
            y = np.asarray([1 if role in rs else 0 for rs in role_sets], dtype=int)
            self.priors[role] = float(np.mean(y))
            if len(set(y.tolist())) < 2:
                self.classifiers[role] = None
                continue
            clf = SGDClassifier(
                loss="log_loss",
                alpha=1e-5,
                penalty="l2",
                max_iter=2500,
                tol=1e-4,
                random_state=self.random_state + ri,
                class_weight="balanced",
            )
            clf.fit(X, y)
            self.classifiers[role] = clf
        self._fitted = True
        return self

    def predict_proba(self, history: History) -> Dict[str, float]:
        if not self._fitted:
            return dict(self.priors)
        X = self.vectorizer.transform([history_to_text(history)])
        out: Dict[str, float] = {}
        for role in self.candidate_roles:
            clf = self.classifiers.get(role)
            if clf is None:
                p = self.priors.get(role, 0.5)
            else:
                probs = clf.predict_proba(X)[0]
                idx = list(clf.classes_).index(1)
                p = float(probs[idx])
            out[role] = min(max(float(p), 1e-5), 1.0 - 1e-5)
        return out

    def expected_k(self, history: History) -> float:
        return float(sum(self.predict_proba(history).values()))

    def hard_present(self, history: History, threshold: float = 0.5) -> Set[str]:
        return {r for r, p in self.predict_proba(history).items() if p >= threshold}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "RolePresenceEstimator":
        obj = joblib.load(path)
        if not isinstance(obj, RolePresenceEstimator):
            raise TypeError("not a RolePresenceEstimator artifact")
        return obj
