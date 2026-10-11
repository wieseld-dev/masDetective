from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

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
    # Coarse black-box behavioral features encoded as tokens.
    n = len(history)
    parts.append(f"__turns_{min(n, 10)} __chars_bin_{min(total_chars // 400, 20)}")
    return "\n".join(parts)


class KBeliefEstimator:
    """
    Black-box-only K estimator.

    Features contain ONLY user queries and external MAS outputs. K labels are used
    during estimator training, never as inference features.
    """

    def __init__(self, min_k: int = 2, max_k: int = 8, random_state: int = 73):
        self.classes_ = np.arange(min_k, max_k + 1, dtype=int)
        self.vectorizer = HashingVectorizer(
            n_features=2**14,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            lowercase=True,
        )
        self.clf = SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            penalty="l2",
            max_iter=2000,
            tol=1e-4,
            random_state=random_state,
            class_weight="balanced",
        )
        self._fitted = False

    def fit(self, histories: Sequence[History], labels: Sequence[int]) -> "KBeliefEstimator":
        if len(histories) != len(labels) or not histories:
            raise ValueError("histories and labels must be non-empty and aligned")
        X = self.vectorizer.transform(history_to_text(h) for h in histories)
        y = np.asarray(labels, dtype=int)
        present = sorted(set(int(v) for v in y))
        if len(present) < 2:
            raise ValueError("estimator bootstrap needs at least two distinct K labels")
        self.clf.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, history: History) -> Dict[int, float]:
        if not self._fitted:
            # Uniform prior before bootstrap.
            p = 1.0 / len(self.classes_)
            return {int(k): p for k in self.classes_}
        X = self.vectorizer.transform([history_to_text(history)])
        raw = self.clf.predict_proba(X)[0]
        out = {int(k): 0.0 for k in self.classes_}
        for k, p in zip(self.clf.classes_, raw):
            out[int(k)] = float(p)
        s = sum(out.values())
        if s <= 0:
            p = 1.0 / len(out)
            return {k: p for k in out}
        return {k: v / s for k, v in out.items()}

    def predict(self, history: History) -> int:
        probs = self.predict_proba(history)
        return max(probs, key=probs.get)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "KBeliefEstimator":
        obj = joblib.load(path)
        if not isinstance(obj, KBeliefEstimator):
            raise TypeError("not a KBeliefEstimator artifact")
        return obj
