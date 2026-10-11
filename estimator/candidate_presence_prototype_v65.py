from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from estimator.candidate_presence_contrastive_v64 import CandidateRecord


def _unit(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    aa = a / max(float(np.linalg.norm(a)), 1e-12)
    bb = b / max(float(np.linalg.norm(b)), 1e-12)
    return float(np.dot(aa, bb))


class HFMeanPoolEmbedder:
    """Sentence embedding backend using transformers only (no sentence-transformers dependency)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu",
                 max_length: int = 256, batch_size: int = 32):
        self.model_name = model_name
        self.device = device
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self._load()
        import torch
        out = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(map(str, texts[start:start + self.batch_size]))
            tok = self._tokenizer(
                batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            )
            tok = {k: v.to(self.device) for k, v in tok.items()}
            with torch.no_grad():
                h = self._model(**tok).last_hidden_state
                mask = tok["attention_mask"].unsqueeze(-1).to(h.dtype)
                pooled = (h * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled.cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)


@dataclass
class _RoleModel:
    pos_target: np.ndarray
    neg_target: np.ndarray
    pos_delta: np.ndarray
    neg_delta: np.ndarray
    calibrator: LogisticRegression
    threshold: float


class CandidatePresencePrototypeV65:
    """Candidate-by-candidate detector designed to generalize across compositions.

    Observable features at inference:
      candidate role description, neutral anchor output, targeted external output.

    It deliberately does NOT consume router decisions, selected roles, K, topology mode,
    internal traces, or other privileged information.
    """

    def __init__(self, candidate_roles: Sequence[str], role_descriptions: Mapping[str, str],
                 *, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 random_state: int = 73):
        self.candidate_roles = tuple(map(str, candidate_roles))
        self.role_descriptions = {str(k): str(v) for k, v in role_descriptions.items()}
        self.embedding_model = embedding_model
        self.random_state = int(random_state)
        self.embedder = HFMeanPoolEmbedder(embedding_model)
        self.role_models: Dict[str, _RoleModel] = {}
        self._role_emb: Dict[str, np.ndarray] = {}

    @staticmethod
    def _feature(target, anchor, delta, role_e, pt, nt, pd, nd, lt: int, la: int) -> np.ndarray:
        return np.asarray([
            _cos(target, pt),
            _cos(target, nt),
            _cos(target, pt) - _cos(target, nt),
            _cos(delta, pd),
            _cos(delta, nd),
            _cos(delta, pd) - _cos(delta, nd),
            _cos(target, role_e),
            _cos(anchor, role_e),
            _cos(target, role_e) - _cos(anchor, role_e),
            math.log1p(max(lt, 0)),
            math.log1p(max(la, 0)),
            math.log1p(abs(lt - la)),
        ], dtype=np.float64)

    def fit(self, records: Sequence[CandidateRecord]) -> "CandidatePresencePrototypeV65":
        records = list(records)
        if not records:
            raise ValueError("records must be non-empty")

        # Embed each unique external output only once.
        unique_texts = list(dict.fromkeys(
            [r.anchor_output for r in records] + [r.target_output for r in records] +
            [self.role_descriptions.get(r, r) for r in self.candidate_roles]
        ))
        emb = self.embedder.encode(unique_texts)
        E = dict(zip(unique_texts, emb))
        self._role_emb = {r: E[self.role_descriptions.get(r, r)] for r in self.candidate_roles}

        for role in self.candidate_roles:
            rr = [r for r in records if r.role == role]
            y = np.asarray([r.label for r in rr], dtype=int)
            if len(rr) < 8 or len(np.unique(y)) < 2:
                raise ValueError(f"insufficient balanced data for role={role}: n={len(rr)} classes={set(y.tolist())}")

            T = np.stack([E[r.target_output] for r in rr])
            A = np.stack([E[r.anchor_output] for r in rr])
            D = _unit(T - A)
            pos = np.where(y == 1)[0]
            neg = np.where(y == 0)[0]
            if len(pos) < 3 or len(neg) < 3:
                raise ValueError(f"need >=3 present and >=3 absent examples for {role}")

            # Full centroids for inference.
            pt = _unit(T[pos].mean(0, keepdims=True))[0]
            nt = _unit(T[neg].mean(0, keepdims=True))[0]
            pd = _unit(D[pos].mean(0, keepdims=True))[0]
            nd = _unit(D[neg].mean(0, keepdims=True))[0]

            # Leave-one-out prototype features for calibrator fitting: avoids giving each
            # training example credit for contributing to its own class centroid.
            X = []
            pos_sum, neg_sum = T[pos].sum(0), T[neg].sum(0)
            pos_dsum, neg_dsum = D[pos].sum(0), D[neg].sum(0)
            for i, r in enumerate(rr):
                if y[i] == 1:
                    pti = _unit(((pos_sum - T[i]) / (len(pos)-1))[None, :])[0]
                    nti = nt
                    pdi = _unit(((pos_dsum - D[i]) / (len(pos)-1))[None, :])[0]
                    ndi = nd
                else:
                    pti = pt
                    nti = _unit(((neg_sum - T[i]) / (len(neg)-1))[None, :])[0]
                    pdi = pd
                    ndi = _unit(((neg_dsum - D[i]) / (len(neg)-1))[None, :])[0]
                X.append(self._feature(
                    T[i], A[i], D[i], self._role_emb[role], pti, nti, pdi, ndi,
                    len(r.target_output), len(r.anchor_output)
                ))
            X = np.stack(X)
            clf = LogisticRegression(
                C=0.5, class_weight="balanced", max_iter=3000, solver="liblinear",
                random_state=self.random_state,
            ).fit(X, y)

            # Tune a role-specific threshold only on leave-one-out training scores.
            pp = clf.predict_proba(X)[:, 1]
            best_thr, best_ba = 0.5, -1.0
            for thr in np.linspace(0.25, 0.75, 51):
                pred = pp >= thr
                tpr = pred[y == 1].mean()
                tnr = (~pred[y == 0]).mean()
                ba = 0.5 * (float(tpr) + float(tnr))
                if ba > best_ba:
                    best_ba, best_thr = ba, float(thr)

            self.role_models[role] = _RoleModel(pt, nt, pd, nd, clf, best_thr)
        return self

    def predict_proba_records(self, records: Sequence[CandidateRecord]) -> np.ndarray:
        records = list(records)
        unique_texts = list(dict.fromkeys([r.anchor_output for r in records] + [r.target_output for r in records]))
        emb = self.embedder.encode(unique_texts)
        E = dict(zip(unique_texts, emb))
        out = []
        for r in records:
            m = self.role_models[r.role]
            t = E[r.target_output]
            a = E[r.anchor_output]
            d = _unit((t - a)[None, :])[0]
            x = self._feature(
                t, a, d, self._role_emb[r.role], m.pos_target, m.neg_target, m.pos_delta, m.neg_delta,
                len(r.target_output), len(r.anchor_output)
            )
            out.append(float(m.calibrator.predict_proba(x[None, :])[0, 1]))
        return np.asarray(out, dtype=np.float64)

    def threshold_for(self, role: str) -> float:
        return float(self.role_models[str(role)].threshold)

    def predict_present_records(self, records: Sequence[CandidateRecord], probs: Sequence[float] | None = None) -> np.ndarray:
        if probs is None:
            probs = self.predict_proba_records(records)
        return np.asarray([float(p) >= self.threshold_for(r.role) for r, p in zip(records, probs)], dtype=bool)

    def save(self, path: str | Path) -> None:
        # Transformer objects are lazily loaded and may be large/non-portable. Strip runtime
        # objects before serialization; they reload on first inference.
        tok, mdl = self.embedder._tokenizer, self.embedder._model
        self.embedder._tokenizer = None; self.embedder._model = None
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self, path)
        finally:
            self.embedder._tokenizer, self.embedder._model = tok, mdl

    @classmethod
    def load(cls, path: str | Path) -> "CandidatePresencePrototypeV65":
        return joblib.load(path)
