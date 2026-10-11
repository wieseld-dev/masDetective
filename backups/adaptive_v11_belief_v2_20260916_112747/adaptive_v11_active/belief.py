from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .schema import EvidenceFrame


EPS = 1e-5


def _clip01(p: float) -> float:
    return min(1.0 - EPS, max(EPS, float(p)))


def _rank_fraction(values: Mapping[str, float], role: str) -> float:
    ordered = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
    for i, (r, _) in enumerate(ordered):
        if r == role:
            return i / max(1, len(ordered) - 1)
    return 1.0


def _series_stats(xs: Sequence[float]) -> list[float]:
    if not xs:
        return [0.0] * 7
    a = np.asarray(xs, dtype=np.float64)
    slope = float(a[-1] - a[0]) / max(1, len(a) - 1)
    return [
        float(a[-1]),
        float(a.mean()),
        float(a.std()),
        float(a.min()),
        float(a.max()),
        slope,
        float(np.median(a)),
    ]


def role_history_features(history: Sequence[EvidenceFrame], role: str, roles: Sequence[str]) -> np.ndarray:
    """Action-target-free features, so fresh adaptive queries need no synthetic target label."""
    v135 = [float(h.v135.get(role, 0.0)) for h in history]
    v134 = [float(h.v134.get(role, 0.0)) for h in history]
    ranks135 = [_rank_fraction(h.v135, role) for h in history]
    ranks134 = [_rank_fraction(h.v134, role) for h in history]
    gaps = [float(h.v134.get(role, 0.0)) - float(h.v135.get(role, 0.0)) for h in history]
    x = [math.log1p(len(history))]
    x += _series_stats(v135)
    x += _series_stats(v134)
    x += _series_stats(ranks135)
    x += _series_stats(ranks134)
    x += _series_stats(gaps)
    # Current cross-role context for this role.
    if history:
        last = history[-1]
        a135 = np.asarray([float(last.v135.get(r, 0.0)) for r in roles], dtype=np.float64)
        a134 = np.asarray([float(last.v134.get(r, 0.0)) for r in roles], dtype=np.float64)
        r135 = float(last.v135.get(role, 0.0))
        r134 = float(last.v134.get(role, 0.0))
        x += [
            r135 - float(a135.mean()),
            r134 - float(a134.mean()),
            r135 - float(np.median(a135)),
            r134 - float(np.median(a134)),
        ]
    else:
        x += [0.0] * 4
    onehot = [1.0 if role == r else 0.0 for r in roles]
    return np.asarray(x + onehot, dtype=np.float32)


def binary_entropy(p: float) -> float:
    p = _clip01(p)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def k_features(history: Sequence[EvidenceFrame], role_probs: Mapping[str, float], roles: Sequence[str]) -> np.ndarray:
    p = np.asarray([_clip01(role_probs[r]) for r in roles], dtype=np.float64)
    s = np.sort(p)[::-1]
    top = [float(s[i]) if i < len(s) else 0.0 for i in range(8)]
    gaps = [float(s[i] - s[i + 1]) if i + 1 < len(s) else 0.0 for i in range(7)]
    x: list[float] = [
        math.log1p(len(history)),
        float(p.sum()),
        float(p.mean()),
        float(p.std()),
        float(np.mean([binary_entropy(v) for v in p])),
    ] + top + gaps
    for channel in ("v135", "v134"):
        if history:
            last_map = getattr(history[-1], channel)
            last = np.asarray([float(last_map.get(r, 0.0)) for r in roles], dtype=np.float64)
            all_maps = [getattr(h, channel) for h in history]
            per_role_mean = np.asarray(
                [np.mean([float(m.get(r, 0.0)) for m in all_maps]) for r in roles], dtype=np.float64
            )
            x += [
                float(last.mean()), float(last.std()), float(last.max()), float(last.min()),
                float(np.median(last)), float(np.sort(last)[-2] if len(last) > 1 else last[0]),
                float(per_role_mean.mean()), float(per_role_mean.std()), float(per_role_mean.max()),
            ]
        else:
            x += [0.0] * 9
    return np.asarray(x, dtype=np.float32)


@dataclass
class _System:
    system_id: str
    truth_roles: tuple[str, ...]
    observations: list[EvidenceFrame]

    @property
    def k(self) -> int:
        return len(self.truth_roles)


class FrozenBeliefBundle:
    VERSION = "adaptive_v11_frozen_belief_v1"

    def __init__(self, roles: Sequence[str], ks: Sequence[int] = (3, 4, 5, 6), *, role_C: float = 0.25, k_C: float = 0.20):
        self.roles = tuple(map(str, roles))
        self.ks = tuple(map(int, ks))
        self.role_C = float(role_C)
        self.k_C = float(k_C)
        self.role_model = self._new_role_model()
        self.k_model = self._new_k_model()
        self._fitted = False
        self.fit_summary: dict = {}

    def _new_role_model(self):
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=self.role_C, class_weight="balanced", max_iter=4000, solver="lbfgs")),
        ])

    def _new_k_model(self):
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=self.k_C, class_weight="balanced", max_iter=4000, solver="lbfgs")),
        ])

    def initial(self) -> tuple[dict[int, float], dict[str, float]]:
        kp = {k: 1.0 / len(self.ks) for k in self.ks}
        expected_k = sum(k * p for k, p in kp.items())
        rp = {r: _clip01(expected_k / len(self.roles)) for r in self.roles}
        return kp, rp

    @staticmethod
    def _prefixes(system: _System, *, seed: int, permutations: int = 4) -> list[list[EvidenceFrame]]:
        n = len(system.observations)
        if n == 0:
            return []
        lengths = sorted({1, min(2, n), min(3, n), min(5, n), min(8, n), min(12, n), n})
        out: list[list[EvidenceFrame]] = []
        for j in range(permutations):
            rng = random.Random(seed + 1009 * j + sum(ord(c) for c in system.system_id))
            order = list(range(n))
            rng.shuffle(order)
            for m in lengths:
                out.append([system.observations[i] for i in order[:m]])
        return out

    def _fit_role(self, systems: Sequence[_System], *, seed: int):
        X: list[np.ndarray] = []
        y: list[int] = []
        w: list[float] = []
        for s in systems:
            prefs = self._prefixes(s, seed=seed)
            denom = max(1, len(prefs) * len(self.roles))
            for h in prefs:
                truth = set(s.truth_roles)
                for r in self.roles:
                    X.append(role_history_features(h, r, self.roles))
                    y.append(int(r in truth))
                    w.append(1.0 / denom)
        model = self._new_role_model()
        model.fit(np.vstack(X), np.asarray(y), clf__sample_weight=np.asarray(w) * len(systems))
        return model

    def _predict_roles_with(self, model, history: Sequence[EvidenceFrame]) -> dict[str, float]:
        if not history:
            return self.initial()[1]
        X = np.vstack([role_history_features(history, r, self.roles) for r in self.roles])
        p = model.predict_proba(X)[:, 1]
        return {r: _clip01(float(v)) for r, v in zip(self.roles, p)}

    def fit(self, systems: Sequence[_System], *, seed: int = 20260916, folds: int = 4) -> dict:
        systems = list(systems)
        if len(systems) < 4:
            raise ValueError("need at least four training systems")
        n_splits = min(int(folds), len(systems))
        ids = np.asarray([s.system_id for s in systems])
        groups = ids.copy()
        dummy = np.zeros(len(systems))
        gkf = GroupKFold(n_splits=n_splits)

        kX: list[np.ndarray] = []
        ky: list[int] = []
        kw: list[float] = []
        oof_role_rows = 0
        for fold, (tr, va) in enumerate(gkf.split(dummy, dummy, groups=groups), start=1):
            role_model = self._fit_role([systems[i] for i in tr], seed=seed + 10000 * fold)
            for i in va:
                s = systems[i]
                prefs = self._prefixes(s, seed=seed + 70000 + fold)
                denom = max(1, len(prefs))
                for h in prefs:
                    rp = self._predict_roles_with(role_model, h)
                    kX.append(k_features(h, rp, self.roles))
                    ky.append(s.k)
                    kw.append(1.0 / denom)
                    oof_role_rows += len(self.roles)

        self.k_model = self._new_k_model()
        self.k_model.fit(np.vstack(kX), np.asarray(ky), clf__sample_weight=np.asarray(kw) * len(systems))
        self.role_model = self._fit_role(systems, seed=seed + 999999)
        self._fitted = True

        # Training-only sanity metrics using frozen-final models; descriptive, not held-out claims.
        k_hits = 0
        role_correct = 0
        role_total = 0
        for s in systems:
            h = s.observations
            kp, rp = self.predict(h)
            k_hits += int(max(kp, key=kp.get) == s.k)
            truth = set(s.truth_roles)
            role_correct += sum(int((rp[r] >= 0.5) == (r in truth)) for r in self.roles)
            role_total += len(self.roles)
        self.fit_summary = {
            "version": self.VERSION,
            "systems": len(systems),
            "folds": n_splits,
            "oof_direct_k_prefixes": len(kX),
            "oof_role_rows": oof_role_rows,
            "descriptive_final_train_k_acc": k_hits / len(systems),
            "descriptive_final_train_role_acc_at_05": role_correct / max(1, role_total),
            "warning": "descriptive in-sample final-model checks only; use disjoint dev for performance claims",
        }
        return dict(self.fit_summary)

    def predict(self, history: Sequence[EvidenceFrame]) -> tuple[dict[int, float], dict[str, float]]:
        if not history:
            return self.initial()
        if not self._fitted:
            raise RuntimeError("belief bundle is not fitted")
        rp = self._predict_roles_with(self.role_model, history)
        x = k_features(history, rp, self.roles).reshape(1, -1)
        probs = self.k_model.predict_proba(x)[0]
        classes = [int(x) for x in self.k_model.named_steps["clf"].classes_]
        raw = {k: EPS for k in self.ks}
        for k, p in zip(classes, probs):
            if k in raw:
                raw[k] = max(EPS, float(p))
        z = sum(raw.values())
        kp = {k: float(v / z) for k, v in raw.items()}
        return kp, rp

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "version": self.VERSION,
            "roles": self.roles,
            "ks": self.ks,
            "role_C": self.role_C,
            "k_C": self.k_C,
            "role_model": self.role_model,
            "k_model": self.k_model,
            "fitted": self._fitted,
            "fit_summary": self.fit_summary,
        }, p)

    @classmethod
    def load(cls, path: str | Path) -> "FrozenBeliefBundle":
        d = joblib.load(path)
        obj = cls(d["roles"], d["ks"], role_C=float(d.get("role_C", 0.25)), k_C=float(d.get("k_C", 0.20)))
        obj.role_model = d["role_model"]
        obj.k_model = d["k_model"]
        obj._fitted = bool(d.get("fitted", True))
        obj.fit_summary = dict(d.get("fit_summary", {}))
        return obj


def systems_from_replay_rows(rows: Sequence[Mapping], roles: Sequence[str], *, split: str = "train") -> list[_System]:
    by: dict[str, list[EvidenceFrame]] = {}
    truth: dict[str, tuple[str, ...]] = {}
    seen_split: dict[str, str] = {}
    role_set = set(map(str, roles))
    for row in rows:
        sid = str(row["system_id"])
        sp = str(row.get("split", "train"))
        seen_split[sid] = sp
        tr = tuple(map(str, row.get("truth_roles", row.get("truth_roles_training_only", ()))))
        if sid in truth and set(tr) != set(truth[sid]):
            raise RuntimeError(f"inconsistent truth roles for {sid}")
        truth[sid] = tr
        v135 = {str(k): float(v) for k, v in dict(row.get("evidence", {})).items()}
        v134 = {str(k): float(v) for k, v in dict(row.get("metadata", {}).get("evidence_v134", {})).items()}
        if set(v135) != role_set or set(v134) != role_set:
            raise RuntimeError(f"{sid} replay row does not contain exactly the requested role universe in both judge channels")
        # Answers are irrelevant for the belief model and may be unavailable in replay rows.
        by.setdefault(sid, []).append(EvidenceFrame(query=str(row.get("query", "")), answer="", v135=v135, v134=v134))
    return [
        _System(sid, truth[sid], obs)
        for sid, obs in sorted(by.items())
        if seen_split.get(sid) == split
    ]
