from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .schema import EvidenceFrame


EPS = 1e-6


def _clip01(p: float) -> float:
    return min(1.0 - EPS, max(EPS, float(p)))


def _clip_vec(x: Sequence[float]) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64), EPS, 1.0 - EPS)


def _normalize(x: Sequence[float]) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    a = np.clip(a, EPS, None)
    return a / a.sum()


def _logit(x: Sequence[float] | float) -> np.ndarray:
    p = _clip_vec(np.atleast_1d(x))
    return np.log(p) - np.log1p(-p)


def _sigmoid(x: Sequence[float] | float) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _softmax(logits: Sequence[float]) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z)
    ez = np.exp(z)
    return ez / ez.sum()


def _rank_fraction(values: Mapping[str, float], role: str) -> float:
    ordered = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
    for i, (r, _) in enumerate(ordered):
        if r == role:
            return i / max(1, len(ordered) - 1)
    return 1.0


def _series_stats(xs: Sequence[float]) -> list[float]:
    if not xs:
        return [0.0] * 9
    a = np.asarray(xs, dtype=np.float64)
    slope = float(a[-1] - a[0]) / max(1, len(a) - 1)
    q25, med, q75 = np.quantile(a, [0.25, 0.50, 0.75])
    return [
        float(a[-1]),
        float(a.mean()),
        float(a.std()),
        float(a.min()),
        float(q25),
        float(med),
        float(q75),
        float(a.max()),
        slope,
    ]


def _summary9(xs: Sequence[float]) -> list[float]:
    if not xs:
        return [0.0] * 9
    a = np.asarray(xs, dtype=np.float64)
    s = np.sort(a)
    top1 = float(s[-1])
    top2 = float(s[-2]) if len(s) > 1 else top1
    q25, med, q75 = np.quantile(a, [0.25, 0.50, 0.75])
    return [
        float(a.mean()), float(a.std()), float(a.min()), float(q25),
        float(med), float(q75), float(a.max()), top1, float(top1 - top2),
    ]


def binary_entropy(p: float) -> float:
    p = _clip01(p)
    return -(p * math.log(p) + (1.0 - p) * math.log1p(-p))


def role_history_features(history: Sequence[EvidenceFrame], role: str, roles: Sequence[str]) -> np.ndarray:
    """Target-free role features: only visible-query history + frozen Judge outputs."""
    v135 = [float(h.v135.get(role, 0.0)) for h in history]
    v134 = [float(h.v134.get(role, 0.0)) for h in history]
    ranks135 = [_rank_fraction(h.v135, role) for h in history]
    ranks134 = [_rank_fraction(h.v134, role) for h in history]
    gaps = [a - b for a, b in zip(v134, v135)]
    avgs = [0.5 * (a + b) for a, b in zip(v134, v135)]

    x: list[float] = [math.log1p(len(history))]
    for seq in (v135, v134, ranks135, ranks134, gaps, avgs):
        x += _series_stats(seq)

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
            r135 - float(a135.max()),
            r134 - float(a134.max()),
        ]
    else:
        x += [0.0] * 6

    # A role identity intercept is allowed; it contains no hidden truth.
    x += [1.0 if role == r else 0.0 for r in roles]
    return np.asarray(x, dtype=np.float32)


def k_features(history: Sequence[EvidenceFrame], role_probs: Mapping[str, float], roles: Sequence[str]) -> np.ndarray:
    """Low-dimensional target-free global features for direct K inference."""
    p = _clip_vec([role_probs[r] for r in roles])
    s = np.sort(p)[::-1]
    top8 = [float(s[i]) if i < len(s) else 0.0 for i in range(8)]
    gaps7 = [float(s[i] - s[i + 1]) if i + 1 < len(s) else 0.0 for i in range(7)]
    ent = np.asarray([binary_entropy(float(v)) for v in p], dtype=np.float64)

    x: list[float] = [
        math.log1p(len(history)),
        float(p.sum()), float(p.mean()), float(p.std()),
        float(ent.mean()), float(ent.sum()),
        float(np.sum(p >= 0.20)), float(np.sum(p >= 0.30)),
        float(np.sum(p >= 0.40)), float(np.sum(p >= 0.50)),
        float(np.sum(p >= 0.60)), float(np.sum(p >= 0.70)), float(np.sum(p >= 0.80)),
    ] + top8 + gaps7

    for channel in ("v135", "v134"):
        all_values: list[float] = []
        if history:
            maps = [getattr(h, channel) for h in history]
            last = [float(maps[-1].get(r, 0.0)) for r in roles]
            all_values = [float(m.get(r, 0.0)) for m in maps for r in roles]
            means = [float(np.mean([m.get(r, 0.0) for m in maps])) for r in roles]
            maxs = [float(np.max([m.get(r, 0.0) for m in maps])) for r in roles]
            x += _summary9(last) + _summary9(all_values) + _summary9(means) + _summary9(maxs)
        else:
            x += [0.0] * 36

    return np.asarray(x, dtype=np.float32)


def poisson_binomial(probs: Sequence[float]) -> np.ndarray:
    probs = _clip_vec(probs)
    dist = np.zeros(len(probs) + 1, dtype=np.float64)
    dist[0] = 1.0
    for p in probs:
        nxt = np.zeros_like(dist)
        nxt[:-1] += dist[:-1] * (1.0 - p)
        nxt[1:] += dist[:-1] * p
        dist = nxt
    return _normalize(dist)


def _restrict_k(full: Sequence[float], ks: Sequence[int]) -> np.ndarray:
    a = np.asarray(full, dtype=np.float64)
    return _normalize([a[k] if 0 <= int(k) < len(a) else EPS for k in ks])


def temperature_scale_distribution(p: Sequence[float], temperature: float) -> np.ndarray:
    p = _normalize(p)
    t = max(0.05, float(temperature))
    return _softmax(np.log(np.clip(p, EPS, 1.0)) / t)


def fuse_k_distributions(p_direct: Sequence[float], p_roles: Sequence[float], alpha: float) -> np.ndarray:
    pd = _normalize(p_direct)
    pr = _normalize(p_roles)
    a = float(np.clip(alpha, 0.0, 1.0))
    return _softmax(a * np.log(pd) + (1.0 - a) * np.log(pr))


def shift_probs_to_mean(probs: Sequence[float], target_mean: float) -> np.ndarray:
    p = _clip_vec(probs)
    target = float(np.clip(target_mean, EPS, len(p) - EPS))
    z = _logit(p)
    lo, hi = -35.0, 35.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        q = _sigmoid(z + mid)
        if float(q.sum()) < target:
            lo = mid
        else:
            hi = mid
    return _sigmoid(z + 0.5 * (lo + hi))


@dataclass
class _System:
    system_id: str
    truth_roles: tuple[str, ...]
    observations: list[EvidenceFrame]

    @property
    def k(self) -> int:
        return len(self.truth_roles)


class FrozenBeliefBundle:
    VERSION = "adaptive_v11_frozen_belief_v2"

    def __init__(self, roles: Sequence[str], ks: Sequence[int] = (3, 4, 5, 6), *, role_C: float = 0.25, k_C: float = 0.20):
        self.roles = tuple(map(str, roles))
        self.ks = tuple(map(int, ks))
        self.role_C = float(role_C)
        self.k_C = float(k_C)
        self.role_model = self._new_role_model()
        self.role_platt: LogisticRegression | None = None
        self.k_model = self._new_k_model()
        self.direct_temperature = 1.0
        self.roles_temperature = 1.0
        self.fusion_alpha = 0.5
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
        # Locked design prior: K is uniform over 3..6; role identities are symmetric.
        kp = {k: 1.0 / len(self.ks) for k in self.ks}
        expected_k = sum(k * p for k, p in kp.items())
        rp = {r: _clip01(expected_k / len(self.roles)) for r in self.roles}
        return kp, rp

    @staticmethod
    def _prefixes(system: _System, *, seed: int, permutations: int = 4) -> list[list[EvidenceFrame]]:
        n = len(system.observations)
        if n == 0:
            return []
        lengths = sorted({x for x in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, n) if 1 <= x <= n})
        out: list[list[EvidenceFrame]] = []
        stable = sum((i + 1) * ord(c) for i, c in enumerate(system.system_id))
        for j in range(max(1, int(permutations))):
            rng = random.Random(seed + stable + 1009 * j)
            order = list(range(n))
            if j > 0:
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
            truth = set(s.truth_roles)
            for h in prefs:
                for r in self.roles:
                    X.append(role_history_features(h, r, self.roles))
                    y.append(int(r in truth))
                    w.append(1.0 / denom)
        model = self._new_role_model()
        model.fit(np.vstack(X), np.asarray(y), clf__sample_weight=np.asarray(w) * len(systems))
        return model

    def _raw_role_vector_with(self, model, history: Sequence[EvidenceFrame]) -> np.ndarray:
        X = np.vstack([role_history_features(history, r, self.roles) for r in self.roles])
        return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)

    @staticmethod
    def _fit_platt(records: Sequence[dict]) -> LogisticRegression:
        counts = Counter(r["system_id"] for r in records)
        X: list[list[float]] = []
        y: list[int] = []
        w: list[float] = []
        for rec in records:
            sid_w = 1.0 / max(1, counts[rec["system_id"]])
            logits = _logit(rec["raw_role_probs"])
            for z, yy in zip(logits, rec["labels"]):
                X.append([float(z)])
                y.append(int(yy))
                w.append(sid_w / len(rec["labels"]))
        clf = LogisticRegression(C=10.0, solver="lbfgs", max_iter=3000)
        clf.fit(np.asarray(X), np.asarray(y), sample_weight=np.asarray(w))
        return clf

    @staticmethod
    def _apply_platt(platt: LogisticRegression, raw: Sequence[float]) -> np.ndarray:
        return np.asarray(platt.predict_proba(_logit(raw).reshape(-1, 1))[:, 1], dtype=np.float64)

    def _direct_probs_with(self, model, history: Sequence[EvidenceFrame], rp: Mapping[str, float]) -> np.ndarray:
        x = k_features(history, rp, self.roles).reshape(1, -1)
        probs = model.predict_proba(x)[0]
        classes = [int(v) for v in model.named_steps["clf"].classes_]
        out = np.full(len(self.ks), EPS, dtype=np.float64)
        k_to_i = {k: i for i, k in enumerate(self.ks)}
        for k, p in zip(classes, probs):
            if k in k_to_i:
                out[k_to_i[k]] = max(EPS, float(p))
        return _normalize(out)

    def _role_k_probs(self, pcal: Sequence[float]) -> np.ndarray:
        return _restrict_k(poisson_binomial(pcal), self.ks)

    def _collect_oof_role_records(self, systems: Sequence[_System], *, seed: int, folds: int) -> list[dict]:
        n = len(systems)
        splitter = KFold(n_splits=min(folds, n), shuffle=True, random_state=seed)
        idx = np.arange(n)
        records: list[dict] = []
        for fold, (tr, va) in enumerate(splitter.split(idx), start=1):
            model = self._fit_role([systems[i] for i in tr], seed=seed + 10000 * fold)
            for i in va:
                s = systems[i]
                labels = np.asarray([int(r in set(s.truth_roles)) for r in self.roles], dtype=np.int64)
                for h in self._prefixes(s, seed=seed + 70000 + fold):
                    records.append({
                        "system_id": s.system_id,
                        "history": h,
                        "labels": labels,
                        "true_k": s.k,
                        "raw_role_probs": self._raw_role_vector_with(model, h),
                    })
        if not records:
            raise RuntimeError("no OOF role records")
        return records

    def _meta_crossfit(self, records: Sequence[dict], *, seed: int, folds: int) -> list[dict]:
        sids = np.asarray(sorted({r["system_id"] for r in records}), dtype=object)
        splitter = KFold(n_splits=min(folds, len(sids)), shuffle=True, random_state=seed + 333)
        out: list[dict] = []
        for fold, (tr_i, va_i) in enumerate(splitter.split(sids), start=1):
            tr_sids = set(sids[tr_i].tolist())
            va_sids = set(sids[va_i].tolist())
            tr = [r for r in records if r["system_id"] in tr_sids]
            va = [r for r in records if r["system_id"] in va_sids]
            platt = self._fit_platt(tr)

            counts = Counter(r["system_id"] for r in tr)
            X: list[np.ndarray] = []
            y: list[int] = []
            w: list[float] = []
            for rec in tr:
                pcal = self._apply_platt(platt, rec["raw_role_probs"])
                rp = {r: float(pcal[j]) for j, r in enumerate(self.roles)}
                X.append(k_features(rec["history"], rp, self.roles))
                y.append(int(rec["true_k"]))
                w.append(1.0 / max(1, counts[rec["system_id"]]))
            head = self._new_k_model()
            head.fit(np.vstack(X), np.asarray(y), clf__sample_weight=np.asarray(w))

            for rec in va:
                pcal = self._apply_platt(platt, rec["raw_role_probs"])
                rp = {r: float(pcal[j]) for j, r in enumerate(self.roles)}
                out.append({
                    "system_id": rec["system_id"],
                    "true_k": int(rec["true_k"]),
                    "labels": rec["labels"],
                    "pcal": pcal,
                    "p_direct": self._direct_probs_with(head, rec["history"], rp),
                    "p_roles": self._role_k_probs(pcal),
                })
        if len(out) != len(records):
            raise RuntimeError(f"meta cross-fit lost records: {len(out)} vs {len(records)}")
        return out

    @staticmethod
    def _weighted_k_nll(rows: Sequence[dict], key: str, temperature: float, ks: Sequence[int]) -> float:
        counts = Counter(r["system_id"] for r in rows)
        k_to_i = {k: i for i, k in enumerate(ks)}
        total = 0.0
        total_w = 0.0
        for r in rows:
            p = temperature_scale_distribution(r[key], temperature)
            w = 1.0 / max(1, counts[r["system_id"]])
            total += w * (-math.log(max(float(p[k_to_i[int(r["true_k"])] ]), EPS)))
            total_w += w
        return total / max(total_w, EPS)

    def _tune_meta(self, rows: Sequence[dict]) -> dict:
        temp_grid = [0.55, 0.70, 0.85, 1.00, 1.20, 1.45, 1.75, 2.10, 2.50, 3.0]
        d_scores = {t: self._weighted_k_nll(rows, "p_direct", t, self.ks) for t in temp_grid}
        r_scores = {t: self._weighted_k_nll(rows, "p_roles", t, self.ks) for t in temp_grid}
        td = float(min(d_scores, key=d_scores.get))
        tr = float(min(r_scores, key=r_scores.get))
        counts = Counter(r["system_id"] for r in rows)
        k_to_i = {k: i for i, k in enumerate(self.ks)}
        alpha_scores: dict[float, float] = {}
        for a in [i / 20.0 for i in range(21)]:
            total = 0.0; total_w = 0.0
            for row in rows:
                pd = temperature_scale_distribution(row["p_direct"], td)
                pr = temperature_scale_distribution(row["p_roles"], tr)
                pk = fuse_k_distributions(pd, pr, a)
                w = 1.0 / max(1, counts[row["system_id"]])
                total += w * (-math.log(max(float(pk[k_to_i[int(row["true_k"])] ]), EPS)))
                total_w += w
            alpha_scores[a] = total / max(total_w, EPS)
        alpha = float(min(alpha_scores, key=alpha_scores.get))

        # OOF proper-score diagnostics at the chosen meta parameters.
        nlls: list[float] = []
        briers: list[float] = []
        hits: list[float] = []
        maes: list[float] = []
        role_ll: list[float] = []
        role_brier: list[float] = []
        for row in rows:
            pd = temperature_scale_distribution(row["p_direct"], td)
            pr = temperature_scale_distribution(row["p_roles"], tr)
            pk = fuse_k_distributions(pd, pr, alpha)
            truth_k = int(row["true_k"])
            yi = k_to_i[truth_k]
            yk = np.zeros(len(self.ks)); yk[yi] = 1.0
            nlls.append(-math.log(max(float(pk[yi]), EPS)))
            briers.append(float(np.sum((pk - yk) ** 2)))
            k_hat = int(self.ks[int(np.argmax(pk))])
            hits.append(float(k_hat == truth_k)); maes.append(abs(k_hat - truth_k))
            pcal = _clip_vec(row["pcal"])
            labels = np.asarray(row["labels"], dtype=np.float64)
            role_ll.append(float(np.mean(-(labels*np.log(pcal)+(1-labels)*np.log1p(-pcal)))))
            role_brier.append(float(np.mean((pcal-labels)**2)))
        return {
            "direct_temperature": td,
            "roles_temperature": tr,
            "fusion_alpha": alpha,
            "meta_oof_k_nll": float(np.mean(nlls)),
            "meta_oof_k_brier": float(np.mean(briers)),
            "meta_oof_k_accuracy": float(np.mean(hits)),
            "meta_oof_k_mae": float(np.mean(maes)),
            "meta_oof_role_log_loss_pre_projection": float(np.mean(role_ll)),
            "meta_oof_role_brier_pre_projection": float(np.mean(role_brier)),
            "direct_temperature_nll": {str(k): float(v) for k, v in d_scores.items()},
            "roles_temperature_nll": {str(k): float(v) for k, v in r_scores.items()},
            "fusion_alpha_nll": {str(k): float(v) for k, v in alpha_scores.items()},
        }

    def fit(self, systems: Sequence[_System], *, seed: int = 20260916, folds: int = 4) -> dict:
        systems = list(systems)
        if len(systems) < 8:
            raise ValueError("need at least eight training systems for nested cross-fitting")
        n_splits = min(int(folds), len(systems))

        records = self._collect_oof_role_records(systems, seed=seed, folds=n_splits)
        meta_rows = self._meta_crossfit(records, seed=seed, folds=n_splits)
        tuned = self._tune_meta(meta_rows)
        self.direct_temperature = float(tuned["direct_temperature"])
        self.roles_temperature = float(tuned["roles_temperature"])
        self.fusion_alpha = float(tuned["fusion_alpha"])

        self.role_platt = self._fit_platt(records)
        counts = Counter(r["system_id"] for r in records)
        X: list[np.ndarray] = []
        y: list[int] = []
        w: list[float] = []
        for rec in records:
            pcal = self._apply_platt(self.role_platt, rec["raw_role_probs"])
            rp = {r: float(pcal[j]) for j, r in enumerate(self.roles)}
            X.append(k_features(rec["history"], rp, self.roles))
            y.append(int(rec["true_k"]))
            w.append(1.0 / max(1, counts[rec["system_id"]]))
        self.k_model = self._new_k_model()
        self.k_model.fit(np.vstack(X), np.asarray(y), clf__sample_weight=np.asarray(w))
        self.role_model = self._fit_role(systems, seed=seed + 999999)
        self._fitted = True

        self.fit_summary = {
            "version": self.VERSION,
            "systems": len(systems),
            "folds": n_splits,
            "oof_prefixes": len(records),
            **tuned,
            "prior_k_nll_uniform_3_6": float(math.log(len(self.ks))),
            "prior_role_log_loss_p0225": float(-(4.5/20)*math.log(4.5/20) - (1-4.5/20)*math.log(1-4.5/20)),
            "warning": "nested grouped/cross-fitted training diagnostics only; use held-out dev for the go/no-go decision",
        }
        return dict(self.fit_summary)

    def predict(self, history: Sequence[EvidenceFrame]) -> tuple[dict[int, float], dict[str, float]]:
        if not history:
            return self.initial()
        if not self._fitted or self.role_platt is None:
            raise RuntimeError("belief bundle is not fitted")
        raw = self._raw_role_vector_with(self.role_model, history)
        pcal = self._apply_platt(self.role_platt, raw)
        rp_pre = {r: float(pcal[j]) for j, r in enumerate(self.roles)}
        pd = temperature_scale_distribution(self._direct_probs_with(self.k_model, history, rp_pre), self.direct_temperature)
        pr = temperature_scale_distribution(self._role_k_probs(pcal), self.roles_temperature)
        pk_vec = fuse_k_distributions(pd, pr, self.fusion_alpha)
        expected_k = float(sum(k * pk_vec[i] for i, k in enumerate(self.ks)))
        projected = shift_probs_to_mean(pcal, expected_k)
        kp = {k: float(pk_vec[i]) for i, k in enumerate(self.ks)}
        rp = {r: _clip01(float(projected[j])) for j, r in enumerate(self.roles)}
        return kp, rp

    def save(self, path: str | Path) -> None:
        if not self._fitted or self.role_platt is None:
            raise RuntimeError("cannot save unfitted belief")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "version": self.VERSION,
            "roles": self.roles,
            "ks": self.ks,
            "role_C": self.role_C,
            "k_C": self.k_C,
            "role_model": self.role_model,
            "role_platt": self.role_platt,
            "k_model": self.k_model,
            "direct_temperature": self.direct_temperature,
            "roles_temperature": self.roles_temperature,
            "fusion_alpha": self.fusion_alpha,
            "fitted": self._fitted,
            "fit_summary": self.fit_summary,
        }, p)

    @classmethod
    def load(cls, path: str | Path) -> "FrozenBeliefBundle":
        d = joblib.load(path)
        obj = cls(d["roles"], d["ks"], role_C=float(d.get("role_C", 0.25)), k_C=float(d.get("k_C", 0.20)))
        obj.role_model = d["role_model"]
        obj.role_platt = d.get("role_platt")
        obj.k_model = d["k_model"]
        obj.direct_temperature = float(d.get("direct_temperature", 1.0))
        obj.roles_temperature = float(d.get("roles_temperature", 1.0))
        obj.fusion_alpha = float(d.get("fusion_alpha", 0.5))
        obj._fitted = bool(d.get("fitted", True))
        obj.fit_summary = dict(d.get("fit_summary", {}))
        if obj.role_platt is None:
            raise RuntimeError(f"{path} is an old v1 belief; refit with FrozenBeliefBundle v2")
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
        by.setdefault(sid, []).append(EvidenceFrame(query=str(row.get("query", "")), answer="", v135=v135, v134=v134))
    return [_System(sid, truth[sid], obs) for sid, obs in sorted(by.items()) if seen_split.get(sid) == split]
