from __future__ import annotations

"""
K-first adaptive inference for black-box MAS composition recovery.

Primary objective
-----------------
1) infer K = number of hidden agents
2) infer the active role set, constrained to be consistent with K

The module deliberately does NOT model topology.

Architecture
------------
Dual-Judge history
      |
      v
frozen-style PrefixBeliefModel features / role head
      |
      +--> out-of-fold calibrated role marginals p(role_i | history)
      |          |
      |          +--> exact Poisson-binomial P_roles(K)
      |
      +--> low-dimensional global history features
                 |
                 +--> direct cardinality head P_direct(K)

P_roles(K) and P_direct(K) are temperature-calibrated and fused with a
cross-fitted product of experts. The fused E[K] is then projected back onto the
role marginals with one shared logit shift, preserving the role ranking while
enforcing sum_i p_i = E[K].

The adaptive controller minimizes a K-heavy posterior risk and uses an empirical
posterior-predictive outcome model fitted on TRAIN systems only.
"""

import copy
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from adaptive_v1.belief import BankSystem, PrefixBeliefModel
from adaptive_v1.schema import Action, EpisodeState, EvidenceObservation

EPS = 1e-7
STOP = "STOP"


def _clip_prob(x):
    return np.clip(np.asarray(x, dtype=np.float64), EPS, 1.0 - EPS)


def _logit(x):
    p = _clip_prob(x)
    return np.log(p) - np.log1p(-p)


def _sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    s = float(ex.sum())
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(ex) / len(ex)
    return ex / s


def _normalize(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, EPS, None)
    return p / p.sum()


def binary_entropy(p: float) -> float:
    p = float(np.clip(p, EPS, 1.0 - EPS))
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))


def categorical_entropy(p: Sequence[float]) -> float:
    p = _normalize(p)
    return float(-np.sum(p * np.log(np.clip(p, EPS, 1.0))))


def poisson_binomial(probs: Sequence[float]) -> np.ndarray:
    """Exact cardinality distribution for independent Bernoulli marginals."""
    probs = np.asarray(_clip_prob(probs), dtype=np.float64)
    n = len(probs)
    dist = np.zeros(n + 1, dtype=np.float64)
    dist[0] = 1.0
    for p in probs:
        nxt = np.zeros_like(dist)
        nxt[:-1] += dist[:-1] * (1.0 - p)
        nxt[1:] += dist[:-1] * p
        dist = nxt
    return _normalize(dist)


def temperature_scale_distribution(p: Sequence[float], temperature: float) -> np.ndarray:
    p = _normalize(p)
    t = max(float(temperature), 0.05)
    return _softmax(np.log(np.clip(p, EPS, 1.0)) / t)


def fuse_k_distributions(
    p_direct: Sequence[float],
    p_roles: Sequence[float],
    *,
    alpha: float,
) -> np.ndarray:
    """Product-of-experts fusion in log-probability space."""
    pd = _normalize(p_direct)
    pr = _normalize(p_roles)
    a = float(np.clip(alpha, 0.0, 1.0))
    logp = a * np.log(np.clip(pd, EPS, 1.0)) + (1.0 - a) * np.log(np.clip(pr, EPS, 1.0))
    return _softmax(logp)


def posterior_k_point(pk: Sequence[float]) -> int:
    """MAP K; ties are broken toward the posterior mean."""
    p = _normalize(pk)
    mx = float(np.max(p))
    candidates = np.where(np.isclose(p, mx, rtol=0.0, atol=1e-12))[0]
    mean = float(np.dot(np.arange(len(p), dtype=np.float64), p))
    return int(sorted(candidates.tolist(), key=lambda k: (abs(k - mean), k))[0])


def shift_probs_to_mean(
    probs: Sequence[float],
    target_mean: float,
    *,
    iters: int = 80,
) -> np.ndarray:
    """
    Preserve role ranking while enforcing sum(q_i) = target_mean:
        q_i = sigmoid(logit(p_i) + eta)
    """
    p = np.asarray(_clip_prob(probs), dtype=np.float64)
    n = len(p)
    target = float(np.clip(target_mean, EPS, n - EPS))
    z = _logit(p)

    lo, hi = -35.0, 35.0
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        q = _sigmoid(z + mid)
        if float(q.sum()) < target:
            lo = mid
        else:
            hi = mid
    return np.asarray(_sigmoid(z + 0.5 * (lo + hi)), dtype=np.float64)


def _prefix_sizes(n: int, max_prefix: int) -> list[int]:
    cap = min(int(n), int(max_prefix))
    base = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, cap]
    return sorted({x for x in base if 1 <= x <= cap})


def iter_prefixes(
    system: BankSystem,
    *,
    permutations: int,
    max_prefix: int,
    seed: int,
) -> Iterable[list[EvidenceObservation]]:
    obs = list(system.observations)
    if not obs:
        return

    stable = sum((i + 1) * ord(c) for i, c in enumerate(system.system_id))
    rng = random.Random(seed + stable)
    orders = [list(range(len(obs)))]

    for _ in range(max(0, int(permutations) - 1)):
        idx = list(range(len(obs)))
        rng.shuffle(idx)
        orders.append(idx)

    seen = set()
    for order in orders:
        for k in _prefix_sizes(len(order), max_prefix):
            key = tuple(order[:k])
            if key in seen:
                continue
            seen.add(key)
            yield [obs[i] for i in key]


def _summary9(values: Sequence[float]) -> list[float]:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return [0.0] * 9
    s = np.sort(x)
    top1 = float(s[-1])
    top2 = float(s[-2]) if len(s) >= 2 else top1
    return [
        float(np.mean(x)),
        float(np.std(x)),
        float(np.min(x)),
        float(np.quantile(x, 0.25)),
        float(np.median(x)),
        float(np.quantile(x, 0.75)),
        float(np.max(x)),
        top1,
        float(top1 - top2),
    ]


def global_history_features(
    history: Sequence[EvidenceObservation],
    calibrated_role_probs: Sequence[float],
    roles: Sequence[str],
) -> np.ndarray:
    """
    Low-dimensional global features for the direct K head.

    The head does not consume truth. It receives only:
      - calibrated role marginals
      - global distributions of both Judge channels across the observed history
      - evidence on the actually targeted role
      - query-count / role-coverage information
    """
    p = np.asarray(_clip_prob(calibrated_role_probs), dtype=np.float64)
    n = len(p)
    s = np.sort(p)[::-1]
    ent = np.asarray([binary_entropy(float(v)) for v in p], dtype=np.float64)

    top6 = np.zeros(6, dtype=np.float64)
    top6[: min(6, n)] = s[: min(6, n)]

    role_feats = [
        float(p.sum()),
        float(p.mean()),
        float(p.std()),
        float(ent.mean()),
        float(ent.sum()),
        float(np.sum(p >= 0.30)),
        float(np.sum(p >= 0.50)),
        float(np.sum(p >= 0.70)),
        float(np.sum(p >= 0.85)),
        *map(float, top6.tolist()),
    ]

    v135_all: list[float] = []
    v134_all: list[float] = []
    v135_target: list[float] = []
    v134_target: list[float] = []

    for obs in history:
        ev135 = {str(k): float(v) for k, v in dict(obs.evidence).items()}
        ev134 = {
            str(k): float(v)
            for k, v in dict((obs.metadata or {}).get("evidence_v134", {})).items()
        }

        v135_all.extend(ev135.get(r, 0.0) for r in roles)
        v134_all.extend(ev134.get(r, 0.0) for r in roles)

        v135_target.append(float(ev135.get(obs.action.target, 0.0)))
        v134_target.append(float(ev134.get(obs.action.target, 0.0)))

    coverage = 0.0
    if roles:
        coverage = len({o.action.target for o in history}) / float(len(roles))

    global_feats = (
        _summary9(v135_all)
        + _summary9(v134_all)
        + _summary9(v135_target)
        + _summary9(v134_target)
        + [
            float(len(history) / max(1, len(roles))),
            float(coverage),
        ]
    )

    return np.asarray(role_feats + global_feats, dtype=np.float64)


def _fit_platt(records: Sequence[dict]) -> LogisticRegression:
    by_sid = Counter(r["system_id"] for r in records)
    x, y, w = [], [], []

    for rec in records:
        sid_w = 1.0 / max(1, by_sid[rec["system_id"]])
        logits = _logit(rec["raw_role_probs"])
        labels = rec["labels"]
        for z, yy in zip(logits, labels):
            x.append([float(z)])
            y.append(int(yy))
            w.append(float(sid_w / len(labels)))

    clf = LogisticRegression(C=10.0, solver="lbfgs", max_iter=2000)
    clf.fit(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.int64),
        sample_weight=np.asarray(w, dtype=np.float64),
    )
    return clf


def _apply_platt(clf: LogisticRegression, raw: Sequence[float]) -> np.ndarray:
    x = _logit(raw).reshape(-1, 1)
    return np.asarray(clf.predict_proba(x)[:, 1], dtype=np.float64)


class DirectKHead:
    """
    Small regularized multiclass head over ordered/global evidence features.
    The set of supported K values is learned from the train split.
    """

    def __init__(self, n_roles: int, *, C: float = 0.20):
        self.n_roles = int(n_roles)
        self.C = float(C)
        self.pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=self.C,
                        class_weight="balanced",
                        max_iter=3000,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        kwargs = {}
        if sample_weight is not None:
            kwargs["clf__sample_weight"] = np.asarray(sample_weight, dtype=np.float64)
        self.pipe.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64), **kwargs)
        self._fitted = True
        return self

    def predict_full(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("DirectKHead is not fitted")

        X = np.asarray(X, dtype=np.float64)
        probs = self.pipe.predict_proba(X)
        classes = [int(x) for x in self.pipe.named_steps["clf"].classes_]

        full = np.full((len(X), self.n_roles + 1), EPS, dtype=np.float64)
        for j, kval in enumerate(classes):
            if 0 <= kval <= self.n_roles:
                full[:, kval] = np.maximum(probs[:, j], EPS)

        full /= full.sum(axis=1, keepdims=True)
        return full


@dataclass
class StructuredBelief:
    role_probs: dict[str, float]
    k_probs: np.ndarray
    k_probs_direct: np.ndarray
    k_probs_roles: np.ndarray
    expected_k: float
    k_map: int
    predicted_roles: tuple[str, ...]
    k_confidence: float
    membership_margin: float
    k_head_js: float

    def as_dict(self) -> dict:
        return {
            "role_probs": dict(self.role_probs),
            "k_probs": [float(x) for x in self.k_probs],
            "k_probs_direct": [float(x) for x in self.k_probs_direct],
            "k_probs_roles": [float(x) for x in self.k_probs_roles],
            "expected_k": float(self.expected_k),
            "k_map": int(self.k_map),
            "predicted_roles": list(self.predicted_roles),
            "k_confidence": float(self.k_confidence),
            "membership_margin": float(self.membership_margin),
            "k_head_js": float(self.k_head_js),
        }


def _js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    p = _normalize(p)
    q = _normalize(q)
    m = 0.5 * (p + q)
    kl1 = float(np.sum(p * (np.log(np.clip(p, EPS, 1.0)) - np.log(np.clip(m, EPS, 1.0)))))
    kl2 = float(np.sum(q * (np.log(np.clip(q, EPS, 1.0)) - np.log(np.clip(m, EPS, 1.0)))))
    return 0.5 * (kl1 + kl2)


class KFirstBeliefModel:
    VERSION = "adaptive_kfirst_belief_v2"

    def __init__(
        self,
        roles: Sequence[str],
        *,
        base_C: float = 0.30,
        direct_C: float = 0.20,
        n_folds: int = 4,
        seed: int = 76123,
    ):
        self.roles = tuple(map(str, roles))
        self.base_C = float(base_C)
        self.direct_C = float(direct_C)
        self.n_folds = int(n_folds)
        self.seed = int(seed)

        self.base = PrefixBeliefModel(self.roles, C=self.base_C)
        self.platt: LogisticRegression | None = None
        self.direct = DirectKHead(len(self.roles), C=self.direct_C)

        self.role_prior = np.full(len(self.roles), 0.5, dtype=np.float64)
        self.k_prior = np.ones(len(self.roles) + 1, dtype=np.float64) / (len(self.roles) + 1)

        self.direct_temperature = 1.0
        self.roles_temperature = 1.0
        self.fusion_alpha = 0.5

        self._fitted = False
        self.fit_diagnostics: dict = {}

    def _raw_role_vector(
        self,
        model: PrefixBeliefModel,
        history: Sequence[EvidenceObservation],
    ) -> np.ndarray:
        pred = model.predict(history)
        return np.asarray([float(pred[r]) for r in self.roles], dtype=np.float64)

    def _direct_features(self, history, pcal) -> np.ndarray:
        return global_history_features(history, pcal, self.roles)

    def _make_structured(
        self,
        pcal: np.ndarray,
        p_direct: np.ndarray,
        p_roles: np.ndarray,
    ) -> StructuredBelief:
        pd = temperature_scale_distribution(p_direct, self.direct_temperature)
        pr = temperature_scale_distribution(p_roles, self.roles_temperature)
        pk = fuse_k_distributions(pd, pr, alpha=self.fusion_alpha)

        expected_k = float(np.dot(np.arange(len(pk), dtype=np.float64), pk))
        k_map = posterior_k_point(pk)

        projected = shift_probs_to_mean(pcal, expected_k)
        order = np.argsort(-projected)
        predicted_roles = tuple(self.roles[i] for i in order[:k_map])

        if 0 < k_map < len(projected):
            membership_margin = float(projected[order[k_map - 1]] - projected[order[k_map]])
        else:
            membership_margin = 1.0

        return StructuredBelief(
            role_probs={r: float(projected[i]) for i, r in enumerate(self.roles)},
            k_probs=pk,
            k_probs_direct=pd,
            k_probs_roles=pr,
            expected_k=expected_k,
            k_map=k_map,
            predicted_roles=predicted_roles,
            k_confidence=float(pk[k_map]),
            membership_margin=membership_margin,
            k_head_js=float(_js_divergence(pd, pr)),
        )

    def _prior_structured(self) -> StructuredBelief:
        pcal = np.asarray(self.role_prior, dtype=np.float64)
        p_roles = poisson_binomial(pcal)
        return self._make_structured(pcal, self.k_prior, p_roles)

    def predict_structured(
        self,
        history: Sequence[EvidenceObservation],
    ) -> StructuredBelief:
        if not self._fitted or self.platt is None:
            raise RuntimeError("KFirstBeliefModel must be fitted before inference")

        if not history:
            return self._prior_structured()

        raw = self._raw_role_vector(self.base, history)
        pcal = _apply_platt(self.platt, raw)

        feat = self._direct_features(history, pcal).reshape(1, -1)
        p_direct = self.direct.predict_full(feat)[0]
        p_roles = poisson_binomial(pcal)

        return self._make_structured(pcal, p_direct, p_roles)

    def predict(self, history: Sequence[EvidenceObservation]) -> dict[str, float]:
        return self.predict_structured(history).role_probs

    def _collect_base_oof_records(
        self,
        systems: Sequence[BankSystem],
        *,
        permutations_per_system: int,
        max_prefix: int,
    ) -> list[dict]:
        systems = list(systems)
        n = len(systems)
        if n < 4:
            raise RuntimeError("K-first cross-fitting needs at least 4 train systems")

        n_splits = min(self.n_folds, n)
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        idx = np.arange(n)
        records: list[dict] = []

        for fold, (tr_idx, va_idx) in enumerate(splitter.split(idx)):
            train_fold = [systems[i] for i in tr_idx]
            valid_fold = [systems[i] for i in va_idx]

            model = PrefixBeliefModel(self.roles, C=self.base_C)
            model.fit(
                train_fold,
                permutations_per_system=permutations_per_system,
                max_prefix=max_prefix,
                seed=self.seed + 101 * fold,
            )

            for sys in valid_fold:
                truth = set(sys.truth_roles)
                labels = np.asarray([1 if r in truth else 0 for r in self.roles], dtype=np.int64)

                for hist in iter_prefixes(
                    sys,
                    permutations=permutations_per_system,
                    max_prefix=max_prefix,
                    seed=self.seed + 10000 + 211 * fold,
                ):
                    records.append(
                        {
                            "system_id": sys.system_id,
                            "raw_role_probs": self._raw_role_vector(model, hist),
                            "labels": labels,
                            "true_k": int(len(truth)),
                            "history": hist,
                        }
                    )

        if not records:
            raise RuntimeError("no out-of-fold prefix records were produced")
        return records

    def _meta_crossfit(self, records: Sequence[dict]) -> list[dict]:
        """
        Second-level grouped cross-fit.

        Each held-out system gets:
          - calibrated role marginals from a Platt model that did not see it
          - direct-K probabilities from a head that did not see it
          - role-implied Poisson-binomial K probabilities

        These are the only predictions used to tune temperatures/fusion.
        """
        records = list(records)
        sids = sorted({r["system_id"] for r in records})
        n_splits = min(self.n_folds, len(sids))
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed + 333)
        sid_arr = np.asarray(sids, dtype=object)

        out: list[dict] = []

        for fold, (tr_sid_idx, va_sid_idx) in enumerate(splitter.split(sid_arr)):
            tr_sids = set(sid_arr[tr_sid_idx].tolist())
            va_sids = set(sid_arr[va_sid_idx].tolist())

            tr = [r for r in records if r["system_id"] in tr_sids]
            va = [r for r in records if r["system_id"] in va_sids]

            platt = _fit_platt(tr)
            tr_counts = Counter(r["system_id"] for r in tr)

            Xtr, ytr, wtr = [], [], []
            for rec in tr:
                pcal = _apply_platt(platt, rec["raw_role_probs"])
                Xtr.append(self._direct_features(rec["history"], pcal))
                ytr.append(int(rec["true_k"]))
                wtr.append(1.0 / max(1, tr_counts[rec["system_id"]]))

            head = DirectKHead(len(self.roles), C=self.direct_C)
            head.fit(
                np.vstack(Xtr),
                np.asarray(ytr, dtype=np.int64),
                sample_weight=np.asarray(wtr, dtype=np.float64),
            )

            for rec in va:
                pcal = _apply_platt(platt, rec["raw_role_probs"])
                feat = self._direct_features(rec["history"], pcal).reshape(1, -1)
                pd = head.predict_full(feat)[0]
                pr = poisson_binomial(pcal)
                out.append(
                    {
                        "system_id": rec["system_id"],
                        "true_k": int(rec["true_k"]),
                        "p_direct": pd,
                        "p_roles": pr,
                    }
                )

        if len(out) != len(records):
            raise RuntimeError(
                f"meta cross-fit lost records: got {len(out)} expected {len(records)}"
            )
        return out

    @staticmethod
    def _weighted_nll(
        rows: Sequence[dict],
        prob_key: str,
        *,
        temperature: float,
    ) -> float:
        counts = Counter(r["system_id"] for r in rows)
        total = 0.0
        total_w = 0.0

        for r in rows:
            p = temperature_scale_distribution(r[prob_key], temperature)
            k = int(r["true_k"])
            w = 1.0 / max(1, counts[r["system_id"]])
            total += w * (-math.log(max(float(p[k]), EPS)))
            total_w += w

        return float(total / max(total_w, EPS))

    def _tune_meta(self, meta_rows: Sequence[dict]) -> dict:
        temp_grid = [0.55, 0.70, 0.85, 1.00, 1.20, 1.45, 1.75, 2.10, 2.50]

        d_scores = {
            t: self._weighted_nll(meta_rows, "p_direct", temperature=t)
            for t in temp_grid
        }
        r_scores = {
            t: self._weighted_nll(meta_rows, "p_roles", temperature=t)
            for t in temp_grid
        }

        td = float(min(d_scores, key=d_scores.get))
        tr = float(min(r_scores, key=r_scores.get))

        counts = Counter(r["system_id"] for r in meta_rows)
        alpha_grid = [i / 20.0 for i in range(21)]
        alpha_scores = {}

        for alpha in alpha_grid:
            total = 0.0
            total_w = 0.0
            for row in meta_rows:
                pd = temperature_scale_distribution(row["p_direct"], td)
                pr = temperature_scale_distribution(row["p_roles"], tr)
                pk = fuse_k_distributions(pd, pr, alpha=alpha)
                k = int(row["true_k"])
                w = 1.0 / max(1, counts[row["system_id"]])
                total += w * (-math.log(max(float(pk[k]), EPS)))
                total_w += w
            alpha_scores[alpha] = float(total / max(total_w, EPS))

        alpha = float(min(alpha_scores, key=alpha_scores.get))

        # Cross-fitted K diagnostics at the chosen hyperparameters.
        k_exact = []
        k_abs = []
        direct_exact = []
        roles_exact = []

        for row in meta_rows:
            pd = temperature_scale_distribution(row["p_direct"], td)
            pr = temperature_scale_distribution(row["p_roles"], tr)
            pk = fuse_k_distributions(pd, pr, alpha=alpha)
            truth = int(row["true_k"])

            kf = posterior_k_point(pk)
            kd = posterior_k_point(pd)
            kr = posterior_k_point(pr)

            k_exact.append(float(kf == truth))
            k_abs.append(abs(kf - truth))
            direct_exact.append(float(kd == truth))
            roles_exact.append(float(kr == truth))

        return {
            "direct_temperature": td,
            "roles_temperature": tr,
            "fusion_alpha": alpha,
            "direct_temperature_nll": {str(k): float(v) for k, v in d_scores.items()},
            "roles_temperature_nll": {str(k): float(v) for k, v in r_scores.items()},
            "fusion_alpha_nll": {str(k): float(v) for k, v in alpha_scores.items()},
            "meta_oof_fused_k_accuracy": float(np.mean(k_exact)),
            "meta_oof_fused_k_mae": float(np.mean(k_abs)),
            "meta_oof_direct_k_accuracy": float(np.mean(direct_exact)),
            "meta_oof_roles_k_accuracy": float(np.mean(roles_exact)),
        }

    def fit(
        self,
        systems: Sequence[BankSystem],
        *,
        permutations_per_system: int = 4,
        max_prefix: int = 20,
    ) -> dict:
        systems = list(systems)
        if not systems:
            raise RuntimeError("no train systems")

        # Priors are train-only and used before the first black-box observation.
        role_counts = np.zeros(len(self.roles), dtype=np.float64)
        k_counts = np.full(len(self.roles) + 1, 0.10, dtype=np.float64)

        for sys in systems:
            truth = set(sys.truth_roles)
            role_counts += np.asarray([1 if r in truth else 0 for r in self.roles], dtype=np.float64)
            k_counts[min(len(truth), len(self.roles))] += 1.0

        self.role_prior = np.clip(role_counts / len(systems), 0.01, 0.99)
        self.k_prior = _normalize(k_counts)

        # Level 1: OOF role-head predictions by system.
        records = self._collect_base_oof_records(
            systems,
            permutations_per_system=permutations_per_system,
            max_prefix=max_prefix,
        )

        # Level 2: cross-fitted calibrator + direct-K head predictions.
        meta_rows = self._meta_crossfit(records)
        tuned = self._tune_meta(meta_rows)

        self.direct_temperature = float(tuned["direct_temperature"])
        self.roles_temperature = float(tuned["roles_temperature"])
        self.fusion_alpha = float(tuned["fusion_alpha"])

        # Final role calibrator on all OOF records.
        self.platt = _fit_platt(records)

        # Final direct-K head on all OOF records, with equal total weight per system.
        by_sid = Counter(r["system_id"] for r in records)
        X, y, w = [], [], []

        for rec in records:
            pcal = _apply_platt(self.platt, rec["raw_role_probs"])
            X.append(self._direct_features(rec["history"], pcal))
            y.append(int(rec["true_k"]))
            w.append(1.0 / max(1, by_sid[rec["system_id"]]))

        self.direct.fit(
            np.vstack(X),
            np.asarray(y, dtype=np.int64),
            sample_weight=np.asarray(w, dtype=np.float64),
        )

        # Final base role-belief model on all train systems.
        base_diag = self.base.fit(
            systems,
            permutations_per_system=permutations_per_system,
            max_prefix=max_prefix,
            seed=self.seed + 999,
        )

        self._fitted = True
        self.fit_diagnostics = {
            "version": self.VERSION,
            "n_train_systems": int(len(systems)),
            "n_base_oof_prefixes": int(len(records)),
            **tuned,
            "base_fit": base_diag,
            "warning": (
                "All tuning diagnostics are grouped/cross-fitted development diagnostics. "
                "Use the held-out adaptive dev systems for reported comparison."
            ),
        }
        return dict(self.fit_diagnostics)

    def save(self, directory: str | Path) -> None:
        if not self._fitted or self.platt is None:
            raise RuntimeError("cannot save unfitted KFirstBeliefModel")

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        self.base.save(d / "prefix_belief.joblib")
        joblib.dump(
            {
                "version": self.VERSION,
                "roles": self.roles,
                "base_C": self.base_C,
                "direct_C": self.direct_C,
                "n_folds": self.n_folds,
                "seed": self.seed,
                "platt": self.platt,
                "direct": self.direct,
                "role_prior": self.role_prior,
                "k_prior": self.k_prior,
                "direct_temperature": self.direct_temperature,
                "roles_temperature": self.roles_temperature,
                "fusion_alpha": self.fusion_alpha,
                "fit_diagnostics": self.fit_diagnostics,
            },
            d / "kfirst_v2.joblib",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "KFirstBeliefModel":
        d = Path(directory)
        meta = joblib.load(d / "kfirst_v2.joblib")

        obj = cls(
            meta["roles"],
            base_C=float(meta["base_C"]),
            direct_C=float(meta["direct_C"]),
            n_folds=int(meta["n_folds"]),
            seed=int(meta["seed"]),
        )
        obj.base = PrefixBeliefModel.load(d / "prefix_belief.joblib")
        obj.platt = meta["platt"]
        obj.direct = meta["direct"]
        obj.role_prior = np.asarray(meta["role_prior"], dtype=np.float64)
        obj.k_prior = np.asarray(meta["k_prior"], dtype=np.float64)
        obj.direct_temperature = float(meta["direct_temperature"])
        obj.roles_temperature = float(meta["roles_temperature"])
        obj.fusion_alpha = float(meta["fusion_alpha"])
        obj.fit_diagnostics = dict(meta.get("fit_diagnostics", {}))
        obj._fitted = True
        return obj


def kfirst_risk(b: StructuredBelief) -> float:
    """
    Posterior risk used only for choosing the next probe.

    Most weight is deliberately on K:
      50% entropy of fused P(K)
      15% posterior mean absolute spread of K
      15% average role entropy
      10% ambiguity around the top-K membership boundary
      10% disagreement between the direct-K and role-implied K heads
    """
    n = max(1, len(b.role_probs))
    hk = categorical_entropy(b.k_probs) / max(math.log(n + 1), EPS)

    ks = np.arange(len(b.k_probs), dtype=np.float64)
    spread = float(np.sum(np.abs(ks - b.expected_k) * b.k_probs)) / max(1.0, n / 4.0)
    spread = float(np.clip(spread, 0.0, 1.0))

    hr = float(np.mean([binary_entropy(v) for v in b.role_probs.values()])) / math.log(2.0)
    boundary = float(np.clip(1.0 - b.membership_margin, 0.0, 1.0))
    disagreement = float(np.clip(b.k_head_js / math.log(2.0), 0.0, 1.0))

    return float(
        0.50 * hk
        + 0.15 * spread
        + 0.15 * hr
        + 0.10 * boundary
        + 0.10 * disagreement
    )


class EmpiricalKOutcomeModel:
    """
    Train-only posterior-predictive model.

    For a candidate action, every observed TRAIN-system outcome is a possible future
    observation. Source outcomes are weighted by how compatible that source
    composition is with the current role and K posterior.

    Dev truth is never used here.
    """

    def __init__(
        self,
        systems: Sequence[BankSystem],
        roles: Sequence[str],
        *,
        composition_temperature: float = 4.0,
        k_weight: float = 1.5,
        seed: int = 0,
    ):
        self.roles = tuple(roles)
        self.composition_temperature = float(composition_temperature)
        self.k_weight = float(k_weight)
        self.seed = int(seed)

        self.by_action: dict[str, list[tuple[EvidenceObservation, frozenset[str]]]] = defaultdict(list)
        self.by_target: dict[str, list[tuple[EvidenceObservation, frozenset[str]]]] = defaultdict(list)

        for sys in systems:
            truth = frozenset(map(str, sys.truth_roles))
            for obs in sys.observations:
                item = (obs, truth)
                self.by_action[obs.action.key].append(item)
                self.by_target[obs.action.target].append(item)

    def _pool(self, action: Action):
        pool = self.by_action.get(action.key)
        if pool:
            return pool
        return self.by_target.get(action.target, [])

    def weighted_samples(
        self,
        action: Action,
        belief: StructuredBelief,
        *,
        n: int,
        seed: int,
    ) -> list[EvidenceObservation]:
        pool = self._pool(action)
        if not pool:
            return []

        p = {r: float(np.clip(belief.role_probs.get(r, 0.5), EPS, 1.0 - EPS)) for r in self.roles}
        logw = []

        for _obs, truth in pool:
            ll = 0.0
            for r in self.roles:
                ll += math.log(p[r] if r in truth else (1.0 - p[r]))
            ll /= max(1, len(self.roles))

            kval = min(len(truth), len(belief.k_probs) - 1)
            lk = math.log(max(float(belief.k_probs[kval]), EPS))

            logw.append(self.composition_temperature * ll + self.k_weight * lk)

        weights = _softmax(np.asarray(logw, dtype=np.float64))
        rng = np.random.default_rng(self.seed + int(seed))
        idxs = rng.choice(np.arange(len(pool)), size=int(n), replace=True, p=weights)

        out = []
        for idx in idxs:
            src = pool[int(idx)][0]
            out.append(
                EvidenceObservation(
                    action=action,
                    query=src.query,
                    evidence=dict(src.evidence),
                    system_id="hypothetical",
                    response_id=f"hyp::{action.key}",
                    method=src.method,
                    metadata=copy.deepcopy(dict(src.metadata)),
                )
            )
        return out


def action_priority(
    belief: StructuredBelief,
    action: Action,
    *,
    target_count: int,
) -> float:
    p = float(belief.role_probs.get(action.target, 0.5))
    ent = binary_entropy(p) / math.log(2.0)

    ordered = sorted(belief.role_probs.values(), reverse=True)
    k = int(np.clip(belief.k_map, 0, len(ordered)))

    if 0 < k < len(ordered):
        threshold = 0.5 * (ordered[k - 1] + ordered[k])
    else:
        threshold = 0.5

    boundary = math.exp(-abs(p - threshold) / 0.12)
    novelty = 1.0 / (1.0 + max(0, int(target_count)))

    return float((0.60 * boundary + 0.40 * ent) * novelty)


class RandomController:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def choose(self, state: EpisodeState, available: Sequence[Action], belief: StructuredBelief):
        return self.rng.choice(list(available)) if available else STOP


class MaxUncertaintyController:
    name = "max_uncertainty"

    def choose(self, state: EpisodeState, available: Sequence[Action], belief: StructuredBelief):
        if not available:
            return STOP
        return max(
            available,
            key=lambda a: binary_entropy(float(belief.role_probs.get(a.target, 0.5))),
        )


class KBoundaryController:
    name = "k_boundary"

    def choose(self, state: EpisodeState, available: Sequence[Action], belief: StructuredBelief):
        if not available:
            return STOP
        counts = state.target_counts()
        return max(
            available,
            key=lambda a: action_priority(
                belief,
                a,
                target_count=counts.get(a.target, 0),
            ),
        )


@dataclass
class KVOIDecision:
    action: Action | str
    score: float
    risk_before: float
    expected_risk_after: float
    expected_risk_reduction: float
    query_cost: float
    details: dict


class KFirstVOIController:
    name = "kfirst_voi"

    def __init__(
        self,
        belief_model: KFirstBeliefModel,
        outcome_model: EmpiricalKOutcomeModel,
        *,
        mc_samples: int = 16,
        max_candidate_actions: int = 12,
        query_cost: float = 0.0125,
        allow_stop: bool = False,
        min_queries_before_stop: int = 3,
        stop_k_confidence: float = 0.70,
        stop_membership_margin: float = 0.10,
        seed: int = 0,
    ):
        self.belief_model = belief_model
        self.outcome_model = outcome_model
        self.mc_samples = int(mc_samples)
        self.max_candidate_actions = int(max_candidate_actions)
        self.query_cost = float(query_cost)
        self.allow_stop = bool(allow_stop)
        self.min_queries_before_stop = int(min_queries_before_stop)
        self.stop_k_confidence = float(stop_k_confidence)
        self.stop_membership_margin = float(stop_membership_margin)
        self.seed = int(seed)

    def value(
        self,
        state: EpisodeState,
        action: Action,
        belief: StructuredBelief,
    ) -> KVOIDecision:
        r0 = kfirst_risk(belief)

        samples = self.outcome_model.weighted_samples(
            action,
            belief,
            n=self.mc_samples,
            seed=self.seed + 1009 * state.step + sum(ord(c) for c in action.key),
        )
        if not samples:
            return KVOIDecision(
                action, -1e9, r0, r0, 0.0, self.query_cost, {"samples": 0}
            )

        risks = []
        for obs in samples:
            future = self.belief_model.predict_structured(list(state.history) + [obs])
            risks.append(kfirst_risk(future))

        r1 = float(np.mean(risks))
        gain = max(0.0, r0 - r1)

        return KVOIDecision(
            action=action,
            score=float(gain - self.query_cost),
            risk_before=float(r0),
            expected_risk_after=float(r1),
            expected_risk_reduction=float(gain),
            query_cost=float(self.query_cost),
            details={"samples": len(samples)},
        )

    def choose_with_details(
        self,
        state: EpisodeState,
        available: Sequence[Action],
        belief: StructuredBelief,
    ) -> KVOIDecision:
        if not available:
            r = kfirst_risk(belief)
            return KVOIDecision(STOP, 0.0, r, r, 0.0, self.query_cost, {"reason": "no_actions"})

        counts = state.target_counts()
        ranked = sorted(
            available,
            key=lambda a: action_priority(
                belief,
                a,
                target_count=counts.get(a.target, 0),
            ),
            reverse=True,
        )
        shortlist = ranked[: max(1, self.max_candidate_actions)]
        vals = [self.value(state, a, belief) for a in shortlist]
        best = max(vals, key=lambda x: x.score)

        if (
            self.allow_stop
            and state.step >= self.min_queries_before_stop
            and belief.k_confidence >= self.stop_k_confidence
            and belief.membership_margin >= self.stop_membership_margin
            and best.score <= 0.0
        ):
            return KVOIDecision(
                STOP,
                0.0,
                best.risk_before,
                best.risk_before,
                0.0,
                self.query_cost,
                {
                    "reason": "confident_and_no_positive_value",
                    "k_confidence": belief.k_confidence,
                    "membership_margin": belief.membership_margin,
                    "best_candidate_score": best.score,
                },
            )

        return best

    def choose(self, state: EpisodeState, available: Sequence[Action], belief: StructuredBelief):
        return self.choose_with_details(state, available, belief).action


def set_metrics(
    belief: StructuredBelief,
    truth_roles: Sequence[str],
    all_roles: Sequence[str],
) -> dict:
    truth = set(map(str, truth_roles))
    pred = set(belief.predicted_roles)
    true_k = len(truth)

    y_true = np.asarray([1 if r in truth else 0 for r in all_roles], dtype=np.int64)
    y_pred = np.asarray([1 if r in pred else 0 for r in all_roles], dtype=np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)

    union = len(truth | pred)
    jaccard = 1.0 if union == 0 else len(truth & pred) / union

    kd = posterior_k_point(belief.k_probs_direct)
    kr = posterior_k_point(belief.k_probs_roles)

    return {
        "true_k": int(true_k),
        "pred_k": int(belief.k_map),
        "pred_k_direct": int(kd),
        "pred_k_roles": int(kr),
        "expected_k": float(belief.expected_k),
        "k_exact": float(belief.k_map == true_k),
        "k_exact_direct": float(kd == true_k),
        "k_exact_roles": float(kr == true_k),
        "k_abs_error": float(abs(belief.k_map - true_k)),
        "expected_k_abs_error": float(abs(belief.expected_k - true_k)),
        "set_precision": float(precision),
        "set_recall": float(recall),
        "set_f1": float(f1),
        "set_jaccard": float(jaccard),
        "exact_set": float(pred == truth),
        "k_confidence": float(belief.k_confidence),
        "membership_margin": float(belief.membership_margin),
        "k_head_js": float(belief.k_head_js),
        "role_probs": [float(belief.role_probs[r]) for r in all_roles],
        "role_truth": y_true.tolist(),
    }
