#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


# =============================================================================
# CONFIG
# =============================================================================

SINGLES = ["A", "B", "C"]
PAIRS = ["AB", "BA", "AC", "CA", "BC", "CB"]
TRIPLES = ["ABC", "ACB", "BAC", "BCA", "CAB", "CBA"]
ALL_VARIANTS = SINGLES + PAIRS + TRIPLES

PRIMARY_SUFFIX = [
    ("ABC", "BAC", "AB", "C"),
    ("ACB", "CAB", "AC", "B"),
    ("BCA", "CBA", "BC", "A"),
]

PRIMARY_PREFIX = [
    ("ABC", "ACB", "BC", "A"),
    ("BAC", "BCA", "AC", "B"),
    ("CAB", "CBA", "AB", "C"),
]

SEED = 73


# =============================================================================
# BASICS
# =============================================================================

def safe_rho(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else 0.0


def row_l2(x, eps=1e-12):
    x = np.asarray(x, float)
    return x / np.maximum(
        np.linalg.norm(x, axis=1, keepdims=True),
        eps,
    )


def utility(a, b):
    return np.sum((a - b) ** 2, axis=1)


def score_from_K(q, K):
    return np.sum((q @ K.T) ** 2, axis=1)


def top_lift(score, actual, frac=0.20):
    score = np.asarray(score)
    actual = np.asarray(actual)

    n = len(score)
    k = max(1, math.ceil(frac * n))

    top = np.argsort(score)[::-1][:k]
    oracle = np.argsort(actual)[::-1][:k]

    mean = max(float(np.mean(actual)), 1e-12)

    return (
        float(np.mean(actual[top]) / mean),
        float(np.mean(actual[oracle]) / mean),
    )


def metric(score, actual):
    lift, oracle = top_lift(score, actual)
    return {
        "rho": safe_rho(score, actual),
        "lift20": lift,
        "oracle20": oracle,
        "n": int(len(score)),
    }


def compose(W, seq):
    d = next(iter(W.values())).shape[0]
    M = np.eye(d)

    for r in seq:
        M = W[r] @ M

    return M


def kendall_distance(a, b):
    pos = {x: i for i, x in enumerate(b)}
    arr = [pos[x] for x in a]

    inv = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            inv += int(arr[i] > arr[j])

    return inv


def quadratic_features(q):
    d = q.shape[1]
    ii, jj = np.triu_indices(d)

    z = q[:, ii] * q[:, jj]

    off = ii != jj
    z[:, off] *= math.sqrt(2.0)

    return z


def eig_score(q, K, k):
    D = K.T @ K
    vals, vecs = np.linalg.eigh(D)

    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]

    k = min(k, len(vals))

    p = q @ vecs[:, :k]
    return np.sum(
        (p ** 2) * vals[:k][None, :],
        axis=1,
    )


def spectral_stats(K):
    vals = np.linalg.eigvalsh(K.T @ K)
    vals = np.maximum(vals[::-1], 0.0)

    total = max(float(vals.sum()), 1e-12)

    return {
        "lambda1": float(vals[0]),
        "gap12": float(
            vals[0] / max(vals[1], 1e-12)
        ) if len(vals) > 1 else float("inf"),
        "top1_fraction": float(vals[:1].sum() / total),
        "top4_fraction": float(vals[:4].sum() / total),
        "effective_rank": float(
            np.exp(
                -np.sum(
                    (vals / total)
                    * np.log(
                        np.maximum(vals / total, 1e-12)
                    )
                )
            )
        ),
    }


# =============================================================================
# FLEXIBLE CACHE DISCOVERY
# =============================================================================

def text_value(v):
    if v is None:
        return ""

    if isinstance(v, str):
        return v.strip()

    if isinstance(v, dict):
        for k in (
            "final",
            "text",
            "output",
            "content",
            "response",
            "answer",
        ):
            if k in v:
                t = text_value(v[k])
                if t:
                    return t

        if "message" in v:
            return text_value(v["message"])

    return ""


def get_query(row):
    for k in (
        "query",
        "task",
        "prompt",
        "query_text",
        "input",
        "input_text",
    ):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    return ""


def get_pid(row):
    for k in (
        "probe_id",
        "id",
        "query_id",
        "task_id",
    ):
        if k in row:
            return str(row[k])

    return ""


def variants_from_row(row):
    out = {}

    # Common combined-row layouts.
    for container_name in (
        None,
        "outputs",
        "runs",
        "results",
        "systems",
    ):
        obj = (
            row
            if container_name is None
            else row.get(container_name)
        )

        if not isinstance(obj, dict):
            continue

        for v in ALL_VARIANTS:
            if v in obj:
                t = text_value(obj[v])
                if t:
                    out[v] = t

    # Common one-system-per-row layout.
    order = (
        row.get("order")
        or row.get("sequence")
        or row.get("system")
        or row.get("composition")
    )

    if isinstance(order, str):
        order = order.upper().replace(" ", "").replace("-", "")

        if order in ALL_VARIANTS:
            t = (
                text_value(row.get("final"))
                or text_value(row.get("output"))
                or text_value(row.get("response"))
            )

            if t:
                out[order] = t

    return out


def candidate_jsonls(root):
    root = Path(root)

    preferred = [
        root / "operator3_poc" / "api_runs.jsonl",
    ]

    found = []

    for p in preferred:
        if p.exists():
            found.append(p)

    for p in root.rglob("*.jsonl"):
        s = str(p).lower()

        if (
            "operator3" in s
            or "operator_3" in s
            or "op3" in s
            or "commutator" in s
        ):
            if p not in found:
                found.append(p)

    return found


def read_cache_file(path):
    rows = []

    try:
        with path.open(
            encoding="utf-8",
            errors="ignore",
        ) as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    r = json.loads(line)
                except Exception:
                    continue

                if not isinstance(r, dict):
                    continue

                q = get_query(r)
                vv = variants_from_row(r)

                if q and vv:
                    rows.append(
                        (
                            line_no,
                            get_pid(r),
                            q,
                            vv,
                        )
                    )
    except Exception:
        pass

    return rows


def discover_dataset(root):
    paths = candidate_jsonls(root)

    merged = {}
    order = []

    source_counts = {}

    for path in paths:
        rows = read_cache_file(path)

        if not rows:
            continue

        source_counts[str(path)] = len(rows)

        for _, pid, q, vv in rows:
            # Query text is the safest cross-cache join key.
            key = " ".join(q.split())

            if key not in merged:
                merged[key] = {
                    "query": q,
                    "ids": [],
                    "outputs": {},
                    "sources": [],
                }
                order.append(key)

            x = merged[key]

            if pid and pid not in x["ids"]:
                x["ids"].append(pid)

            for v, t in vv.items():
                if v not in x["outputs"]:
                    x["outputs"][v] = t

            if str(path) not in x["sources"]:
                x["sources"].append(str(path))

    entries = [merged[k] for k in order]

    complete = [
        x for x in entries
        if all(v in x["outputs"] for v in ALL_VARIANTS)
    ]

    return complete, source_counts


# =============================================================================
# OPERATOR MODEL
# =============================================================================

class ThreeOps(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

        self.delta = nn.Parameter(
            torch.zeros(3, d, d)
        )

        nn.init.normal_(
            self.delta,
            mean=0.0,
            std=0.02,
        )

    def matrices(self):
        I = torch.eye(
            self.d,
            device=self.delta.device,
        )

        return {
            "A": I + self.delta[0],
            "B": I + self.delta[1],
            "C": I + self.delta[2],
        }

    def compose(self, seq):
        W = self.matrices()

        M = torch.eye(
            self.d,
            device=self.delta.device,
        )

        for r in seq:
            M = W[r] @ M

        return M

    def predict(self, q, seq):
        return q @ self.compose(seq).T


def fit_ops(
    qtr,
    ztr,
    *,
    steps,
    restarts,
    seed,
):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tq = torch.tensor(
        qtr,
        dtype=torch.float32,
        device=device,
    )

    tz = {
        p: torch.tensor(
            ztr[p],
            dtype=torch.float32,
            device=device,
        )
        for p in PAIRS
    }

    best_global = float("inf")
    best_W = None
    restart_losses = []

    for restart in range(restarts):
        torch.manual_seed(seed + 1009 * restart)

        model = ThreeOps(
            qtr.shape[1]
        ).to(device)

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=0.003,
            weight_decay=0.001,
        )

        best = float("inf")
        best_state = None

        for step in range(steps):
            opt.zero_grad(
                set_to_none=True
            )

            losses = []

            for p in PAIRS:
                pred = model.predict(
                    tq,
                    p,
                )

                losses.append(
                    (
                        (
                            pred - tz[p]
                        )
                        ** 2
                    ).mean()
                )

            data = torch.stack(
                losses
            ).mean()

            reg = 0.002 * (
                model.delta ** 2
            ).mean()

            loss = data + reg
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            opt.step()

            val = float(
                loss.detach().cpu()
            )

            if val < best:
                best = val
                best_state = copy.deepcopy(
                    model.state_dict()
                )

        model.load_state_dict(
            best_state
        )

        with torch.no_grad():
            ww = model.matrices()
            W = {
                k: v.detach().cpu().numpy()
                for k, v in ww.items()
            }

        restart_losses.append(best)

        if best < best_global:
            best_global = best
            best_W = W

    return best_W, {
        "best_loss": best_global,
        "restart_losses": restart_losses,
    }


# =============================================================================
# REPRESENTATION
# =============================================================================

def build_representation(
    q_full,
    z_full,
    tr_idx,
    dim,
):
    # Important: TRAIN queries + TRAIN PAIR outputs only.
    # No triple output enters PCA.
    pca_train = np.vstack(
        [
            q_full[tr_idx],
            *[
                z_full[p][tr_idx]
                for p in PAIRS
            ],
        ]
    )

    d = min(
        dim,
        pca_train.shape[0] - 1,
        pca_train.shape[1],
    )

    pca = PCA(
        n_components=d,
        random_state=SEED,
    )

    pca.fit(pca_train)

    q = row_l2(
        pca.transform(q_full)
    )

    z = {
        v: row_l2(
            pca.transform(x)
        )
        for v, x in z_full.items()
    }

    return q, z, pca


# =============================================================================
# RESULT COLLECTION
# =============================================================================

class Results:
    def __init__(self):
        self.rows = []

    def add(
        self,
        *,
        family,
        contrast,
        method,
        target_space,
        score,
        actual,
        **extra,
    ):
        m = metric(
            score,
            actual,
        )

        self.rows.append({
            "family": family,
            "contrast": contrast,
            "method": method,
            "target_space": target_space,
            **m,
            **extra,
        })

    def frame(self):
        return pd.DataFrame(
            self.rows
        )


# =============================================================================
# SYNTHETIC UNIT TEST
# =============================================================================

def synthetic_test(args):
    rng = np.random.default_rng(
        91073
    )

    d = 12
    ntr = 500
    nte = 500

    def rand_W():
        return (
            np.eye(d)
            + 0.12
            * rng.normal(
                size=(d, d)
            )
            / math.sqrt(d)
        )

    Wtrue = {
        "A": rand_W(),
        "B": rand_W(),
        "C": rand_W(),
    }

    q = rng.normal(
        size=(ntr + nte, d)
    )

    q = row_l2(q)

    z = {}

    for p in PAIRS + TRIPLES:
        z[p] = q @ compose(
            Wtrue,
            p,
        ).T

    tr = np.arange(ntr)
    te = np.arange(
        ntr,
        ntr + nte,
    )

    Wfit, info = fit_ops(
        q[tr],
        {
            p: z[p][tr]
            for p in PAIRS
        },
        steps=min(args.steps, 1500),
        restarts=max(2, args.restarts),
        seed=991,
    )

    tests = []

    # Order.
    for x, y, _, _ in (
        PRIMARY_SUFFIX
        + PRIMARY_PREFIX
    ):
        K = (
            compose(Wfit, x)
            - compose(Wfit, y)
        )

        pred = score_from_K(
            q[te],
            K,
        )

        act = utility(
            z[x][te],
            z[y][te],
        )

        tests.append(
            (
                "order",
                safe_rho(pred, act),
            )
        )

    # Count insertion.
    for t in TRIPLES:
        for pos in range(3):
            pair = (
                t[:pos]
                + t[pos + 1:]
            )

            K = (
                compose(Wfit, t)
                - compose(Wfit, pair)
            )

            pred = score_from_K(
                q[te],
                K,
            )

            # Synthetic pair generated from true W.
            zp = q @ compose(
                Wtrue,
                pair,
            ).T

            act = utility(
                z[t][te],
                zp[te],
            )

            tests.append(
                (
                    f"insert_{pos}",
                    safe_rho(pred, act),
                )
            )

    return {
        "pair_fit_loss":
            info["best_loss"],
        "rho_mean":
            float(
                np.mean([
                    r for _, r in tests
                ])
            ),
        "rho_min":
            float(
                np.min([
                    r for _, r in tests
                ])
            ),
        "details":
            tests,
    }


# =============================================================================
# BASELINES
# =============================================================================

def direct_maps(qtr, ztr):
    out = {}

    for v in ALL_VARIANTS:
        rr = Ridge(
            alpha=3.0,
            fit_intercept=False,
        ).fit(
            qtr,
            ztr[v],
        )

        out[v] = rr.coef_.copy()

    return out


def scalar_ridge_score(
    qtr,
    qte,
    train_actual,
):
    rr = Ridge(
        alpha=3.0,
    ).fit(
        qtr,
        train_actual,
    )

    return rr.predict(qte)


def quadratic_ridge_score(
    qtr,
    qte,
    train_actual,
):
    ftr = quadratic_features(qtr)
    fte = quadratic_features(qte)

    rr = Ridge(
        alpha=10.0,
    ).fit(
        ftr,
        train_actual,
    )

    return rr.predict(fte)


# =============================================================================
# PRIMARY ORDER CONTRASTS
# =============================================================================

def comm_for_pair(W, pair):
    x, y = pair[0], pair[1]

    return (
        W[y] @ W[x]
        - W[x] @ W[y]
    )


def evaluate_primary_order(
    R,
    *,
    qtr,
    qte,
    ztr,
    zte,
    zte_full,
    W,
    Dmaps,
):
    aux = []

    for mode, specs in (
        ("suffix", PRIMARY_SUFFIX),
        ("prefix", PRIMARY_PREFIX),
    ):
        for x, y, pair, context in specs:
            actual = utility(
                zte[x],
                zte[y],
            )

            actual384 = utility(
                zte_full[x],
                zte_full[y],
            )

            Kfull = (
                compose(W, x)
                - compose(W, y)
            )

            Kbase = comm_for_pair(
                W,
                pair,
            )

            if mode == "suffix":
                Kcorrect = (
                    W[context] @ Kbase
                )
            else:
                Kcorrect = (
                    Kbase @ W[context]
                )

            # Sanity: formula and full composition should agree.
            exact_err = float(
                np.max(
                    np.abs(
                        Kcorrect - Kfull
                    )
                )
            )

            # Primary factorized.
            score = score_from_K(
                qte,
                Kcorrect,
            )

            R.add(
                family=f"order_{mode}",
                contrast=f"{x}_vs_{y}",
                method="factorized_correct_context",
                target_space="PCA",
                score=score,
                actual=actual,
                exact_formula_error=exact_err,
            )

            R.add(
                family=f"order_{mode}",
                contrast=f"{x}_vs_{y}",
                method="factorized_correct_context",
                target_space="FULL384",
                score=score,
                actual=actual384,
                exact_formula_error=exact_err,
            )

            # No context.
            R.add(
                family=f"order_{mode}",
                contrast=f"{x}_vs_{y}",
                method="commutator_only",
                target_space="PCA",
                score=score_from_K(
                    qte,
                    Kbase,
                ),
                actual=actual,
            )

            # Eigen truncations.
            for kk in (1, 2, 4, 8):
                R.add(
                    family=f"order_{mode}",
                    contrast=f"{x}_vs_{y}",
                    method=f"eigen_top{kk}",
                    target_space="PCA",
                    score=eig_score(
                        qte,
                        Kcorrect,
                        kk,
                    ),
                    actual=actual,
                )

            # Wrong context.
            wrongs = [
                a for a in SINGLES
                if a != context
            ]

            for wrong in wrongs:
                if mode == "suffix":
                    Kw = (
                        W[wrong] @ Kbase
                    )
                else:
                    Kw = (
                        Kbase @ W[wrong]
                    )

                R.add(
                    family=f"order_{mode}",
                    contrast=f"{x}_vs_{y}",
                    method=f"wrong_context_{wrong}",
                    target_space="PCA",
                    score=score_from_K(
                        qte,
                        Kw,
                    ),
                    actual=actual,
                )

            # Direct composition map upper control.
            Kdir = (
                Dmaps[x]
                - Dmaps[y]
            )

            R.add(
                family=f"order_{mode}",
                contrast=f"{x}_vs_{y}",
                method="direct_composition_map_UPPER",
                target_space="PCA",
                score=score_from_K(
                    qte,
                    Kdir,
                ),
                actual=actual,
            )

            # Strong scalar supervised utility baseline.
            atr = utility(
                ztr[x],
                ztr[y],
            )

            R.add(
                family=f"order_{mode}",
                contrast=f"{x}_vs_{y}",
                method="q_scalar_utility_UPPER",
                target_space="PCA",
                score=scalar_ridge_score(
                    qtr,
                    qte,
                    atr,
                ),
                actual=actual,
            )

            # Generic learned quadratic, no operator algebra.
            R.add(
                family=f"order_{mode}",
                contrast=f"{x}_vs_{y}",
                method="generic_quadratic_UPPER",
                target_space="PCA",
                score=quadratic_ridge_score(
                    qtr,
                    qte,
                    atr,
                ),
                actual=actual,
            )

            # Pair-direct transfer.
            p1 = pair
            p2 = pair[::-1]

            Kpairdir = (
                Dmaps[p1]
                - Dmaps[p2]
            )

            R.add(
                family=f"order_{mode}",
                contrast=f"{x}_vs_{y}",
                method="direct_pair_map_transfer",
                target_space="PCA",
                score=score_from_K(
                    qte,
                    Kpairdir,
                ),
                actual=actual,
            )

            # Scalar pair-utility transfer.
            pair_utr = utility(
                ztr[p1],
                ztr[p2],
            )

            R.add(
                family=f"order_{mode}",
                contrast=f"{x}_vs_{y}",
                method="pair_scalar_transfer",
                target_space="PCA",
                score=scalar_ridge_score(
                    qtr,
                    qte,
                    pair_utr,
                ),
                actual=actual,
            )

            # Pair actual -> triple actual oracle diagnostic.
            pair_actual = utility(
                zte[p1],
                zte[p2],
            )

            aux.append({
                "mode": mode,
                "contrast":
                    f"{x}_vs_{y}",
                "pair":
                    f"{p1}_vs_{p2}",
                "rho_pair_actual_to_triple_actual":
                    safe_rho(
                        pair_actual,
                        actual,
                    ),
                "rho_pair_actual_to_triple_full384":
                    safe_rho(
                        pair_actual,
                        actual384,
                    ),
                **spectral_stats(
                    Kcorrect
                ),
            })

    return pd.DataFrame(aux)


# =============================================================================
# ALL 15 TRIPLE PAIRS
# =============================================================================

def evaluate_all_triples(
    R,
    *,
    qtr,
    qte,
    ztr,
    zte,
    zte_full,
    W,
    Dmaps,
):
    rows = []

    for x, y in itertools.combinations(
        TRIPLES,
        2,
    ):
        kd = kendall_distance(
            x,
            y,
        )

        cp = x[0] == y[0]
        cs = x[-1] == y[-1]

        if cp:
            category = "common_prefix"
        elif cs:
            category = "common_suffix"
        else:
            category = "no_common_endpoint"

        actual = utility(
            zte[x],
            zte[y],
        )

        actual384 = utility(
            zte_full[x],
            zte_full[y],
        )

        K = (
            compose(W, x)
            - compose(W, y)
        )

        R.add(
            family="all15_triples",
            contrast=f"{x}_vs_{y}",
            method="factorized_full_composition",
            target_space="PCA",
            score=score_from_K(
                qte,
                K,
            ),
            actual=actual,
            category=category,
            kendall=kd,
        )

        R.add(
            family="all15_triples",
            contrast=f"{x}_vs_{y}",
            method="factorized_full_composition",
            target_space="FULL384",
            score=score_from_K(
                qte,
                K,
            ),
            actual=actual384,
            category=category,
            kendall=kd,
        )

        Kdir = (
            Dmaps[x]
            - Dmaps[y]
        )

        R.add(
            family="all15_triples",
            contrast=f"{x}_vs_{y}",
            method="direct_composition_map_UPPER",
            target_space="PCA",
            score=score_from_K(
                qte,
                Kdir,
            ),
            actual=actual,
            category=category,
            kendall=kd,
        )

        rows.append({
            "x": x,
            "y": y,
            "category": category,
            "kendall": kd,
            "oracle20":
                metric(
                    np.asarray(actual),
                    actual,
                )["oracle20"],
        })

    return pd.DataFrame(rows)


# =============================================================================
# COUNT / INSERTION
# =============================================================================

def evaluate_insertions(
    R,
    *,
    qtr,
    qte,
    ztr,
    zte,
    zte_full,
    W,
    Dmaps,
):
    details = []

    labels = {
        0: "insert_first",
        1: "insert_middle",
        2: "insert_last",
    }

    for triple in TRIPLES:
        for pos in range(3):
            pair = (
                triple[:pos]
                + triple[pos + 1:]
            )

            inserted = triple[pos]

            actual = utility(
                zte[triple],
                zte[pair],
            )

            actual384 = utility(
                zte_full[triple],
                zte_full[pair],
            )

            K = (
                compose(W, triple)
                - compose(W, pair)
            )

            fam = labels[pos]
            name = (
                f"{pair}_to_{triple}"
                f"_insert_{inserted}"
            )

            R.add(
                family=fam,
                contrast=name,
                method="factorized_insertion",
                target_space="PCA",
                score=score_from_K(
                    qte,
                    K,
                ),
                actual=actual,
            )

            R.add(
                family=fam,
                contrast=name,
                method="factorized_insertion",
                target_space="FULL384",
                score=score_from_K(
                    qte,
                    K,
                ),
                actual=actual384,
            )

            Kdir = (
                Dmaps[triple]
                - Dmaps[pair]
            )

            R.add(
                family=fam,
                contrast=name,
                method="direct_composition_map_UPPER",
                target_space="PCA",
                score=score_from_K(
                    qte,
                    Kdir,
                ),
                actual=actual,
            )

            atr = utility(
                ztr[triple],
                ztr[pair],
            )

            R.add(
                family=fam,
                contrast=name,
                method="q_scalar_utility_UPPER",
                target_space="PCA",
                score=scalar_ridge_score(
                    qtr,
                    qte,
                    atr,
                ),
                actual=actual,
            )

            R.add(
                family=fam,
                contrast=name,
                method="generic_quadratic_UPPER",
                target_space="PCA",
                score=quadratic_ridge_score(
                    qtr,
                    qte,
                    atr,
                ),
                actual=actual,
            )

            details.append({
                "triple": triple,
                "pair": pair,
                "inserted": inserted,
                "position": fam,
                **spectral_stats(K),
            })

    return pd.DataFrame(details)


# =============================================================================
# PAIR ROLE-SUBSTITUTION DIAGNOSTIC
# =============================================================================

def evaluate_substitution(
    R,
    *,
    qtr,
    qte,
    ztr,
    zte,
    W,
    Dmaps,
):
    seen = set()

    for x, y in itertools.combinations(
        PAIRS,
        2,
    ):
        same_first = x[0] == y[0]
        same_last = x[1] == y[1]

        if not (
            same_first
            or same_last
        ):
            continue

        key = tuple(
            sorted((x, y))
        )

        if key in seen:
            continue

        seen.add(key)

        mode = (
            "same_prefix"
            if same_first
            else "same_suffix"
        )

        actual = utility(
            zte[x],
            zte[y],
        )

        K = (
            compose(W, x)
            - compose(W, y)
        )

        R.add(
            family=f"pair_substitution_{mode}",
            contrast=f"{x}_vs_{y}",
            method="factorized_substitution",
            target_space="PCA",
            score=score_from_K(
                qte,
                K,
            ),
            actual=actual,
        )

        Kdir = (
            Dmaps[x]
            - Dmaps[y]
        )

        R.add(
            family=f"pair_substitution_{mode}",
            contrast=f"{x}_vs_{y}",
            method="direct_pair_map",
            target_space="PCA",
            score=score_from_K(
                qte,
                Kdir,
            ),
            actual=actual,
        )


# =============================================================================
# AGENT LABEL PERMUTATION CONTROL
# =============================================================================

def label_permutation_control(
    qte,
    zte,
    W,
):
    rows = []

    primary = (
        PRIMARY_SUFFIX
        + PRIMARY_PREFIX
    )

    for perm in itertools.permutations(
        SINGLES
    ):
        mapping = dict(
            zip(
                SINGLES,
                perm,
            )
        )

        Wp = {
            a: W[mapping[a]]
            for a in SINGLES
        }

        rhos = []
        lifts = []

        for x, y, _, _ in primary:
            K = (
                compose(Wp, x)
                - compose(Wp, y)
            )

            score = score_from_K(
                qte,
                K,
            )

            actual = utility(
                zte[x],
                zte[y],
            )

            m = metric(
                score,
                actual,
            )

            rhos.append(
                m["rho"]
            )

            lifts.append(
                m["lift20"]
            )

        rows.append({
            "mapping":
                json.dumps(mapping),
            "is_correct":
                mapping
                == {
                    "A": "A",
                    "B": "B",
                    "C": "C",
                },
            "macro_rho":
                float(
                    np.mean(rhos)
                ),
            "macro_lift":
                float(
                    np.mean(lifts)
                ),
        })

    return pd.DataFrame(rows).sort_values(
        "macro_rho",
        ascending=False,
    )


# =============================================================================
# NULLS
# =============================================================================

def null_controls(
    qte,
    zte,
    W,
    *,
    reps,
):
    rng = np.random.default_rng(
        701
    )

    primary = (
        PRIMARY_SUFFIX
        + PRIMARY_PREFIX
    )

    base = []

    for x, y, _, _ in primary:
        K = (
            compose(W, x)
            - compose(W, y)
        )

        base.append(
            (
                K,
                score_from_K(
                    qte,
                    K,
                ),
                utility(
                    zte[x],
                    zte[y],
                ),
            )
        )

    shuffle_rho = []
    shuffle_lift = []
    randomK_rho = []
    randomK_lift = []

    for _ in range(reps):
        rs = []
        ls = []
        rr = []
        lr = []

        for K, score, actual in base:
            shuffled = score[
                rng.permutation(
                    len(score)
                )
            ]

            ms = metric(
                shuffled,
                actual,
            )

            rs.append(
                ms["rho"]
            )
            ls.append(
                ms["lift20"]
            )

            Kr = rng.normal(
                size=K.shape
            )

            Kr *= (
                np.linalg.norm(
                    K,
                    ord="fro",
                )
                / max(
                    np.linalg.norm(
                        Kr,
                        ord="fro",
                    ),
                    1e-12,
                )
            )

            mr = metric(
                score_from_K(
                    qte,
                    Kr,
                ),
                actual,
            )

            rr.append(
                mr["rho"]
            )
            lr.append(
                mr["lift20"]
            )

        shuffle_rho.append(
            np.mean(rs)
        )
        shuffle_lift.append(
            np.mean(ls)
        )

        randomK_rho.append(
            np.mean(rr)
        )
        randomK_lift.append(
            np.mean(lr)
        )

    def summary(x):
        x = np.asarray(x)

        return {
            "mean":
                float(x.mean()),
            "q025":
                float(
                    np.quantile(
                        x,
                        .025,
                    )
                ),
            "q95":
                float(
                    np.quantile(
                        x,
                        .95,
                    )
                ),
            "q975":
                float(
                    np.quantile(
                        x,
                        .975,
                    )
                ),
        }

    return {
        "shuffle_rho":
            summary(shuffle_rho),
        "shuffle_lift":
            summary(shuffle_lift),
        "randomK_rho":
            summary(randomK_rho),
        "randomK_lift":
            summary(randomK_lift),
    }


# =============================================================================
# BOOTSTRAP OVER TEST PROBES
# =============================================================================

def primary_macro_bootstrap(
    qte,
    zte,
    W,
    *,
    reps,
):
    rng = np.random.default_rng(
        1701
    )

    primary = (
        PRIMARY_SUFFIX
        + PRIMARY_PREFIX
    )

    vals = []

    for _ in range(reps):
        idx = rng.integers(
            0,
            len(qte),
            size=len(qte),
        )

        rhos = []

        for x, y, _, _ in primary:
            K = (
                compose(W, x)
                - compose(W, y)
            )

            s = score_from_K(
                qte,
                K,
            )[idx]

            a = utility(
                zte[x],
                zte[y],
            )[idx]

            rhos.append(
                safe_rho(
                    s,
                    a,
                )
            )

        vals.append(
            np.mean(rhos)
        )

    vals = np.asarray(vals)

    return {
        "mean":
            float(vals.mean()),
        "p025":
            float(
                np.quantile(
                    vals,
                    .025,
                )
            ),
        "p975":
            float(
                np.quantile(
                    vals,
                    .975,
                )
            ),
        "positive_rate":
            float(
                np.mean(vals > 0)
            ),
    }


# =============================================================================
# DIMENSION SWEEP
# =============================================================================

def dimension_sweep(
    q_full,
    z_full,
    tr_idx,
    te_idx,
    args,
):
    rows = []

    for dim in args.dims:
        print(
            f"[dimension] d={dim}",
            flush=True,
        )

        q, z, _ = (
            build_representation(
                q_full,
                z_full,
                tr_idx,
                dim,
            )
        )

        W, info = fit_ops(
            q[tr_idx],
            {
                p: z[p][tr_idx]
                for p in PAIRS
            },
            steps=args.dim_steps,
            restarts=args.dim_restarts,
            seed=SEED + dim,
        )

        suffix_rho = []
        suffix_lift = []
        prefix_rho = []
        prefix_lift = []
        insert_rho = []

        for label, specs in (
            ("suffix", PRIMARY_SUFFIX),
            ("prefix", PRIMARY_PREFIX),
        ):
            for x, y, _, _ in specs:
                K = (
                    compose(W, x)
                    - compose(W, y)
                )

                m = metric(
                    score_from_K(
                        q[te_idx],
                        K,
                    ),
                    utility(
                        z[x][te_idx],
                        z[y][te_idx],
                    ),
                )

                if label == "suffix":
                    suffix_rho.append(
                        m["rho"]
                    )
                    suffix_lift.append(
                        m["lift20"]
                    )
                else:
                    prefix_rho.append(
                        m["rho"]
                    )
                    prefix_lift.append(
                        m["lift20"]
                    )

        for t in TRIPLES:
            for pos in range(3):
                p = (
                    t[:pos]
                    + t[pos + 1:]
                )

                K = (
                    compose(W, t)
                    - compose(W, p)
                )

                insert_rho.append(
                    safe_rho(
                        score_from_K(
                            q[te_idx],
                            K,
                        ),
                        utility(
                            z[t][te_idx],
                            z[p][te_idx],
                        ),
                    )
                )

        rows.append({
            "dim": dim,
            "train_loss":
                info["best_loss"],
            "suffix_rho":
                float(
                    np.mean(
                        suffix_rho
                    )
                ),
            "suffix_lift":
                float(
                    np.mean(
                        suffix_lift
                    )
                ),
            "prefix_rho":
                float(
                    np.mean(
                        prefix_rho
                    )
                ),
            "prefix_lift":
                float(
                    np.mean(
                        prefix_lift
                    )
                ),
            "all_order_rho":
                float(
                    np.mean(
                        suffix_rho
                        + prefix_rho
                    )
                ),
            "insert_rho":
                float(
                    np.mean(
                        insert_rho
                    )
                ),
        })

    return pd.DataFrame(rows)


# =============================================================================
# SEMANTIC CLUSTER STABILITY
# =============================================================================

def semantic_clusters(
    q_full,
    qte,
    zte,
    W,
    te_idx,
    *,
    k=8,
):
    k = min(
        k,
        max(2, len(q_full) // 10),
    )

    km = KMeans(
        n_clusters=k,
        random_state=SEED,
        n_init=20,
    )

    labels = km.fit_predict(
        q_full
    )

    rows = []

    for c in range(k):
        local = np.where(
            labels[te_idx] == c
        )[0]

        if len(local) < 5:
            continue

        rr = []

        for x, y, _, _ in (
            PRIMARY_SUFFIX
            + PRIMARY_PREFIX
        ):
            Kc = (
                compose(W, x)
                - compose(W, y)
            )

            rr.append(
                safe_rho(
                    score_from_K(
                        qte[local],
                        Kc,
                    ),
                    utility(
                        zte[x][local],
                        zte[y][local],
                    ),
                )
            )

        rows.append({
            "cluster": c,
            "n_test":
                int(len(local)),
            "macro_order_rho":
                float(
                    np.mean(rr)
                ),
        })

    return pd.DataFrame(rows)


# =============================================================================
# SUMMARIES
# =============================================================================

def macro_table(df):
    if df.empty:
        return df

    keys = [
        "family",
        "method",
        "target_space",
    ]

    return (
        df
        .groupby(
            keys,
            dropna=False,
        )
        .agg(
            n_contrasts=(
                "contrast",
                "count",
            ),
            macro_rho=(
                "rho",
                "mean",
            ),
            macro_lift=(
                "lift20",
                "mean",
            ),
            macro_oracle=(
                "oracle20",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "family",
                "target_space",
                "macro_rho",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
    )


def category_table(df):
    x = df[
        (
            df["family"]
            == "all15_triples"
        )
        & (
            df["method"]
            == "factorized_full_composition"
        )
        & (
            df["target_space"]
            == "PCA"
        )
    ].copy()

    if x.empty:
        return pd.DataFrame()

    out = []

    for category, g in x.groupby(
        "category"
    ):
        out.append({
            "category":
                category,
            "n":
                len(g),
            "macro_rho":
                float(
                    g["rho"].mean()
                ),
            "macro_lift":
                float(
                    g["lift20"].mean()
                ),
        })

    for kd, g in x.groupby(
        "kendall"
    ):
        out.append({
            "category":
                f"kendall_{int(kd)}",
            "n":
                len(g),
            "macro_rho":
                float(
                    g["rho"].mean()
                ),
            "macro_lift":
                float(
                    g["lift20"].mean()
                ),
        })

    return pd.DataFrame(out)


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--artifact-root",
        default="artifacts",
    )

    ap.add_argument(
        "--out",
        default=(
            "artifacts/"
            "chain_mechanism_audit_v2"
        ),
    )

    ap.add_argument(
        "--embed-model",
        default=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
    )

    ap.add_argument(
        "--dim",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--steps",
        type=int,
        default=2500,
    )

    ap.add_argument(
        "--restarts",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--null-reps",
        type=int,
        default=1500,
    )

    ap.add_argument(
        "--bootstrap-reps",
        type=int,
        default=1500,
    )

    ap.add_argument(
        "--dims",
        default="8,16,32,64",
    )

    ap.add_argument(
        "--dim-steps",
        type=int,
        default=1600,
    )

    ap.add_argument(
        "--dim-restarts",
        type=int,
        default=2,
    )

    args = ap.parse_args()

    args.dims = [
        int(x)
        for x in args.dims.split(",")
        if x.strip()
    ]

    out = Path(args.out)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 110)
    print("EXHAUSTIVE CHAIN MECHANISM AUDIT V2")
    print("=" * 110)
    print("NO API CALLS")
    print()

    # -------------------------------------------------------------------------
    # 0. Synthetic ground-truth test.
    # -------------------------------------------------------------------------

    print(
        "[0] Synthetic operator unit test...",
        flush=True,
    )

    synth = synthetic_test(
        args
    )

    (
        out / "synthetic.json"
    ).write_text(
        json.dumps(
            synth,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "synthetic mean rho =",
        round(
            synth["rho_mean"],
            4,
        ),
        "| min =",
        round(
            synth["rho_min"],
            4,
        ),
    )

    # -------------------------------------------------------------------------
    # 1. Discover raw data.
    # -------------------------------------------------------------------------

    print(
        "\n[1] Discovering operator3 caches...",
        flush=True,
    )

    entries, source_counts = (
        discover_dataset(
            args.artifact_root
        )
    )

    print(
        "complete A/B/C + pairs + triples rows:",
        len(entries),
    )

    print(
        "sources:",
    )

    for k, v in source_counts.items():
        print(
            f"  {k}: {v}"
        )

    (
        out / "cache_sources.json"
    ).write_text(
        json.dumps(
            source_counts,
            indent=2,
        ),
        encoding="utf-8",
    )

    if len(entries) < 30:
        raise RuntimeError(
            "Could not reconstruct enough complete Chain rows. "
            "Need raw A/B/C, six pair, and six triple outputs."
        )

    # Primary historical design:
    # first 60 complete probes = DEV; next 157 = locked external if present.
    if len(entries) >= 217:
        entries = entries[:217]

        tr_idx = np.arange(60)
        te_idx = np.arange(
            60,
            217,
        )

        data_mode = (
            "FROZEN_60_DEV_157_EXTERNAL"
        )

    else:
        # Safe fallback for local diagnostic only.
        idx = np.arange(
            len(entries)
        )

        tr_idx, te_idx = train_test_split(
            idx,
            train_size=0.70,
            random_state=SEED,
            shuffle=True,
        )

        data_mode = (
            "FALLBACK_INTERNAL_SPLIT_"
            f"{len(tr_idx)}_{len(te_idx)}"
        )

    print(
        "DATA MODE:",
        data_mode,
    )

    print(
        "train:",
        len(tr_idx),
        "test:",
        len(te_idx),
    )

    queries = [
        x["query"]
        for x in entries
    ]

    texts = {
        v: [
            x["outputs"][v]
            for x in entries
        ]
        for v in ALL_VARIANTS
    }

    manifest = {
        "data_mode":
            data_mode,
        "n_entries":
            len(entries),
        "n_train":
            len(tr_idx),
        "n_test":
            len(te_idx),
        "sources":
            source_counts,
        "train_queries": [
            queries[i]
            for i in tr_idx
        ],
        "test_queries": [
            queries[i]
            for i in te_idx
        ],
    }

    (
        out / "manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # 2. Embeddings.
    # -------------------------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\n[2] Embeddings on",
        device,
        flush=True,
    )

    enc = SentenceTransformer(
        args.embed_model,
        device=device,
    )

    q_full = enc.encode(
        queries,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    z_full = {}

    for v in ALL_VARIANTS:
        print(
            "encoding",
            v,
            flush=True,
        )

        z_full[v] = enc.encode(
            texts[v],
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    # -------------------------------------------------------------------------
    # 3. Primary PCA16 representation.
    # -------------------------------------------------------------------------

    print(
        "\n[3] Pair-only PCA representation...",
        flush=True,
    )

    q, z, pca = build_representation(
        q_full,
        z_full,
        tr_idx,
        args.dim,
    )

    qtr = q[tr_idx]
    qte = q[te_idx]

    ztr = {
        v: z[v][tr_idx]
        for v in ALL_VARIANTS
    }

    zte = {
        v: z[v][te_idx]
        for v in ALL_VARIANTS
    }

    zte_full = {
        v: z_full[v][te_idx]
        for v in ALL_VARIANTS
    }

    # -------------------------------------------------------------------------
    # 4. Pair-only factorized operators.
    # -------------------------------------------------------------------------

    print(
        "\n[4] Fitting pair-only W_A/W_B/W_C...",
        flush=True,
    )

    W, fit_info = fit_ops(
        qtr,
        {
            p: ztr[p]
            for p in PAIRS
        },
        steps=args.steps,
        restarts=args.restarts,
        seed=SEED,
    )

    (
        out / "operator_fit.json"
    ).write_text(
        json.dumps(
            fit_info,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "best pair train loss:",
        fit_info["best_loss"],
    )

    # -------------------------------------------------------------------------
    # 5. Direct maps.
    # -------------------------------------------------------------------------

    print(
        "\n[5] Fitting direct controls...",
        flush=True,
    )

    Dmaps = direct_maps(
        qtr,
        ztr,
    )

    R = Results()

    # -------------------------------------------------------------------------
    # 6. Primary order: suffix/prefix + mechanism ablations.
    # -------------------------------------------------------------------------

    print(
        "\n[6] Primary order mechanism tests...",
        flush=True,
    )

    pair_persistence = (
        evaluate_primary_order(
            R,
            qtr=qtr,
            qte=qte,
            ztr=ztr,
            zte=zte,
            zte_full=zte_full,
            W=W,
            Dmaps=Dmaps,
        )
    )

    pair_persistence.to_csv(
        out / "pair_to_triple_persistence.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # 7. All 15 triple permutation pairs.
    # -------------------------------------------------------------------------

    print(
        "[7] All 15 triple-pair tests...",
        flush=True,
    )

    triple_details = (
        evaluate_all_triples(
            R,
            qtr=qtr,
            qte=qte,
            ztr=ztr,
            zte=zte,
            zte_full=zte_full,
            W=W,
            Dmaps=Dmaps,
        )
    )

    triple_details.to_csv(
        out / "all15_metadata.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # 8. All 18 insertion/count contrasts.
    # -------------------------------------------------------------------------

    print(
        "[8] All 18 insertion/count tests...",
        flush=True,
    )

    insertion_details = (
        evaluate_insertions(
            R,
            qtr=qtr,
            qte=qte,
            ztr=ztr,
            zte=zte,
            zte_full=zte_full,
            W=W,
            Dmaps=Dmaps,
        )
    )

    insertion_details.to_csv(
        out / "insertion_metadata.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # 9. Pair substitution.
    # -------------------------------------------------------------------------

    print(
        "[9] Pair substitution diagnostics...",
        flush=True,
    )

    evaluate_substitution(
        R,
        qtr=qtr,
        qte=qte,
        ztr=ztr,
        zte=zte,
        W=W,
        Dmaps=Dmaps,
    )

    # -------------------------------------------------------------------------
    # 10. Label-permutation control.
    # -------------------------------------------------------------------------

    print(
        "[10] Agent-label permutation control...",
        flush=True,
    )

    label_perm = (
        label_permutation_control(
            qte,
            zte,
            W,
        )
    )

    label_perm.to_csv(
        out / "agent_label_permutations.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # 11. Random nulls.
    # -------------------------------------------------------------------------

    print(
        "[11] Shuffle + matched-random-K nulls...",
        flush=True,
    )

    nulls = null_controls(
        qte,
        zte,
        W,
        reps=args.null_reps,
    )

    (
        out / "nulls.json"
    ).write_text(
        json.dumps(
            nulls,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # 12. Bootstrap.
    # -------------------------------------------------------------------------

    print(
        "[12] Probe bootstrap...",
        flush=True,
    )

    boot = primary_macro_bootstrap(
        qte,
        zte,
        W,
        reps=args.bootstrap_reps,
    )

    (
        out / "bootstrap.json"
    ).write_text(
        json.dumps(
            boot,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # 13. Semantic cluster stability.
    # -------------------------------------------------------------------------

    print(
        "[13] Semantic-cluster stability...",
        flush=True,
    )

    sem = semantic_clusters(
        q_full,
        qte,
        zte,
        W,
        te_idx,
        k=8,
    )

    sem.to_csv(
        out / "semantic_cluster_stability.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # 14. Dimension sensitivity.
    # -------------------------------------------------------------------------

    print(
        "[14] Dimension sweep...",
        flush=True,
    )

    dims = dimension_sweep(
        q_full,
        z_full,
        tr_idx,
        te_idx,
        args,
    )

    dims.to_csv(
        out / "dimension_sweep.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # 15. Save raw / macro tables.
    # -------------------------------------------------------------------------

    raw = R.frame()

    raw.to_csv(
        out / "all_contrast_results.csv",
        index=False,
    )

    macro = macro_table(
        raw
    )

    macro.to_csv(
        out / "macro_results.csv",
        index=False,
    )

    categories = category_table(
        raw
    )

    categories.to_csv(
        out / "triple_category_results.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # 16. REPORT
    # -------------------------------------------------------------------------

    def lookup(
        family,
        method,
        target="PCA",
    ):
        x = macro[
            (
                macro["family"]
                == family
            )
            & (
                macro["method"]
                == method
            )
            & (
                macro["target_space"]
                == target
            )
        ]

        if x.empty:
            return None

        return x.iloc[0]

    suf = lookup(
        "order_suffix",
        "factorized_correct_context",
    )

    pre = lookup(
        "order_prefix",
        "factorized_correct_context",
    )

    suf384 = lookup(
        "order_suffix",
        "factorized_correct_context",
        "FULL384",
    )

    pre384 = lookup(
        "order_prefix",
        "factorized_correct_context",
        "FULL384",
    )

    ins = []

    for fam in (
        "insert_first",
        "insert_middle",
        "insert_last",
    ):
        x = lookup(
            fam,
            "factorized_insertion",
        )

        if x is not None:
            ins.append(x)

    insert_rho = (
        float(
            np.mean([
                x["macro_rho"]
                for x in ins
            ])
        )
        if ins
        else float("nan")
    )

    pair_transfer = []

    for fam in (
        "order_suffix",
        "order_prefix",
    ):
        x = lookup(
            fam,
            "direct_pair_map_transfer",
        )

        if x is not None:
            pair_transfer.append(
                x["macro_rho"]
            )

    pair_transfer_rho = (
        float(
            np.mean(
                pair_transfer
            )
        )
        if pair_transfer
        else float("nan")
    )

    correct_order_rho = (
        float(
            np.mean([
                suf["macro_rho"],
                pre["macro_rho"],
            ])
        )
        if (
            suf is not None
            and pre is not None
        )
        else float("nan")
    )

    full384_rho = (
        float(
            np.mean([
                suf384["macro_rho"],
                pre384["macro_rho"],
            ])
        )
        if (
            suf384 is not None
            and pre384 is not None
        )
        else float("nan")
    )

    label_rank = None

    if not label_perm.empty:
        correct_idx = np.where(
            label_perm["is_correct"].to_numpy()
        )[0]

        if len(correct_idx):
            label_rank = (
                int(correct_idx[0]) + 1
            )

    report = []

    report.append(
        "=" * 118
    )
    report.append(
        "EXHAUSTIVE CHAIN MECHANISM AUDIT V2"
    )
    report.append(
        "=" * 118
    )

    report.append(
        f"DATA MODE: {data_mode}"
    )

    report.append(
        f"train={len(tr_idx)} | "
        f"test={len(te_idx)} | "
        f"primary_dim={q.shape[1]}"
    )

    report.append("")
    report.append(
        "SYNTHETIC GROUND-TRUTH TEST"
    )

    report.append(
        f"  mean rho = "
        f"{synth['rho_mean']:+.4f}"
    )

    report.append(
        f"  min rho  = "
        f"{synth['rho_min']:+.4f}"
    )

    report.append("")
    report.append(
        "PRIMARY ORDER REPRODUCTION"
    )

    if suf is not None:
        report.append(
            f"  common-SUFFIX: "
            f"rho={suf['macro_rho']:+.4f} "
            f"lift={suf['macro_lift']:.3f}x "
            f"oracle={suf['macro_oracle']:.3f}x"
        )

    if pre is not None:
        report.append(
            f"  common-PREFIX: "
            f"rho={pre['macro_rho']:+.4f} "
            f"lift={pre['macro_lift']:.3f}x "
            f"oracle={pre['macro_oracle']:.3f}x"
        )

    report.append(
        f"  combined order rho ~= "
        f"{correct_order_rho:+.4f}"
    )

    report.append("")
    report.append(
        "REPRESENTATION AUDIT"
    )

    report.append(
        f"  same spectral score vs FULL384 target: "
        f"rho ~= {full384_rho:+.4f}"
    )

    report.append("")
    report.append(
        "PAIR-SENSITIVITY EXPLANATION"
    )

    report.append(
        f"  direct pair-map transfer "
        f"macro rho ~= {pair_transfer_rho:+.4f}"
    )

    if not pair_persistence.empty:
        report.append(
            "  actual pair->triple rho:"
        )

        for _, r in pair_persistence.iterrows():
            report.append(
                f"    {r['contrast']:15s} "
                f"{r['rho_pair_actual_to_triple_actual']:+.4f}"
            )

    report.append("")
    report.append(
        "COUNT / INSERTION"
    )

    for fam in (
        "insert_first",
        "insert_middle",
        "insert_last",
    ):
        x = lookup(
            fam,
            "factorized_insertion",
        )

        if x is not None:
            report.append(
                f"  {fam:15s} "
                f"rho={x['macro_rho']:+.4f} "
                f"lift={x['macro_lift']:.3f}x "
                f"oracle={x['macro_oracle']:.3f}x"
            )

    report.append(
        f"  insertion macro rho ~= "
        f"{insert_rho:+.4f}"
    )

    report.append("")
    report.append(
        "15-TRIPLE STRUCTURAL GENERALIZATION"
    )

    if not categories.empty:
        for _, r in categories.iterrows():
            report.append(
                f"  {r['category']:22s} "
                f"n={int(r['n']):2d} "
                f"rho={r['macro_rho']:+.4f} "
                f"lift={r['macro_lift']:.3f}x"
            )

    report.append("")
    report.append(
        "AGENT-LABEL PERMUTATION CONTROL"
    )

    report.append(
        f"  correct labeling rank among 6 = "
        f"{label_rank}"
    )

    if not label_perm.empty:
        report.append(
            f"  best mapping rho = "
            f"{label_perm.iloc[0]['macro_rho']:+.4f}"
        )

        correct = label_perm[
            label_perm["is_correct"]
        ]

        if not correct.empty:
            report.append(
                f"  correct mapping rho = "
                f"{correct.iloc[0]['macro_rho']:+.4f}"
            )

    report.append("")
    report.append(
        "NULL CONTROLS"
    )

    report.append(
        "  shuffle rho: "
        f"mean={nulls['shuffle_rho']['mean']:+.4f}, "
        f"95th={nulls['shuffle_rho']['q95']:+.4f}"
    )

    report.append(
        "  random-K rho: "
        f"mean={nulls['randomK_rho']['mean']:+.4f}, "
        f"95th={nulls['randomK_rho']['q95']:+.4f}"
    )

    report.append("")
    report.append(
        "BOOTSTRAP — six primary order contrasts"
    )

    report.append(
        f"  rho mean={boot['mean']:+.4f} "
        f"95%CI=[{boot['p025']:+.4f}, "
        f"{boot['p975']:+.4f}] "
        f"positive={boot['positive_rate']:.3f}"
    )

    report.append("")
    report.append(
        "DIMENSION SWEEP"
    )

    for _, r in dims.iterrows():
        report.append(
            f"  d={int(r['dim']):2d} | "
            f"suffix={r['suffix_rho']:+.3f} | "
            f"prefix={r['prefix_rho']:+.3f} | "
            f"all-order={r['all_order_rho']:+.3f} | "
            f"insert={r['insert_rho']:+.3f}"
        )

    report.append("")
    report.append(
        "SEMANTIC CLUSTER STABILITY"
    )

    if sem.empty:
        report.append(
            "  insufficient cluster sizes"
        )
    else:
        for _, r in sem.iterrows():
            report.append(
                f"  cluster {int(r['cluster'])}: "
                f"n={int(r['n_test'])}, "
                f"rho={r['macro_order_rho']:+.3f}"
            )

    report.append("")
    report.append(
        "=" * 118
    )
    report.append(
        "MECHANISM FLAGS"
    )
    report.append(
        "=" * 118
    )

    synthetic_ok = (
        synth["rho_mean"] >= 0.90
    )

    order_signal = (
        np.isfinite(
            correct_order_rho
        )
        and correct_order_rho >= 0.20
    )

    fullspace_signal = (
        np.isfinite(
            full384_rho
        )
        and full384_rho >= 0.15
    )

    insertion_signal = (
        np.isfinite(
            insert_rho
        )
        and insert_rho >= 0.15
    )

    pair_explains = (
        np.isfinite(
            pair_transfer_rho
        )
        and np.isfinite(
            correct_order_rho
        )
        and pair_transfer_rho
        >= correct_order_rho - 0.05
    )

    label_specific = (
        label_rank == 1
    )

    report.append(
        f"synthetic implementation valid : "
        f"{synthetic_ok}"
    )

    report.append(
        f"order signal survives          : "
        f"{order_signal}"
    )

    report.append(
        f"survives FULL384 evaluation    : "
        f"{fullspace_signal}"
    )

    report.append(
        f"count/insertion signal         : "
        f"{insertion_signal}"
    )

    report.append(
        f"pair-only baseline explains it : "
        f"{pair_explains}"
    )

    report.append(
        f"correct agent labels rank #1   : "
        f"{label_specific}"
    )

    report.append("")
    report.append(
        "Interpretation:"
    )

    report.append(
        "- synthetic=False => implementation/identifiability problem; do not interpret real-data failures."
    )

    report.append(
        "- order=True + pair-only-explains=False + label-specific=True => strongest evidence for real compositional algebra."
    )

    report.append(
        "- order=True + pair-only-explains=True => Chain result may mostly be transfer of pair order-sensitivity."
    )

    report.append(
        "- PCA positive but FULL384 weak => representation/evaluation-space artifact risk."
    )

    report.append(
        "- order=True + insertion=True => evidence extends beyond commutator/order into agent insertion/count."
    )

    report.append(
        "- order=True + insertion=False => current algebra appears order-specific."
    )

    report.append(
        "=" * 118
    )

    txt = "\n".join(
        report
    )

    (
        out / "REPORT.txt"
    ).write_text(
        txt + "\n",
        encoding="utf-8",
    )

    print()
    print(txt)

    print()
    print(
        f"Saved full audit to: {out}"
    )


if __name__ == "__main__":
    main()
