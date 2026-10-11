#!/usr/bin/env python3
"""
spectral_probe_selector.py

POC for active probing over a hidden 20-agent membership vector.

Core idea
---------
1) Represent current uncertainty as covariance Sigma over agent presence.
2) Extract the top-r spectral ambiguity directions (eigenvectors of Sigma).
3) Learn a model that predicts, from (belief, query text), WHICH agents a probe
   is expected to move / diagnose.
4) Score each candidate by how strongly its predicted diagnostic sensitivity
   aligns with the currently unresolved spectral subspace.

This deliberately does NOT use sentence-embedding distance as the selector.
The text embedding is only an input feature to a learned sensitivity model.

Expected JSONL schema
---------------------
Each line is one state/round:

{
  "state_id": "...",
  "belief": [p1, ..., p20],

  # optional, better than diagonal covariance:
  "particles": [[0,1,...], ...],
  "particle_weights": [w1, ...],

  # optional, used for oracle evaluation:
  "true_z": [0,1,...,0],

  "candidates": [
    {
      "query": "...",

      # ONE of these training targets is needed:
      "belief_after": [p1_next, ..., p20_next],
      # or:
      "posterior_after": [...],
      # or:
      "agent_sensitivity": [s1, ..., s20],

      # optional oracle scalar for evaluation:
      "oracle_gain": 0.37,

      # optional score from your current selector:
      "selector_score": 0.12
    }
  ]
}

If belief_after/posterior_after is available, the target sensitivity is:
    abs(p_after - p_before)

If true_z is also available, oracle_gain is automatically computed as
Brier-risk reduction when candidate["oracle_gain"] is absent:
    sum((p-z)^2) - sum((p_after-z)^2)

Usage
-----
Train + evaluate:
  python spectral_probe_selector.py \
      --train train.jsonl \
      --eval eval.jsonl \
      --out artifacts/spectral_selector_v1

Synthetic smoke test:
  python spectral_probe_selector.py \
      --synthetic-smoke \
      --out artifacts/spectral_selector_smoke

Notes
-----
- Requires: torch, numpy, sentence-transformers
- scipy is optional (used for Spearman); otherwise a rank-correlation fallback is used.
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


# ----------------------------- utilities -----------------------------

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"{path}:{ln}: invalid JSON: {e}")
    return out


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def get_belief(rec: Dict[str, Any]) -> np.ndarray:
    for k in ("belief", "posterior", "agent_belief", "p"):
        if k in rec:
            x = np.asarray(rec[k], dtype=np.float32)
            if x.ndim != 1:
                raise ValueError(f"{k} must be a 1D list")
            return x
    raise KeyError("Record has no belief/posterior/agent_belief/p field")


def get_candidates(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    for k in ("candidates", "probes", "queries"):
        if k in rec:
            if not isinstance(rec[k], list):
                raise ValueError(f"{k} must be a list")
            return rec[k]
    raise KeyError("Record has no candidates/probes/queries field")


def get_query(cand: Dict[str, Any]) -> str:
    for k in ("query", "probe", "text", "prompt"):
        if k in cand:
            return str(cand[k])
    raise KeyError("Candidate has no query/probe/text/prompt")


def get_after_belief(cand: Dict[str, Any]) -> Optional[np.ndarray]:
    for k in ("belief_after", "posterior_after", "next_belief", "next_p"):
        if k in cand:
            return np.asarray(cand[k], dtype=np.float32)
    return None


def get_explicit_sensitivity(cand: Dict[str, Any]) -> Optional[np.ndarray]:
    for k in ("agent_sensitivity", "sensitivity", "agent_effect"):
        if k in cand:
            return np.asarray(cand[k], dtype=np.float32)
    return None


def get_true_z(rec: Dict[str, Any]) -> Optional[np.ndarray]:
    for k in ("true_z", "target_agents", "agent_mask", "gold_agents"):
        if k in rec:
            return np.asarray(rec[k], dtype=np.float32)
    return None


def get_oracle_gain(rec: Dict[str, Any], cand: Dict[str, Any], p: np.ndarray) -> Optional[float]:
    for k in ("oracle_gain", "gain", "oracle_improvement"):
        if k in cand:
            return float(cand[k])

    z = get_true_z(rec)
    p2 = get_after_belief(cand)
    if z is not None and p2 is not None and len(z) == len(p) == len(p2):
        before = float(np.square(p - z).sum())
        after = float(np.square(p2 - z).sum())
        return before - after
    return None


def covariance_from_record(rec: Dict[str, Any], p: np.ndarray) -> np.ndarray:
    """
    Best case: weighted particle posterior over maps.
    Fallback: independent Bernoulli diagonal covariance.
    """
    if "particles" in rec:
        Z = np.asarray(rec["particles"], dtype=np.float64)
        if Z.ndim != 2 or Z.shape[1] != len(p):
            raise ValueError("particles must be [num_particles, num_agents]")
        w = np.asarray(rec.get("particle_weights", np.ones(len(Z))), dtype=np.float64)
        if len(w) != len(Z):
            raise ValueError("particle_weights length mismatch")
        w = np.maximum(w, 0)
        if w.sum() <= 0:
            w = np.ones_like(w)
        w = w / w.sum()
        mu = np.sum(Z * w[:, None], axis=0)
        D = Z - mu
        Sigma = (D * w[:, None]).T @ D
        Sigma = 0.5 * (Sigma + Sigma.T)
        return Sigma.astype(np.float32)

    # independent Bernoulli approximation
    v = np.clip(p * (1.0 - p), 1e-8, None)
    return np.diag(v).astype(np.float32)


def spectral_state(Sigma: np.ndarray, top_r: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns eigenvalues (descending) and eigenvectors columns (descending).
    """
    vals, vecs = np.linalg.eigh(Sigma.astype(np.float64))
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    r = min(top_r, len(vals))
    return vals[:r].astype(np.float32), vecs[:, :r].astype(np.float32)


def rankdata_simple(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def spearman_safe(a: List[float], b: List[float]) -> float:
    if len(a) < 2:
        return float("nan")
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
        return float("nan")
    if spearmanr is not None:
        return float(spearmanr(aa, bb).correlation)
    ra = rankdata_simple(aa)
    rb = rankdata_simple(bb)
    return float(np.corrcoef(ra, rb)[0, 1])


# ----------------------------- data model -----------------------------

@dataclass
class Example:
    state_id: str
    query: str
    p: np.ndarray
    eigvals: np.ndarray
    eigvecs: np.ndarray
    target_sensitivity: np.ndarray
    oracle_gain: Optional[float]
    current_selector_score: Optional[float]


def build_examples(records: List[Dict[str, Any]], top_r: int) -> Tuple[List[Example], int]:
    exs: List[Example] = []
    n_agents = None

    for ri, rec in enumerate(records):
        p = get_belief(rec)
        if n_agents is None:
            n_agents = len(p)
        elif len(p) != n_agents:
            raise ValueError("All records must use same number of agents")

        Sigma = covariance_from_record(rec, p)
        eigvals, eigvecs = spectral_state(Sigma, top_r)
        sid = str(rec.get("state_id", rec.get("id", ri)))

        for cand in get_candidates(rec):
            s = get_explicit_sensitivity(cand)
            if s is None:
                p2 = get_after_belief(cand)
                if p2 is None:
                    continue
                if len(p2) != len(p):
                    continue
                s = np.abs(p2 - p)

            if len(s) != len(p):
                continue

            og = get_oracle_gain(rec, cand, p)
            css = None
            for k in ("selector_score", "current_selector_score", "score"):
                if k in cand:
                    try:
                        css = float(cand[k])
                    except Exception:
                        pass
                    break

            exs.append(
                Example(
                    state_id=sid,
                    query=get_query(cand),
                    p=p.copy(),
                    eigvals=eigvals.copy(),
                    eigvecs=eigvecs.copy(),
                    target_sensitivity=s.astype(np.float32),
                    oracle_gain=og,
                    current_selector_score=css,
                )
            )

    if not exs:
        raise ValueError(
            "No usable training examples found. Need candidate agent_sensitivity "
            "or belief_after/posterior_after."
        )
    return exs, int(n_agents)


# ----------------------------- embeddings -----------------------------

class TextEmbedder:
    def __init__(self, model_name: str, device: str):
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install with: pip install sentence-transformers"
            )
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)


# ----------------------------- neural selector -----------------------------

class SensitivityNet(nn.Module):
    """
    Predict expected per-agent diagnostic sensitivity.

    Inputs:
      belief p                  [N]
      top-r eigenvalues         [r]
      flattened top-r eigvecs   [N*r]
      text embedding            [D]

    Output:
      predicted sensitivity     [N], nonnegative via softplus
    """
    def __init__(self, n_agents: int, top_r: int, emb_dim: int, hidden: int = 512, dropout: float = 0.15):
        super().__init__()
        in_dim = n_agents + top_r + n_agents * top_r + emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_agents),
        )
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.softplus(self.net(x))


class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.Y = torch.from_numpy(Y.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


def make_features(exs: List[Example], embs: np.ndarray, top_r: int) -> np.ndarray:
    rows = []
    for ex, e in zip(exs, embs):
        # eigvecs [N, r] -> flatten by mode, preserving sign structure
        v = ex.eigvecs[:, :top_r].T.reshape(-1)
        lam = ex.eigvals[:top_r]
        rows.append(np.concatenate([ex.p, lam, v, e], axis=0))
    return np.stack(rows).astype(np.float32)


def train_model(
    X: np.ndarray,
    Y: np.ndarray,
    n_agents: int,
    top_r: int,
    emb_dim: int,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden: int,
    dropout: float,
    seed: int,
):
    seed_all(seed)
    model = SensitivityNet(n_agents, top_r, emb_dim, hidden=hidden, dropout=dropout).to(device)

    idx = np.arange(len(X))
    np.random.shuffle(idx)
    cut = max(1, int(0.9 * len(idx)))
    tr, va = idx[:cut], idx[cut:]
    if len(va) == 0:
        va = tr[: min(32, len(tr))]

    train_loader = DataLoader(ArrayDataset(X[tr], Y[tr]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ArrayDataset(X[va], Y[va]), batch_size=batch_size, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=0.05)

    best = None
    best_val = float("inf")

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += float(loss.item()) * len(xb)
            n += len(xb)

        model.eval()
        va_loss = 0.0
        vn = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                va_loss += float(loss.item()) * len(xb)
                vn += len(xb)

        tr_loss /= max(n, 1)
        va_loss /= max(vn, 1)
        print(f"[epoch {ep:03d}] train={tr_loss:.6f} val={va_loss:.6f}")

        if va_loss < best_val:
            best_val = va_loss
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best is not None:
        model.load_state_dict(best)
    return model, best_val


@torch.no_grad()
def mc_predict(model: nn.Module, X: torch.Tensor, mc: int = 16):
    """
    MC dropout:
      mean sensitivity = learned expected agent diagnostic effect
      variance = epistemic uncertainty of our selector model

    This variance is NOT claimed to be MAS observation noise.
    It is only used as a conservative penalty.
    """
    preds = []
    model.train()  # enable dropout intentionally
    for _ in range(mc):
        preds.append(model(X))
    P = torch.stack(preds, dim=0)
    return P.mean(0), P.var(0, unbiased=False)


def spectral_score(
    a_hat: np.ndarray,
    pred_var: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    uncertainty_penalty: float,
    eps: float = 1e-6,
) -> float:
    """
    Top-r weighted spectral alignment:

        numerator = sum_k lambda_k * (v_k^T a)^2

    We penalize high predictor uncertainty:

        denominator = 1 + beta * mean(pred_var)

    This is intentionally a POC score.
    """
    proj = eigvecs.T @ a_hat  # [r]
    numerator = float(np.sum(eigvals * np.square(proj)))
    denominator = 1.0 + uncertainty_penalty * float(np.mean(pred_var))
    return numerator / max(denominator, eps)


# ----------------------------- evaluation -----------------------------

def evaluate(
    records: List[Dict[str, Any]],
    model: nn.Module,
    embedder: TextEmbedder,
    top_r: int,
    device: str,
    mc: int,
    uncertainty_penalty: float,
    out_dir: Path,
):
    rows_out = []

    all_state_spearman = []
    chosen_gains = []
    oracle_best_gains = []
    regrets = []
    hit_best = []
    current_selector_gains = []
    random_gains = []

    rng = random.Random(12345)

    for ri, rec in enumerate(records):
        p = get_belief(rec)
        Sigma = covariance_from_record(rec, p)
        eigvals, eigvecs = spectral_state(Sigma, top_r)
        sid = str(rec.get("state_id", rec.get("id", ri)))
        cands = get_candidates(rec)
        if len(cands) == 0:
            continue

        texts = [get_query(c) for c in cands]
        E = embedder.encode(texts, batch_size=min(64, len(texts)))

        # State features repeated for each candidate
        feats = []
        for e in E:
            feats.append(
                np.concatenate([
                    p,
                    eigvals[:top_r],
                    eigvecs[:, :top_r].T.reshape(-1),
                    e
                ], axis=0)
            )
        X = torch.from_numpy(np.stack(feats).astype(np.float32)).to(device)
        mean_pred, var_pred = mc_predict(model, X, mc=mc)
        A = mean_pred.cpu().numpy()
        V = var_pred.cpu().numpy()

        scores = [
            spectral_score(A[i], V[i], eigvals, eigvecs, uncertainty_penalty)
            for i in range(len(cands))
        ]

        oracle = [get_oracle_gain(rec, c, p) for c in cands]
        valid = [i for i, g in enumerate(oracle) if g is not None]

        chosen_i = int(np.argmax(scores))
        oracle_best_i = None
        state_sp = float("nan")

        if valid:
            o = np.array([oracle[i] for i in valid], dtype=float)
            s = np.array([scores[i] for i in valid], dtype=float)
            state_sp = spearman_safe(s.tolist(), o.tolist())
            if not math.isnan(state_sp):
                all_state_spearman.append(state_sp)

            oracle_best_i = valid[int(np.argmax(o))]
            chosen_gain = float(oracle[chosen_i]) if oracle[chosen_i] is not None else float("nan")
            oracle_best_gain = float(oracle[oracle_best_i])

            if not math.isnan(chosen_gain):
                chosen_gains.append(chosen_gain)
                oracle_best_gains.append(oracle_best_gain)
                regrets.append(oracle_best_gain - chosen_gain)
                hit_best.append(1.0 if chosen_i == oracle_best_i else 0.0)

            # random baseline over candidates with oracle
            ri2 = rng.choice(valid)
            random_gains.append(float(oracle[ri2]))

            # current selector baseline if available
            css = []
            for i, c in enumerate(cands):
                val = None
                for k in ("selector_score", "current_selector_score", "score"):
                    if k in c:
                        try:
                            val = float(c[k])
                        except Exception:
                            val = None
                        break
                css.append(val)
            css_valid = [i for i in valid if css[i] is not None]
            if css_valid:
                ci = max(css_valid, key=lambda i: css[i])
                current_selector_gains.append(float(oracle[ci]))

        rows_out.append({
            "state_id": sid,
            "top_eigenvalues": eigvals.tolist(),
            "principal_direction": eigvecs[:, 0].tolist(),
            "chosen_index": chosen_i,
            "chosen_query": texts[chosen_i],
            "chosen_score": float(scores[chosen_i]),
            "oracle_best_index": oracle_best_i,
            "state_spearman": state_sp,
            "candidates": [
                {
                    "index": i,
                    "query": texts[i],
                    "spectral_score": float(scores[i]),
                    "oracle_gain": oracle[i],
                    "predicted_sensitivity": A[i].tolist(),
                    "predictive_variance_mean": float(V[i].mean()),
                }
                for i in range(len(cands))
            ],
        })

    write_jsonl(str(out_dir / "eval_detailed.jsonl"), rows_out)

    def mean_or_nan(xs):
        return float(np.mean(xs)) if xs else float("nan")

    summary = {
        "num_states": len(rows_out),
        "mean_state_spearman": mean_or_nan(all_state_spearman),
        "mean_chosen_oracle_gain": mean_or_nan(chosen_gains),
        "mean_oracle_best_gain": mean_or_nan(oracle_best_gains),
        "mean_regret": mean_or_nan(regrets),
        "top1_oracle_hit_rate": mean_or_nan(hit_best),
        "mean_random_oracle_gain": mean_or_nan(random_gains),
        "mean_current_selector_oracle_gain": mean_or_nan(current_selector_gains),
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== EVALUATION SUMMARY =====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    return summary


# ----------------------------- synthetic smoke test -----------------------------

def make_synthetic(seed: int = 7, n_states: int = 120, n_agents: int = 20, k: int = 8, emb_latent: int = 32):
    """
    Generates a controlled toy world where:
    - belief uncertainty creates spectral directions
    - each query has a latent diagnostic sensitivity
    - oracle gain is larger when query sensitivity aligns with unresolved modes

    Texts encode latent query IDs only; therefore sentence embeddings carry little useful
    semantics. This smoke test mainly validates the plumbing, not the language model.
    """
    rng = np.random.default_rng(seed)

    # random query families, mapped to agents
    W = rng.normal(size=(emb_latent, n_agents)).astype(np.float32)

    records = []
    for s in range(n_states):
        p = rng.uniform(0.05, 0.95, size=n_agents).astype(np.float32)

        # introduce 2-4 highly uncertain agents
        hot = rng.choice(n_agents, size=rng.integers(2, 5), replace=False)
        p[hot] = rng.uniform(0.42, 0.58, size=len(hot))

        z = (rng.random(n_agents) < p).astype(np.float32)

        cands = []
        for j in range(k):
            h = rng.normal(size=emb_latent).astype(np.float32)
            a = np.abs(h @ W)
            a = a / (np.max(a) + 1e-6)

            # realized posterior movement, stronger on uncertain agents
            movement = 0.35 * a * (4 * p * (1 - p))
            direction = np.sign(z - p)
            p2 = np.clip(p + direction * movement, 0.001, 0.999)

            before = float(np.square(p - z).sum())
            after = float(np.square(p2 - z).sum())
            gain = before - after

            cands.append({
                "query": f"synthetic probe state={s} cand={j} latent={','.join(f'{x:.3f}' for x in h[:6])}",
                "belief_after": p2.tolist(),
                "oracle_gain": gain,
            })

        records.append({
            "state_id": f"s{s}",
            "belief": p.tolist(),
            "true_z": z.tolist(),
            "candidates": cands,
        })
    return records


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, default=None)
    ap.add_argument("--eval", type=str, default=None)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--text-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--top-r", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--mc", type=int, default=16)
    ap.add_argument("--uncertainty-penalty", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--synthetic-smoke", action="store_true")
    args = ap.parse_args()

    seed_all(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print("device:", device)

    if args.synthetic_smoke:
        all_records = make_synthetic(seed=args.seed)
        cut = int(0.75 * len(all_records))
        train_records = all_records[:cut]
        eval_records = all_records[cut:]
        write_jsonl(str(out_dir / "synthetic_train.jsonl"), train_records)
        write_jsonl(str(out_dir / "synthetic_eval.jsonl"), eval_records)
    else:
        if not args.train or not args.eval:
            raise SystemExit("Need --train and --eval unless --synthetic-smoke is used.")
        train_records = read_jsonl(args.train)
        eval_records = read_jsonl(args.eval)

    train_exs, n_agents = build_examples(train_records, top_r=args.top_r)
    print(f"train examples: {len(train_exs)} | agents: {n_agents}")

    embedder = TextEmbedder(args.text_model, device=device)
    train_texts = [x.query for x in train_exs]
    E = embedder.encode(train_texts, batch_size=args.batch_size)
    emb_dim = E.shape[1]

    X = make_features(train_exs, E, top_r=args.top_r)
    Y = np.stack([x.target_sensitivity for x in train_exs]).astype(np.float32)

    print("feature shape:", X.shape)
    print("target shape :", Y.shape)

    model, best_val = train_model(
        X=X,
        Y=Y,
        n_agents=n_agents,
        top_r=args.top_r,
        emb_dim=emb_dim,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden=args.hidden,
        dropout=args.dropout,
        seed=args.seed,
    )

    ckpt = {
        "state_dict": model.state_dict(),
        "n_agents": n_agents,
        "top_r": args.top_r,
        "emb_dim": emb_dim,
        "text_model": args.text_model,
        "best_val": best_val,
        "args": vars(args),
    }
    torch.save(ckpt, out_dir / "selector.pt")
    print("saved:", out_dir / "selector.pt")

    evaluate(
        records=eval_records,
        model=model,
        embedder=embedder,
        top_r=args.top_r,
        device=device,
        mc=args.mc,
        uncertainty_penalty=args.uncertainty_penalty,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
