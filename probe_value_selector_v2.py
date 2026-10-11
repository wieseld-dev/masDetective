#!/usr/bin/env python3
"""
probe_value_selector_v2.py

V2: predict the VALUE of a probe directly, instead of predicting a non-negative
per-agent "sensitivity" vector.

Mathematical target
-------------------
Let hidden agent map be Z in {0,1}^N.
At round t, belief mean is mu_t and covariance is Sigma_t.

For a direction u, the law of total variance gives:

    u^T Sigma_t u
      = E_y[u^T Sigma_{t+1}(y) u]
        + E_y[(u^T(mu_{t+1}(y)-mu_t))^2]

Therefore the realized quantity

    (u^T Delta_mu)^2

is a Monte-Carlo sample of variance reduction along u.

We use:
- count direction u_count = 1/sqrt(N) * 1
- top-r ambiguity directions v_1...v_r from Sigma_t

and define a realized probe utility

    G(q,y) =
        alpha * (u_count^T Delta_mu)^2
        + (1-alpha) * mean_k (v_k^T Delta_mu)^2

The selector learns BEFORE execution:

    Q(b,q) ~= E_y[G(q,y) | b,q]

This is exactly the quantity needed to rank candidate probes.

Important:
- belief_after is used ONLY as a training/evaluation label.
- At inference, the model only receives current belief + current spectral state + query.
- Synthetic mode supplies explicit numeric query_features so the smoke test checks the
  learning/selection pipeline rather than whether a sentence encoder can decode random numbers.

JSONL
-----
State:
{
  "state_id": "...",
  "belief": [...],
  "particles": [[...], ...],             # optional
  "particle_weights": [...],             # optional
  "true_z": [...],                       # optional, for oracle Brier gain

  "candidates": [
    {
      "query": "...",
      "belief_after": [...],             # needed to build V2 target
      "oracle_gain": 0.42,               # optional; auto from true_z if absent
      "selector_score": 0.13,            # optional baseline
      "query_features": [...]             # optional numeric features
    }
  ]
}

Usage
-----
Synthetic:
  python probe_value_selector_v2.py \
      --synthetic-smoke \
      --out artifacts/probe_value_v2_smoke \
      --objective spectral

Real:
  python probe_value_selector_v2.py \
      --train train.jsonl \
      --eval eval.jsonl \
      --out artifacts/probe_value_v2 \
      --objective hybrid \
      --alpha-count 0.7
"""

import argparse, json, math, random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_jsonl(path: str, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def get_belief(rec):
    for k in ("belief", "posterior", "agent_belief", "p"):
        if k in rec:
            return np.asarray(rec[k], dtype=np.float32)
    raise KeyError("missing belief/posterior/agent_belief/p")


def get_candidates(rec):
    for k in ("candidates", "probes", "queries"):
        if k in rec:
            return rec[k]
    raise KeyError("missing candidates/probes/queries")


def get_query(c):
    for k in ("query", "probe", "text", "prompt"):
        if k in c:
            return str(c[k])
    raise KeyError("candidate missing query/probe/text/prompt")


def get_after(c):
    for k in ("belief_after", "posterior_after", "next_belief", "next_p"):
        if k in c:
            return np.asarray(c[k], dtype=np.float32)
    return None


def get_true_z(rec):
    for k in ("true_z", "target_agents", "agent_mask", "gold_agents"):
        if k in rec:
            return np.asarray(rec[k], dtype=np.float32)
    return None


def covariance(rec, p):
    if "particles" in rec:
        Z = np.asarray(rec["particles"], dtype=np.float64)
        w = np.asarray(rec.get("particle_weights", np.ones(len(Z))), dtype=np.float64)
        w = np.maximum(w, 0)
        w = w / max(w.sum(), 1e-12)
        mu = (Z * w[:, None]).sum(0)
        D = Z - mu
        S = (D * w[:, None]).T @ D
        return (0.5 * (S + S.T)).astype(np.float32)

    # fallback: independent Bernoulli approximation
    return np.diag(np.clip(p * (1 - p), 1e-8, None)).astype(np.float32)


def spectral(S, r):
    vals, vecs = np.linalg.eigh(S.astype(np.float64))
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    r = min(r, len(vals))
    return vals[:r].astype(np.float32), vecs[:, :r].astype(np.float32)


def realized_utility(p, p2, eigvecs, objective, alpha_count):
    d = (p2 - p).astype(np.float64)
    n = len(p)
    u = np.ones(n, dtype=np.float64) / np.sqrt(n)

    count_gain = float((u @ d) ** 2)
    mode_proj = eigvecs.astype(np.float64).T @ d
    spectral_gain = float(np.mean(mode_proj ** 2))

    if objective == "count":
        return count_gain
    if objective == "spectral":
        return spectral_gain
    if objective == "hybrid":
        return alpha_count * count_gain + (1.0 - alpha_count) * spectral_gain
    raise ValueError(objective)


def oracle_gain(rec, c, p):
    if "oracle_gain" in c:
        return float(c["oracle_gain"])
    z = get_true_z(rec)
    p2 = get_after(c)
    if z is None or p2 is None:
        return None
    return float(np.square(p - z).sum() - np.square(p2 - z).sum())


def spearman_safe(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    if spearmanr is not None:
        return float(spearmanr(a, b).correlation)
    ra = np.empty(len(a), float); rb = np.empty(len(b), float)
    ra[np.argsort(a)] = np.arange(len(a))
    rb[np.argsort(b)] = np.arange(len(b))
    return float(np.corrcoef(ra, rb)[0, 1])


class TextEncoder:
    def __init__(self, name, device):
        if SentenceTransformer is None:
            raise RuntimeError("pip install sentence-transformers")
        self.m = SentenceTransformer(name, device=device)

    def encode(self, texts, batch_size=64):
        return self.m.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)


class ValueNet(nn.Module):
    def __init__(self, in_dim, hidden=512, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        # Predict log(1 + utility) to keep the target well-scaled.
        return self.net(x).squeeze(-1)


class ArrDS(Dataset):
    def __init__(self, X, y, groups):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.groups = np.asarray(groups)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def build_rows(records, top_r, objective, alpha_count):
    rows = []
    n_agents = None
    qfeat_dim = None

    for ri, rec in enumerate(records):
        p = get_belief(rec)
        if n_agents is None: n_agents = len(p)
        if len(p) != n_agents:
            raise ValueError("agent count mismatch")
        S = covariance(rec, p)
        vals, vecs = spectral(S, top_r)
        sid = str(rec.get("state_id", rec.get("id", ri)))

        for ci, c in enumerate(get_candidates(rec)):
            p2 = get_after(c)
            if p2 is None:
                continue
            qf = c.get("query_features")
            if qf is not None:
                qf = np.asarray(qf, dtype=np.float32)
                if qfeat_dim is None: qfeat_dim = len(qf)
                if len(qf) != qfeat_dim:
                    raise ValueError("query_features dim mismatch")

            rows.append({
                "state_id": sid,
                "query": get_query(c),
                "p": p,
                "eigvals": vals,
                "eigvecs": vecs,
                "query_features": qf,
                "target": realized_utility(p, p2, vecs, objective, alpha_count),
                "oracle_gain": oracle_gain(rec, c, p),
                "selector_score": c.get("selector_score", c.get("current_selector_score")),
                "candidate_index": ci,
            })
    if not rows:
        raise ValueError("No usable candidates with belief_after/posterior_after.")
    return rows, n_agents, qfeat_dim


def state_prefix(row, top_r):
    return np.concatenate([
        row["p"],
        row["eigvals"][:top_r],
        row["eigvecs"][:, :top_r].T.reshape(-1),
    ]).astype(np.float32)


def make_X(rows, q_features, top_r):
    return np.stack([
        np.concatenate([state_prefix(r, top_r), qf])
        for r, qf in zip(rows, q_features)
    ]).astype(np.float32)


def grouped_split(rows, val_frac=0.15, seed=7):
    states = sorted(set(r["state_id"] for r in rows))
    rng = random.Random(seed)
    rng.shuffle(states)
    nv = max(1, int(round(len(states) * val_frac)))
    val_states = set(states[:nv])
    tr = [i for i, r in enumerate(rows) if r["state_id"] not in val_states]
    va = [i for i, r in enumerate(rows) if r["state_id"] in val_states]
    return np.asarray(tr), np.asarray(va)


def train(X, y, rows, device, epochs, batch_size, lr, hidden, dropout, seed):
    tr, va = grouped_split(rows, seed=seed)
    model = ValueNet(X.shape[1], hidden, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(beta=0.1)

    Xtr = torch.from_numpy(X[tr]); ytr = torch.from_numpy(np.log1p(y[tr]).astype(np.float32))
    Xva = torch.from_numpy(X[va]); yva = torch.from_numpy(np.log1p(y[va]).astype(np.float32))

    tr_loader = DataLoader(list(zip(Xtr, ytr)), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(list(zip(Xva, yva)), batch_size=batch_size, shuffle=False)

    best = None; best_val = float("inf")
    for ep in range(1, epochs + 1):
        model.train(); total = 0.; n = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * len(xb); n += len(xb)

        model.eval(); vt = 0.; vn = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = loss_fn(model(xb), yb)
                vt += float(loss) * len(xb); vn += len(xb)

        a = total/max(n,1); b = vt/max(vn,1)
        print(f"[epoch {ep:03d}] train={a:.6f} val={b:.6f}")
        if b < best_val:
            best_val = b
            best = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}

    model.load_state_dict(best)
    return model, best_val


@torch.no_grad()
def predict_mc(model, X, device, mc=16):
    xt = torch.from_numpy(X.astype(np.float32)).to(device)
    ps = []
    model.train()  # MC dropout
    for _ in range(mc):
        ps.append(model(xt))
    P = torch.stack(ps)
    mean_log = P.mean(0)
    var_log = P.var(0, unbiased=False)
    mean_u = torch.expm1(mean_log).clamp_min(0)
    return mean_u.cpu().numpy(), var_log.cpu().numpy()


def evaluate(records, rows, X, model, device, mc, uncertainty_penalty, out_dir):
    pred, var = predict_mc(model, X, device, mc)
    score = pred / (1.0 + uncertainty_penalty * var)

    by_state = {}
    for i, r in enumerate(rows):
        by_state.setdefault(r["state_id"], []).append(i)

    state_sps = []
    chosen_gain = []; oracle_best_gain = []; regrets = []; hits = []; random_gain = []
    realized_target_sps = []
    rng = random.Random(123)

    detailed = []
    for sid, idxs in by_state.items():
        og = [rows[i]["oracle_gain"] for i in idxs]
        valid = [j for j, g in enumerate(og) if g is not None]
        local_scores = [float(score[i]) for i in idxs]
        local_targets = [float(rows[i]["target"]) for i in idxs]

        sp = float("nan")
        oracle_target_sp = float("nan")
        if len(valid) >= 2:
            os = [og[j] for j in valid]
            ps = [local_scores[j] for j in valid]
            ts = [local_targets[j] for j in valid]
            sp = spearman_safe(ps, os)
            oracle_target_sp = spearman_safe(ts, os)
            if not math.isnan(sp): state_sps.append(sp)
            if not math.isnan(oracle_target_sp): realized_target_sps.append(oracle_target_sp)

            ci = max(valid, key=lambda j: local_scores[j])
            bi = max(valid, key=lambda j: og[j])
            chosen_gain.append(float(og[ci]))
            oracle_best_gain.append(float(og[bi]))
            regrets.append(float(og[bi] - og[ci]))
            hits.append(float(ci == bi))
            random_gain.append(float(og[rng.choice(valid)]))

        detailed.append({
            "state_id": sid,
            "pred_vs_oracle_spearman": sp,
            "realized_utility_vs_oracle_spearman": oracle_target_sp,
            "candidates": [
                {
                    "candidate_index": rows[i]["candidate_index"],
                    "query": rows[i]["query"],
                    "predicted_value": float(pred[i]),
                    "penalized_score": float(score[i]),
                    "predictive_var": float(var[i]),
                    "realized_utility_label": float(rows[i]["target"]),
                    "oracle_gain": rows[i]["oracle_gain"],
                }
                for i in idxs
            ]
        })

    def m(x): return float(np.mean(x)) if x else float("nan")
    summary = {
        "num_states": len(by_state),
        "mean_pred_vs_oracle_spearman": m(state_sps),
        "mean_realized_utility_vs_oracle_spearman": m(realized_target_sps),
        "mean_chosen_oracle_gain": m(chosen_gain),
        "mean_oracle_best_gain": m(oracle_best_gain),
        "mean_regret": m(regrets),
        "top1_oracle_hit_rate": m(hits),
        "mean_random_oracle_gain": m(random_gain),
    }

    write_jsonl(str(out_dir / "eval_detailed.jsonl"), detailed)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== V2 SUMMARY =====")
    for k,v in summary.items(): print(f"{k}: {v}")
    return summary


def synthetic(seed=7, n_states=240, n_agents=20, k=8, qdim=24):
    """
    Controlled world:
    numeric query_features -> per-agent diagnostic profile.
    This deliberately tests the selection math and learned value function,
    not natural-language embedding quality.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(qdim, n_agents)).astype(np.float32)
    recs = []

    for s in range(n_states):
        p = rng.uniform(.05,.95,size=n_agents).astype(np.float32)
        hot = rng.choice(n_agents, size=rng.integers(2,6), replace=False)
        p[hot] = rng.uniform(.42,.58,size=len(hot))
        z = (rng.random(n_agents) < p).astype(np.float32)

        cs = []
        for j in range(k):
            h = rng.normal(size=qdim).astype(np.float32)
            a = np.abs(h @ W)
            a /= (a.max()+1e-8)
            move = .35 * a * (4*p*(1-p))
            p2 = np.clip(p + np.sign(z-p)*move, .001,.999).astype(np.float32)
            gain = float(np.square(p-z).sum() - np.square(p2-z).sum())
            cs.append({
                "query": f"synthetic-candidate-{j}",
                "query_features": h.tolist(),
                "belief_after": p2.tolist(),
                "oracle_gain": gain,
            })

        recs.append({
            "state_id": f"s{s}",
            "belief": p.tolist(),
            "true_z": z.tolist(),
            "candidates": cs,
        })
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train")
    ap.add_argument("--eval")
    ap.add_argument("--out", required=True)
    ap.add_argument("--objective", choices=["count","spectral","hybrid"], default="spectral")
    ap.add_argument("--alpha-count", type=float, default=.7)
    ap.add_argument("--top-r", type=int, default=4)
    ap.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=.15)
    ap.add_argument("--mc", type=int, default=16)
    ap.add_argument("--uncertainty-penalty", type=float, default=.5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--synthetic-smoke", action="store_true")
    args = ap.parse_args()

    seed_all(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device
    print("device:", device)

    if args.synthetic_smoke:
        allr = synthetic(args.seed)
        cut = int(.75*len(allr))
        train_rec, eval_rec = allr[:cut], allr[cut:]
        write_jsonl(str(out/"synthetic_train.jsonl"), train_rec)
        write_jsonl(str(out/"synthetic_eval.jsonl"), eval_rec)
    else:
        if not args.train or not args.eval:
            raise SystemExit("Need --train and --eval unless --synthetic-smoke.")
        train_rec = read_jsonl(args.train)
        eval_rec = read_jsonl(args.eval)

    train_rows, n_agents, qdim_train = build_rows(train_rec, args.top_r, args.objective, args.alpha_count)
    eval_rows, n_agents2, qdim_eval = build_rows(eval_rec, args.top_r, args.objective, args.alpha_count)
    if n_agents != n_agents2:
        raise ValueError("train/eval agent count mismatch")

    use_numeric = qdim_train is not None and qdim_eval is not None
    if use_numeric:
        print("using explicit numeric query_features:", qdim_train)
        Qtr = np.stack([r["query_features"] for r in train_rows]).astype(np.float32)
        Qev = np.stack([r["query_features"] for r in eval_rows]).astype(np.float32)
        qdim = Qtr.shape[1]
        encoder_name = None
    else:
        print("using sentence-transformer query embeddings")
        enc = TextEncoder(args.text_model, device)
        Qtr = enc.encode([r["query"] for r in train_rows], args.batch_size)
        Qev = enc.encode([r["query"] for r in eval_rows], args.batch_size)
        qdim = Qtr.shape[1]
        encoder_name = args.text_model

    Xtr = make_X(train_rows, Qtr, args.top_r)
    Xev = make_X(eval_rows, Qev, args.top_r)
    ytr = np.asarray([r["target"] for r in train_rows], dtype=np.float32)

    print("train examples:", len(train_rows), "| eval examples:", len(eval_rows))
    print("X:", Xtr.shape, "| y mean:", float(ytr.mean()), "| y max:", float(ytr.max()))

    model, best_val = train(
        Xtr, ytr, train_rows, device,
        args.epochs, args.batch_size, args.lr,
        args.hidden, args.dropout, args.seed
    )

    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": Xtr.shape[1],
        "n_agents": n_agents,
        "top_r": args.top_r,
        "query_feature_dim": qdim,
        "encoder_name": encoder_name,
        "objective": args.objective,
        "alpha_count": args.alpha_count,
        "best_val": best_val,
        "args": vars(args),
    }, out/"value_selector.pt")

    evaluate(
        eval_rec, eval_rows, Xev, model, device,
        args.mc, args.uncertainty_penalty, out
    )


if __name__ == "__main__":
    main()
