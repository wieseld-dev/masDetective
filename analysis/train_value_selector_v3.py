#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


def read_jsonl(p):
    with Path(p).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def hbin(p):
    p = np.clip(np.asarray(p, float), 1e-8, 1 - 1e-8)
    return -(p*np.log(p) + (1-p)*np.log(1-p))


def entropy_cat(p):
    p = np.asarray(p, float)
    p = p / max(p.sum(), 1e-12)
    p = np.clip(p, 1e-12, 1)
    return float(-(p*np.log(p)).sum())


def safe_spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None

    if spearmanr is not None:
        x = spearmanr(a, b).correlation
        return None if np.isnan(x) else float(x)

    ra = np.empty(len(a))
    rb = np.empty(len(b))
    ra[np.argsort(a)] = np.arange(len(a))
    rb[np.argsort(b)] = np.arange(len(b))
    x = np.corrcoef(ra, rb)[0,1]
    return None if np.isnan(x) else float(x)


class ValueNetV3(nn.Module):
    def __init__(self, d, hidden=256, dropout=.10):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
        )

        self.role_head = nn.Linear(hidden // 2, 1)
        self.k_head = nn.Linear(hidden // 2, 1)

    def forward(self, x):
        h = self.trunk(x)
        return (
            self.role_head(h).squeeze(-1),
            self.k_head(h).squeeze(-1),
        )


def build_X(rows, qemb, roles, k_keys):
    X = []

    for x in rows:
        p = np.asarray(
            [x["role_probs_before"][r] for r in roles],
            dtype=np.float32,
        )

        kp = np.asarray(
            [
                x["k_probs_before"].get(str(k), 0.0)
                for k in k_keys
            ],
            dtype=np.float32,
        )

        q = qemb[int(x["example_index"])].astype(np.float32)

        # All visible / deployable.
        extra = np.asarray([
            float(hbin(p).sum()),
            float(np.mean(p * (1-p))),
            entropy_cat(kp),

            float(x["mi_k_baseline"]),
            float(x["mi_identity_baseline"]),
            float(x["mi_z_baseline"]),
            float(x["predicted_exposure_baseline"]),
        ], dtype=np.float32)

        feat = np.concatenate([
            p,                  # current role belief
            p * (1-p),         # role uncertainty
            kp,                 # current K belief
            q,                  # visible query embedding
            extra,              # visible planning signals
        ])

        X.append(feat)

    return np.stack(X).astype(np.float32)


def target_array(rows, key):
    return np.asarray(
        [float(x[key]) for x in rows],
        dtype=np.float32,
    )


def normalize_train_val(Xtr, Xva):
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)

    return (
        ((Xtr - mu) / sd).astype(np.float32),
        ((Xva - mu) / sd).astype(np.float32),
        mu.astype(np.float32),
        sd.astype(np.float32),
    )


def evaluate_ranking(rows, pred, target_key, baseline_key):
    groups = defaultdict(list)

    for i, x in enumerate(rows):
        groups[x["state_id"]].append(i)

    chosen = []
    baseline = []
    oracle = []
    random_gain = []
    regrets = []
    baseline_regrets = []
    hits = []
    baseline_hits = []
    rhos = []

    details = []

    for sid, idx in sorted(groups.items()):
        y = np.asarray(
            [rows[i][target_key] for i in idx],
            dtype=float,
        )
        p = np.asarray(
            [pred[i] for i in idx],
            dtype=float,
        )
        b = np.asarray(
            [rows[i][baseline_key] for i in idx],
            dtype=float,
        )

        im = int(np.argmax(p))
        ib = int(np.argmax(b))
        io = int(np.argmax(y))

        mg = float(y[im])
        bg = float(y[ib])
        og = float(y[io])
        rg = float(np.mean(y))

        rho = safe_spearman(p, y)

        chosen.append(mg)
        baseline.append(bg)
        oracle.append(og)
        random_gain.append(rg)

        regrets.append(og - mg)
        baseline_regrets.append(og - bg)

        hits.append(int(mg >= og - 1e-12))
        baseline_hits.append(int(bg >= og - 1e-12))

        if rho is not None:
            rhos.append(rho)

        details.append({
            "state_id": sid,
            "model_candidate_index":
                int(rows[idx[im]]["candidate_index"]),
            "baseline_candidate_index":
                int(rows[idx[ib]]["candidate_index"]),
            "oracle_candidate_index":
                int(rows[idx[io]]["candidate_index"]),
            "model_gain": mg,
            "baseline_gain": bg,
            "oracle_gain": og,
            "random_mean_gain": rg,
            "model_regret": og - mg,
            "baseline_regret": og - bg,
            "spearman": rho,
        })

    m = lambda z: float(np.mean(z)) if z else None

    oracle_m = m(oracle)
    random_m = m(random_gain)
    model_m = m(chosen)

    denom = (
        oracle_m - random_m
        if oracle_m is not None and random_m is not None
        else 0.0
    )

    capture = (
        (model_m - random_m) / denom
        if abs(denom) > 1e-12 else None
    )

    return {
        "states": len(groups),

        "mean_model_gain": model_m,
        "mean_baseline_gain": m(baseline),
        "mean_random_gain": random_m,
        "mean_oracle_gain": oracle_m,

        "mean_model_regret": m(regrets),
        "mean_baseline_regret": m(baseline_regrets),

        "model_oracle_hit_rate": m(hits),
        "baseline_oracle_hit_rate": m(baseline_hits),

        "mean_within_state_spearman": m(rhos),

        "model_positive_gain_fraction":
            m([x > 0 for x in chosen]),

        "oracle_headroom_over_random":
            None if oracle_m is None else oracle_m - random_m,

        "model_headroom_capture_fraction":
            capture,

        "details": details,
    }


def strip_details(d):
    z = dict(d)
    z.pop("details", None)
    return z


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--train",
        default="artifacts/visible_realized_gain_v3/train.jsonl",
    )
    ap.add_argument(
        "--val",
        default="artifacts/visible_realized_gain_v3/val.jsonl",
    )
    ap.add_argument(
        "--obs-dir",
        default="artifacts/observation_model_v1",
    )
    ap.add_argument(
        "--out",
        default="artifacts/value_selector_v3",
    )

    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=.10)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1390)

    args = ap.parse_args()
    seed_all(args.seed)

    out = Path(args.out)

    if out.exists() and any(out.iterdir()):
        raise RuntimeError(
            f"refusing non-empty output dir: {out}"
        )

    out.mkdir(parents=True)

    tr = read_jsonl(args.train)
    va = read_jsonl(args.val)

    print(
        f"train rows={len(tr)} "
        f"states={len(set(x['state_id'] for x in tr))}"
    )
    print(
        f"val rows={len(va)} "
        f"states={len(set(x['state_id'] for x in va))}"
    )

    ckpt = torch.load(
        Path(args.obs_dir) / "model.pt",
        map_location="cpu",
    )
    roles = list(ckpt["role_order"])

    qemb = np.load(
        Path(args.obs_dir) / "query_embeddings.npy"
    ).astype(np.float32)

    k_keys = sorted({
        int(k)
        for x in tr + va
        for k in x["k_probs_before"]
    })

    print("roles:", len(roles))
    print("K keys:", k_keys)
    print("query dim:", qemb.shape[1])

    Xtr = build_X(tr, qemb, roles, k_keys)
    Xva = build_X(va, qemb, roles, k_keys)

    Xtr, Xva, x_mu, x_sd = normalize_train_val(
        Xtr, Xva
    )

    yr_tr = target_array(
        tr,
        "role_brier_gain_training_only",
    )
    yk_tr = target_array(
        tr,
        "k_brier_gain_training_only",
    )

    yr_va = target_array(
        va,
        "role_brier_gain_training_only",
    )
    yk_va = target_array(
        va,
        "k_brier_gain_training_only",
    )

    # Signed targets: standardized, NOT log1p.
    yr_mu, yr_sd = float(yr_tr.mean()), float(yr_tr.std())
    yk_mu, yk_sd = float(yk_tr.mean()), float(yk_tr.std())

    yr_sd = max(yr_sd, 1e-8)
    yk_sd = max(yk_sd, 1e-8)

    Yr = ((yr_tr - yr_mu) / yr_sd).astype(np.float32)
    Yk = ((yk_tr - yk_mu) / yk_sd).astype(np.float32)

    Vr = ((yr_va - yr_mu) / yr_sd).astype(np.float32)
    Vk = ((yk_va - yk_mu) / yk_sd).astype(np.float32)

    print(
        f"role target mean={yr_mu:+.6f} sd={yr_sd:.6f}"
    )
    print(
        f"K target    mean={yk_mu:+.6f} sd={yk_sd:.6f}"
    )
    print("X shape:", Xtr.shape)

    ds = TensorDataset(
        torch.from_numpy(Xtr),
        torch.from_numpy(Yr),
        torch.from_numpy(Yk),
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    device = torch.device("cpu")

    model = ValueNetV3(
        Xtr.shape[1],
        args.hidden,
        args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    lossfn = nn.SmoothL1Loss(beta=.5)

    Xv = torch.from_numpy(Xva).to(device)
    Vrt = torch.from_numpy(Vr).to(device)
    Vkt = torch.from_numpy(Vk).to(device)

    best = None
    best_loss = float("inf")
    best_epoch = None
    bad = 0
    history = []

    for ep in range(1, args.epochs + 1):
        model.train()

        total = 0.0
        n = 0

        for xb, rb, kb in dl:
            xb = xb.to(device)
            rb = rb.to(device)
            kb = kb.to(device)

            opt.zero_grad(set_to_none=True)

            pr, pk = model(xb)

            lr = lossfn(pr, rb)
            lk = lossfn(pk, kb)

            loss = lr + lk
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            opt.step()

            total += float(loss.detach()) * len(xb)
            n += len(xb)

        model.eval()

        with torch.no_grad():
            vr, vk = model(Xv)

            vlr = float(lossfn(vr, Vrt))
            vlk = float(lossfn(vk, Vkt))
            vl = vlr + vlk

        row = {
            "epoch": ep,
            "train_loss": total / max(n, 1),
            "val_role_loss": vlr,
            "val_k_loss": vlk,
            "val_total_loss": vl,
        }
        history.append(row)

        print(
            f"epoch={ep:02d} "
            f"train={row['train_loss']:.5f} "
            f"val_role={vlr:.5f} "
            f"val_K={vlk:.5f} "
            f"total={vl:.5f}"
        )

        if vl < best_loss - 1e-5:
            best_loss = vl
            best_epoch = ep
            best = {
                k: v.detach().cpu().clone()
                for k,v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1

            if bad >= args.patience:
                print("early stop")
                break

    model.load_state_dict(best)
    model.eval()

    with torch.no_grad():
        pr, pk = model(Xv)

    pred_role = (
        pr.numpy() * yr_sd + yr_mu
    )
    pred_k = (
        pk.numpy() * yk_sd + yk_mu
    )

    role_eval = evaluate_ranking(
        va,
        pred_role,
        "role_brier_gain_training_only",
        "mi_z_baseline",
    )

    k_eval = evaluate_ranking(
        va,
        pred_k,
        "k_brier_gain_training_only",
        "mi_k_baseline",
    )

    summary = {
        "protocol": "value_selector_v3_round1",
        "inputs_visible_only": True,
        "training_labels_truth_side_only": True,
        "test_used": False,

        "train_rows": len(tr),
        "train_states":
            len(set(x["state_id"] for x in tr)),
        "val_rows": len(va),
        "val_states":
            len(set(x["state_id"] for x in va)),

        "feature_dim": int(Xtr.shape[1]),
        "query_embedding_dim": int(qemb.shape[1]),
        "roles": roles,
        "k_keys": k_keys,

        "best_epoch": best_epoch,
        "best_val_loss": best_loss,

        "role_target_train_mean": yr_mu,
        "role_target_train_sd": yr_sd,
        "k_target_train_mean": yk_mu,
        "k_target_train_sd": yk_sd,

        "role_head": strip_details(role_eval),
        "k_head": strip_details(k_eval),
    }

    torch.save({
        "state_dict": best,
        "feature_dim": Xtr.shape[1],
        "hidden": args.hidden,
        "dropout": args.dropout,
        "roles": roles,
        "k_keys": k_keys,
        "x_mean": x_mu,
        "x_std": x_sd,
        "role_target_mean": yr_mu,
        "role_target_std": yr_sd,
        "k_target_mean": yk_mu,
        "k_target_std": yk_sd,
        "best_epoch": best_epoch,
        "args": vars(args),
    }, out / "value_selector_v3.pt")

    with (out / "training_history.json").open("w") as f:
        json.dump(history, f, indent=2)

    with (out / "val_role_detailed.jsonl").open("w") as f:
        for x in role_eval["details"]:
            f.write(json.dumps(x) + "\n")

    with (out / "val_k_detailed.jsonl").open("w") as f:
        for x in k_eval["details"]:
            f.write(json.dumps(x) + "\n")

    with (out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== VALUE SELECTOR V3 =====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
