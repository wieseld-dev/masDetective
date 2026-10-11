from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adaptive_v11_active.io import read_jsonl
from adaptive_v11_active.schema import PublicBeliefState

from adaptive_v11_curriculum_v2.selector import (
    FEATURE_DIM,
    feature_vector,
)

from adaptive_v11_curriculum_v2.selector_v3 import build_model


def bucket(text: str) -> int:
    d = hashlib.blake2b(
        str(text).encode("utf-8"),
        digest_size=8,
    ).digest()

    return int.from_bytes(d, "big") % 100


def split_for_session(session_id: str) -> str:
    b = bucket(session_id)

    if b < 70:
        return "train"
    if b < 85:
        return "val"
    return "test"


def load_groups(root: Path):
    groups = []

    for sdir in sorted(root.glob("stage_*")):
        try:
            stage = int(sdir.name.split("_")[-1])
        except Exception:
            continue

        for train_file in sorted(
            sdir.glob("cycle_*/train_rows_round_*.jsonl")
        ):
            cdir = train_file.parent

            rnd = int(train_file.stem.split("_")[-1])

            state_file = (
                cdir /
                f"states_train_round_{rnd - 1:02d}.jsonl"
            )

            if not state_file.exists():
                continue

            states = {
                str(x["session_id"]): x
                for x in read_jsonl(state_file)
            }

            grouped = defaultdict(list)

            for r in read_jsonl(train_file):
                reward = r.get("reward")

                if not isinstance(reward, (int, float)):
                    continue

                sid = str(r["session_id"])
                gid = str(r.get(
                    "group_id",
                    f"{sid}::r{rnd:02d}",
                ))

                grouped[(sid, gid)].append(r)

            for (sid, gid), rows in grouped.items():

                if sid not in states or len(rows) < 2:
                    continue

                state = PublicBeliefState.from_public_dict(
                    states[sid]
                )

                X = []
                y = []
                stored = []

                rows = sorted(
                    rows,
                    key=lambda r: int(r["candidate_index"]),
                )

                for r in rows:
                    X.append(
                        feature_vector(
                            state,
                            str(r.get("query", "")),
                            int(r["candidate_index"]),
                        )
                    )

                    y.append(float(r["reward"]))

                    p = r.get("selector_predicted_utility")

                    if (
                        isinstance(p, (int, float))
                        and math.isfinite(float(p))
                    ):
                        stored.append(float(p))
                    else:
                        stored.append(float("nan"))

                groups.append({
                    "stage": stage,
                    "round": rnd,
                    "session_id": sid,
                    "group_id": gid,
                    "split": split_for_session(sid),

                    "X": np.stack(X).astype(np.float32),
                    "y": np.asarray(y, dtype=np.float32),

                    # Historical V2 selector prediction.
                    "stored": np.asarray(
                        stored,
                        dtype=np.float32,
                    ),
                })

    return groups


def scaler(groups):
    X = np.concatenate([
        g["X"]
        for g in groups
        if g["split"] == "train"
    ])

    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)

    sd[sd < 1e-6] = 1.0

    return mu, sd


def offline_ridge(groups, mu, sd, ridge=4.0):
    X = np.concatenate([
        (g["X"] - mu) / sd
        for g in groups
        if g["split"] == "train"
    ]).astype(np.float64)

    y = np.concatenate([
        g["y"]
        for g in groups
        if g["split"] == "train"
    ]).astype(np.float64)

    # Explicit intercept because standardization removes
    # the original constant feature.
    Xa = np.concatenate([
        X,
        np.ones((len(X), 1), dtype=np.float64),
    ], axis=1)

    penalty = np.eye(Xa.shape[1], dtype=np.float64) * ridge
    penalty[-1, -1] = 0.0

    A = Xa.T @ Xa + penalty
    b = Xa.T @ y

    try:
        wb = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        wb = np.linalg.lstsq(A, b, rcond=None)[0]

    return (
        wb[:-1].astype(np.float32),
        float(wb[-1]),
    )


def make_pairs(groups, mu, sd, min_gap):
    xa = []
    xb = []
    sign = []
    weight = []

    for g in groups:

        if g["split"] != "train":
            continue

        X = ((g["X"] - mu) / sd).astype(np.float32)
        y = g["y"]

        reward_span = max(
            1e-6,
            float(y.max() - y.min()),
        )

        for i in range(len(y)):
            for j in range(i + 1, len(y)):

                gap = float(y[i] - y[j])

                if abs(gap) < min_gap:
                    continue

                xa.append(X[i])
                xb.append(X[j])

                sign.append(
                    1.0 if gap > 0 else -1.0
                )

                # Large ranking errors matter more.
                normalized_gap = min(
                    1.0,
                    abs(gap) / reward_span,
                )

                weight.append(
                    0.25 + 0.75 * normalized_gap
                )

    return (
        torch.from_numpy(
            np.asarray(xa, dtype=np.float32)
        ),
        torch.from_numpy(
            np.asarray(xb, dtype=np.float32)
        ),
        torch.from_numpy(
            np.asarray(sign, dtype=np.float32)
        ),
        torch.from_numpy(
            np.asarray(weight, dtype=np.float32)
        ),
    )


def spearman(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    if len(a) < 2:
        return float("nan")

    ra = np.empty(len(a))
    rb = np.empty(len(b))

    ra[np.argsort(a)] = np.arange(len(a))
    rb[np.argsort(b)] = np.arange(len(b))

    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")

    return float(np.corrcoef(ra, rb)[0, 1])


def ndcg(reward, score):
    reward = np.asarray(reward)
    score = np.asarray(score)

    relevance = reward - reward.min()

    if float(relevance.max()) <= 1e-12:
        return 1.0

    def dcg(order):
        return float(sum(
            float(relevance[idx]) /
            math.log2(rank + 2)
            for rank, idx in enumerate(order)
        ))

    predicted = np.argsort(-score)
    ideal = np.argsort(-reward)

    denominator = dcg(ideal)

    return (
        dcg(predicted) / denominator
        if denominator > 0
        else 1.0
    )


def evaluate(groups, scorer, split, stage=None):
    result = []

    for g in groups:

        if g["split"] != split:
            continue

        if stage is not None and g["stage"] != stage:
            continue

        y = g["y"].astype(float)
        scores = np.asarray(
            scorer(g),
            dtype=float,
        )

        if (
            len(scores) != len(y)
            or not np.all(np.isfinite(scores))
        ):
            continue

        order = np.argsort(-scores)

        chosen_idx = int(order[0])
        best_reward = float(y.max())
        chosen_reward = float(y[chosen_idx])

        true_order = np.argsort(-y)

        rank = int(
            np.where(true_order == chosen_idx)[0][0]
        ) + 1

        pair_correct = 0
        pair_total = 0

        for i in range(len(y)):
            for j in range(i + 1, len(y)):

                if abs(y[i] - y[j]) < 1e-9:
                    continue

                pair_total += 1

                if (
                    (scores[i] - scores[j]) *
                    (y[i] - y[j])
                    > 0
                ):
                    pair_correct += 1

        result.append({
            "oracle": best_reward,
            "random": float(y.mean()),
            "chosen": chosen_reward,

            "regret":
                best_reward - chosen_reward,

            "top1":
                float(
                    abs(
                        chosen_reward -
                        best_reward
                    ) < 1e-9
                ),

            "top2":
                float(rank <= 2),

            "top3":
                float(rank <= 3),

            "positive":
                float(chosen_reward > 0),

            "pair_acc":
                pair_correct / pair_total
                if pair_total
                else float("nan"),

            "ndcg":
                ndcg(y, scores),

            "spearman":
                spearman(scores, y),
        })

    if not result:
        return {}

    output = {
        "groups": len(result),
    }

    for key in result[0]:
        vals = [
            float(x[key])
            for x in result
            if math.isfinite(float(x[key]))
        ]

        output[key] = (
            float(np.mean(vals))
            if vals
            else float("nan")
        )

    output["uplift_over_random"] = (
        output["chosen"] -
        output["random"]
    )

    denominator = (
        output["oracle"] -
        output["random"]
    )

    output["oracle_signal_captured"] = (
        output["uplift_over_random"] /
        denominator
        if denominator > 1e-9
        else float("nan")
    )

    return output


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        default="artifacts/adaptive_v11_curriculum_v2",
    )

    ap.add_argument(
        "--out",
        default=(
            "artifacts/adaptive_v11_curriculum_v2/"
            "selector_v3"
        ),
    )

    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--min-gap", type=float, default=0.02)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260921)

    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    torch.set_num_threads(
        max(1, min(4, torch.get_num_threads()))
    )

    root = Path(a.root)
    out = Path(a.out)

    out.mkdir(parents=True, exist_ok=True)

    print("[load] reconstructing public-only features...")
    groups = load_groups(root)

    counts = {
        split: sum(
            g["split"] == split
            for g in groups
        )
        for split in ("train", "val", "test")
    }

    print(
        f"[data] groups={len(groups)} "
        f"train={counts['train']} "
        f"val={counts['val']} "
        f"test={counts['test']}",
        flush=True,
    )

    mu, sd = scaler(groups)

    ridge_w, ridge_b = offline_ridge(
        groups,
        mu,
        sd,
    )

    xa, xb, pair_sign, pair_weight = make_pairs(
        groups,
        mu,
        sd,
        a.min_gap,
    )

    n_pairs = len(pair_sign)

    if n_pairs == 0:
        raise RuntimeError("no pairwise training examples")

    print(
        f"[pairs] training_pairs={n_pairs}",
        flush=True,
    )

    model = build_model()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=a.lr,
        weight_decay=a.weight_decay,
    )

    best_val = -1e18
    best_state = None
    bad_epochs = 0

    for epoch in range(1, a.epochs + 1):

        model.train()

        permutation = torch.randperm(n_pairs)

        losses = []

        for start in range(
            0,
            n_pairs,
            a.batch_size,
        ):
            idx = permutation[
                start:start + a.batch_size
            ]

            score_a = model(xa[idx]).squeeze(-1)
            score_b = model(xb[idx]).squeeze(-1)

            signed_difference = (
                pair_sign[idx] *
                (score_a - score_b)
            )

            loss_per_pair = F.softplus(
                -signed_difference
            )

            weighted_loss = (
                loss_per_pair *
                pair_weight[idx]
            ).sum() / (
                pair_weight[idx]
                .sum()
                .clamp_min(1e-6)
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            weighted_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            losses.append(
                float(
                    weighted_loss.detach()
                )
            )

        model.eval()

        def v3_score(g):
            X = (
                (g["X"] - mu) /
                sd
            ).astype(np.float32)

            with torch.no_grad():
                return (
                    model(
                        torch.from_numpy(X)
                    )
                    .squeeze(-1)
                    .cpu()
                    .numpy()
                )

        val = evaluate(
            groups,
            v3_score,
            "val",
        )

        objective = val.get(
            "uplift_over_random",
            float("-inf"),
        )

        print(
            f"[epoch {epoch:02d}] "
            f"loss={np.mean(losses):.5f} "
            f"val_reward={val.get('chosen', float('nan')):.4f} "
            f"val_random={val.get('random', float('nan')):.4f} "
            f"val_uplift={val.get('uplift_over_random', float('nan')):.4f} "
            f"capture={val.get('oracle_signal_captured', float('nan')):.3f} "
            f"regret={val.get('regret', float('nan')):.4f} "
            f"top1={val.get('top1', float('nan')):.3f} "
            f"top2={val.get('top2', float('nan')):.3f} "
            f"ndcg={val.get('ndcg', float('nan')):.3f}",
            flush=True,
        )

        if objective > best_val + 1e-5:

            best_val = objective

            best_state = {
                k: v.detach().cpu().clone()
                for k, v
                in model.state_dict().items()
            }

            bad_epochs = 0

        else:
            bad_epochs += 1

            if bad_epochs >= a.patience:
                print(
                    f"[early-stop] epoch={epoch}",
                    flush=True,
                )
                break

    if best_state is None:
        raise RuntimeError(
            "training produced no model"
        )

    model.load_state_dict(best_state)
    model.eval()

    def v3_score(g):
        X = (
            (g["X"] - mu) /
            sd
        ).astype(np.float32)

        with torch.no_grad():
            return (
                model(
                    torch.from_numpy(X)
                )
                .squeeze(-1)
                .cpu()
                .numpy()
            )

    def ridge_score(g):
        X = (
            (g["X"] - mu) /
            sd
        ).astype(np.float32)

        return X @ ridge_w + ridge_b

    def stored_v2_score(g):
        return g["stored"]

    scorers = {
        "stored_v2_selector":
            stored_v2_score,

        "offline_absolute_ridge":
            ridge_score,

        "v3_pairwise_mlp":
            v3_score,
    }

    report = {
        "data_counts": counts,
        "pairwise_examples": int(n_pairs),
        "best_validation_uplift": float(best_val),
        "test": {},
        "stage10_test": {},
        "by_stage": {},
    }

    for name, scorer in scorers.items():

        report["test"][name] = evaluate(
            groups,
            scorer,
            "test",
        )

        report["stage10_test"][name] = evaluate(
            groups,
            scorer,
            "test",
            stage=10,
        )

    for stage in sorted(
        set(g["stage"] for g in groups)
    ):
        report["by_stage"][str(stage)] = {}

        for name, scorer in scorers.items():
            report["by_stage"][str(stage)][name] = evaluate(
                groups,
                scorer,
                "test",
                stage=stage,
            )

    checkpoint = {
        "state_dict":
            model.state_dict(),

        "feature_dim":
            FEATURE_DIM,

        "mu":
            mu,

        "sd":
            sd,

        "seed":
            a.seed,

        "training_groups":
            counts["train"],

        "pairwise_examples":
            int(n_pairs),

        "architecture":
            "192-128-64-1_gelu",
    }

    torch.save(
        checkpoint,
        out / "pairwise_mlp.pt",
    )

    np.savez_compressed(
        out / "offline_ridge_baseline.npz",
        w=ridge_w,
        bias=np.asarray(ridge_b),
        mu=mu,
        sd=sd,
    )

    (out / "report.json").write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
    )

    lines = []

    lines.append(
        "SELECTOR V3 OFFLINE EVALUATION"
    )

    lines.append(
        f"groups: "
        f"train={counts['train']} "
        f"val={counts['val']} "
        f"test={counts['test']}"
    )

    lines.append(
        f"pairwise examples: {n_pairs}"
    )

    def add_section(title, source):
        lines.append("")
        lines.append(title)

        for name, m in source.items():

            if not m:
                lines.append(
                    f"{name:28s} NO DATA"
                )
                continue

            lines.append(
                f"{name:28s} "
                f"oracle={m['oracle']:.4f} "
                f"random={m['random']:.4f} "
                f"reward={m['chosen']:.4f} "
                f"uplift={m['uplift_over_random']:.4f} "
                f"capture={m['oracle_signal_captured']:.3f} "
                f"regret={m['regret']:.4f} "
                f"top1={m['top1']:.3f} "
                f"top2={m['top2']:.3f} "
                f"positive={m['positive']:.3f} "
                f"pairacc={m['pair_acc']:.3f} "
                f"ndcg={m['ndcg']:.3f} "
                f"rho={m['spearman']:.3f}"
            )

    add_section(
        "===== HELD-OUT TEST =====",
        report["test"],
    )

    add_section(
        "===== HELD-OUT STAGE 10 =====",
        report["stage10_test"],
    )

    text = "\n".join(lines) + "\n"

    (out / "REPORT.txt").write_text(
        text
    )

    print()
    print(text)

    print(
        f"[saved] {out / 'pairwise_mlp.pt'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
