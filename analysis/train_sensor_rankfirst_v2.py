from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from v8.embedding import FrozenOutputEmbedderV8
from v11.role_universe import ROLE_DESCRIPTIONS


TRAIN_PATH = Path("artifacts/v137_presence_v1/dataset_v1/train_v137.jsonl")
VAL_PATH = Path("artifacts/v137_presence_v1/dataset_v1/dev_v137.jsonl")
OUT = Path("artifacts/sensor_rankfirst_v2")

SEEDS = [1397, 2027, 4099, 7919, 12011, 17041, 23003, 31013]

EPOCHS = 600
LR = 1e-3
WEIGHT_DECAY = 1e-3
DROPOUT = 0.10

CF_WEIGHT = 0.35
BCE_WEIGHT = 0.10

LINEAR_C = 0.05


def read_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def write_json(path: Path, obj):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / den)


FEATURE_NAMES = [
    "cos_output_role",
    "cos_output_query",
    "cos_query_role",
    "norm_output_minus_role",
    "norm_output_minus_query",
    "norm_query_minus_role",
    "cos_output_role_minus_query_role",
    "cos_output_query_minus_query_role",
    "cos_output_minus_query_role",
    "cos_output_minus_query_role_minus_query",
    "cos_output_minus_role_query",
    "cos_output_minus_role_query_minus_role",
    "norm_response_shift_dup",
    "norm_output_minus_role_dup",
    "norm_role_minus_query_dup",
    "projection_shift_on_role",
    "projection_shift_on_role_minus_query",
    "triangle_slack_role_via_query",
    "triangle_slack_query_via_role",
    "response_shift_over_role_query_distance",
]


print("Loading FrozenOutputEmbedderV8 on CPU...", flush=True)
embedder = FrozenOutputEmbedderV8(device="cpu")
embedding_cache = {}


def emb(text: str):
    text = str(text)
    if text not in embedding_cache:
        x = embedder.encode(text)
        embedding_cache[text] = np.asarray(
            x, dtype=np.float32
        ).reshape(-1)
    return embedding_cache[text]


def geom_from_vectors(r, q, o):
    oq = o - q
    or_ = o - r
    rq = r - q
    qr = q - r

    d_or = float(np.linalg.norm(or_))
    d_oq = float(np.linalg.norm(oq))
    d_qr = float(np.linalg.norm(q - r))

    cos_or = cosine(o, r)
    cos_oq = cosine(o, q)
    cos_qr = cosine(q, r)

    return np.asarray(
        [
            cos_or,
            cos_oq,
            cos_qr,

            d_or,
            d_oq,
            d_qr,

            cos_or - cos_qr,
            cos_oq - cos_qr,

            cosine(oq, r),
            cosine(oq, rq),
            cosine(or_, q),
            cosine(or_, qr),

            d_oq,
            d_or,
            d_qr,

            float(np.dot(oq, r)),
            float(np.dot(oq, rq)),

            d_oq + d_qr - d_or,
            d_or + d_qr - d_oq,

            d_oq / max(d_qr, 1e-6),
        ],
        dtype=np.float32,
    )


def geom_features(row, mode="full"):
    role = str(row["target_role"])
    desc = ROLE_DESCRIPTIONS.get(role, role)

    r = emb(desc)
    q = emb(str(row["query"]))

    if mode == "full":
        o = emb(str(row["final_output"]))
    elif mode == "query_only":
        # Same dimensionality/capacity, but zero response-specific
        # information: O is replaced by Q.
        o = q.copy()
    else:
        raise ValueError(mode)

    return geom_from_vectors(r, q, o)


def make_matrix(rows, mode):
    return np.stack(
        [geom_features(x, mode=mode) for x in rows]
    ).astype(np.float32)


def group_indices(rows, key):
    out = defaultdict(list)
    for i, row in enumerate(rows):
        out[str(row[key])].append(i)
    return dict(out)


def build_t1_pairs(rows):
    by_pair = group_indices(rows, "pair_id")
    pairs = []

    for pid, idxs in sorted(by_pair.items()):
        if pid in ("", "None", "null"):
            continue

        rr = [rows[i] for i in idxs]
        if len(rr) != 2:
            raise RuntimeError(
                f"T1 pair {pid}: expected 2 rows, got {len(rr)}"
            )

        pos = [i for i in idxs if int(rows[i]["label"]) == 1]
        neg = [i for i in idxs if int(rows[i]["label"]) == 0]

        if len(pos) != 1 or len(neg) != 1:
            raise RuntimeError(
                f"T1 pair {pid}: expected one positive and one negative"
            )

        a = rows[pos[0]]
        b = rows[neg[0]]

        if str(a["target_role"]) != str(b["target_role"]):
            raise RuntimeError(f"T1 role mismatch: {pid}")
        if str(a["query"]) != str(b["query"]):
            raise RuntimeError(f"T1 query mismatch: {pid}")

        pairs.append((pos[0], neg[0]))

    return pairs


def evaluate(rows, scores):
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray([int(x["label"]) for x in rows], dtype=np.int64)

    result = {
        "rows": len(rows),
        "systems": len(set(str(x["system_id"]) for x in rows)),
    }

    if len(np.unique(y)) == 2:
        result["global_auc"] = float(roc_auc_score(y, scores))
    else:
        result["global_auc"] = None

    by_role = defaultdict(list)
    for i, row in enumerate(rows):
        by_role[str(row["target_role"])].append(i)

    role_aucs = {}
    for role, idxs in sorted(by_role.items()):
        yy = y[idxs]
        ss = scores[idxs]
        if len(np.unique(yy)) == 2:
            role_aucs[role] = float(roc_auc_score(yy, ss))

    result["per_role_auc"] = role_aucs
    result["mean_role_auc"] = (
        float(np.mean(list(role_aucs.values())))
        if role_aucs else None
    )
    result["median_role_auc"] = (
        float(np.median(list(role_aucs.values())))
        if role_aucs else None
    )

    by_system = group_indices(rows, "system_id")

    system_aucs = []
    pair_accs = []
    margins = []
    perfect = []
    exact_topk = []

    for sid, idxs in sorted(by_system.items()):
        yy = y[idxs]
        ss = scores[idxs]

        p = np.where(yy == 1)[0]
        n = np.where(yy == 0)[0]

        if len(p) == 0 or len(n) == 0:
            continue

        system_aucs.append(
            float(roc_auc_score(yy, ss))
        )

        wins = []
        for pi in p:
            for ni in n:
                if ss[pi] > ss[ni]:
                    wins.append(1.0)
                elif ss[pi] < ss[ni]:
                    wins.append(0.0)
                else:
                    wins.append(0.5)

        pair_accs.append(float(np.mean(wins)))

        margin = float(
            np.min(ss[p]) - np.max(ss[n])
        )
        margins.append(margin)
        perfect.append(float(margin > 0.0))

        k_true = int(len(p))
        order = np.argsort(-ss)
        pred_local = set(order[:k_true].tolist())
        true_local = set(p.tolist())

        exact_topk.append(
            float(pred_local == true_local)
        )

    result["mean_within_system_auc"] = float(np.mean(system_aucs))
    result["mean_system_pair_accuracy"] = float(np.mean(pair_accs))
    result["perfect_order_rate"] = float(np.mean(perfect))
    result["exact_topK_oracleK_rate"] = float(np.mean(exact_topk))
    result["mean_full_separation_margin"] = float(np.mean(margins))
    result["median_full_separation_margin"] = float(np.median(margins))

    return result


def shuffled_output_rows(rows):
    # Keep candidate role + query fixed.
    # Rotate outputs across systems within each role.
    out = [dict(x) for x in rows]
    by_role = group_indices(rows, "target_role")

    for role, idxs in sorted(by_role.items()):
        if len(idxs) < 2:
            continue

        ordered = sorted(
            idxs,
            key=lambda i: str(rows[i]["system_id"]),
        )

        rotated = ordered[1:] + ordered[:1]

        for dst_i, src_i in zip(ordered, rotated):
            out[dst_i]["final_output"] = rows[src_i]["final_output"]

    return out


class RankMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).reshape(-1)


def system_rank_loss(scores, rows, groups):
    losses = []

    for sid, idxs in groups.items():
        pos = [
            i for i in idxs
            if int(rows[i]["label"]) == 1
        ]
        neg = [
            i for i in idxs
            if int(rows[i]["label"]) == 0
        ]

        if not pos or not neg:
            continue

        sp = scores[pos].reshape(-1, 1)
        sn = scores[neg].reshape(1, -1)

        # RankNet: positive role should outrank negative role.
        losses.append(
            F.softplus(sn - sp).mean()
        )

    if not losses:
        return scores.sum() * 0.0

    # Equal mass per system.
    return torch.stack(losses).mean()


def balanced_system_bce(scores, rows, groups):
    losses = []

    for sid, idxs in groups.items():
        pos = [
            i for i in idxs
            if int(rows[i]["label"]) == 1
        ]
        neg = [
            i for i in idxs
            if int(rows[i]["label"]) == 0
        ]

        if not pos or not neg:
            continue

        lp = F.softplus(-scores[pos]).mean()
        ln = F.softplus(scores[neg]).mean()

        # Positive and negative mass balanced inside each system.
        losses.append(0.5 * (lp + ln))

    if not losses:
        return scores.sum() * 0.0

    return torch.stack(losses).mean()


def cf_rank_loss(scores, pairs):
    if not pairs:
        return scores.sum() * 0.0

    vals = [
        F.softplus(scores[neg_i] - scores[pos_i])
        for pos_i, neg_i in pairs
    ]

    # Equal mass per matched pair.
    return torch.stack(vals).mean()


def grad_norm_for_component(
    model,
    Xt0,
    Xt1,
    t0_rows,
    t0_groups,
    t1_pairs,
    component,
):
    model.zero_grad(set_to_none=True)

    s0 = model(Xt0)
    s1 = model(Xt1)

    if component == "rank":
        loss = system_rank_loss(s0, t0_rows, t0_groups)
    elif component == "cf":
        loss = cf_rank_loss(s1, t1_pairs)
    elif component == "bce":
        loss = balanced_system_bce(s0, t0_rows, t0_groups)
    else:
        raise ValueError(component)

    loss.backward()

    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += float(
                torch.sum(p.grad.detach() ** 2).cpu()
            )

    return {
        "loss": float(loss.detach().cpu()),
        "grad_norm_unweighted": math.sqrt(sq),
    }


def train_mlp_one(
    seed,
    Xt0_np,
    Xt1_np,
    t0_rows,
    t0_groups,
    t1_pairs,
    use_cf,
    use_bce,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    Xt0 = torch.tensor(Xt0_np, dtype=torch.float32)
    Xt1 = torch.tensor(Xt1_np, dtype=torch.float32)

    model = RankMLP(Xt0.shape[1])

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        s0 = model(Xt0)
        s1 = model(Xt1)

        lrk = system_rank_loss(
            s0, t0_rows, t0_groups
        )

        lcf = cf_rank_loss(
            s1, t1_pairs
        )

        lbce = balanced_system_bce(
            s0, t0_rows, t0_groups
        )

        total = lrk

        if use_cf:
            total = total + CF_WEIGHT * lcf

        if use_bce:
            total = total + BCE_WEIGHT * lbce

        total.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 5.0
        )
        opt.step()

        if epoch in (1, 50, 100, 200, 400, EPOCHS):
            rec = {
                "epoch": epoch,
                "total": float(total.detach()),
                "rank": float(lrk.detach()),
                "cf": float(lcf.detach()),
                "bce": float(lbce.detach()),
            }
            history.append(rec)

    model.eval()

    diagnostics = {
        "rank": grad_norm_for_component(
            model, Xt0, Xt1,
            t0_rows, t0_groups, t1_pairs,
            "rank",
        ),
        "cf": grad_norm_for_component(
            model, Xt0, Xt1,
            t0_rows, t0_groups, t1_pairs,
            "cf",
        ),
        "bce": grad_norm_for_component(
            model, Xt0, Xt1,
            t0_rows, t0_groups, t1_pairs,
            "bce",
        ),
    }

    diagnostics["cf"]["weighted_grad_norm"] = (
        diagnostics["cf"]["grad_norm_unweighted"] *
        (CF_WEIGHT if use_cf else 0.0)
    )

    diagnostics["bce"]["weighted_grad_norm"] = (
        diagnostics["bce"]["grad_norm_unweighted"] *
        (BCE_WEIGHT if use_bce else 0.0)
    )

    diagnostics["rank"]["weighted_grad_norm"] = (
        diagnostics["rank"]["grad_norm_unweighted"]
    )

    return model, history, diagnostics


def fit_linear_rank(Xt0, rows, groups):
    scaler = StandardScaler()
    Z = scaler.fit_transform(Xt0)

    diffs = []

    for sid, idxs in groups.items():
        pos = [
            i for i in idxs
            if int(rows[i]["label"]) == 1
        ]
        neg = [
            i for i in idxs
            if int(rows[i]["label"]) == 0
        ]

        for pi in pos:
            for ni in neg:
                diffs.append(Z[pi] - Z[ni])

    Dpos = np.stack(diffs)

    D = np.vstack([Dpos, -Dpos])

    yy = np.concatenate([
        np.ones(len(Dpos), dtype=int),
        np.zeros(len(Dpos), dtype=int),
    ])

    clf = LogisticRegression(
        C=LINEAR_C,
        fit_intercept=False,
        class_weight="balanced",
        max_iter=5000,
        solver="lbfgs",
        random_state=0,
    )

    clf.fit(D, yy)

    return scaler, clf


def train_ensemble(
    name,
    Xall_train,
    Xt0,
    Xt1,
    Xval,
    Xshuf,
    t0_rows,
    t0_groups,
    t1_pairs,
    use_cf,
    use_bce,
):
    scaler = StandardScaler()
    scaler.fit(Xall_train)

    z0 = scaler.transform(Xt0).astype(np.float32)
    z1 = scaler.transform(Xt1).astype(np.float32)
    zv = scaler.transform(Xval).astype(np.float32)
    zs = scaler.transform(Xshuf).astype(np.float32)

    zv_t = torch.tensor(zv, dtype=torch.float32)
    zs_t = torch.tensor(zs, dtype=torch.float32)

    val_scores = []
    shuf_scores = []
    histories = {}
    diagnostics = {}

    for seed in SEEDS:
        print(
            f"[{name}] seed={seed} "
            f"cf={use_cf} bce={use_bce}",
            flush=True,
        )

        model, hist, diag = train_mlp_one(
            seed,
            z0,
            z1,
            t0_rows,
            t0_groups,
            t1_pairs,
            use_cf,
            use_bce,
        )

        histories[str(seed)] = hist
        diagnostics[str(seed)] = diag

        with torch.no_grad():
            val_scores.append(
                model(zv_t).cpu().numpy()
            )
            shuf_scores.append(
                model(zs_t).cpu().numpy()
            )

        torch.save(
            {
                "state_dict": model.state_dict(),
                "feature_names": FEATURE_NAMES,
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "seed": seed,
                "use_cf": use_cf,
                "use_bce": use_bce,
            },
            OUT / f"{name}_seed{seed}.pt",
        )

    val_ensemble = np.mean(
        np.stack(val_scores), axis=0
    )
    shuf_ensemble = np.mean(
        np.stack(shuf_scores), axis=0
    )

    return {
        "val_scores": val_ensemble,
        "shuffled_scores": shuf_ensemble,
        "histories": histories,
        "gradient_diagnostics": diagnostics,
    }


def main():
    if (OUT / "summary.json").exists():
        raise SystemExit(
            f"{OUT}/summary.json already exists; "
            "refusing to overwrite."
        )

    OUT.mkdir(parents=True, exist_ok=True)

    train = read_jsonl(TRAIN_PATH)
    val = read_jsonl(VAL_PATH)

    t0 = [
        x for x in train
        if str(x["probe_family"]) == "T0"
    ]
    t1 = [
        x for x in train
        if str(x["probe_family"]) == "T1"
    ]

    if len(t0) != 720:
        raise RuntimeError(f"Expected 720 T0 rows, got {len(t0)}")

    if len(t1) != 190:
        raise RuntimeError(f"Expected 190 T1 rows, got {len(t1)}")

    if len(val) != 760:
        raise RuntimeError(f"Expected 760 VAL rows, got {len(val)}")

    t0_groups = group_indices(t0, "system_id")
    t1_pairs = build_t1_pairs(t1)

    if len(t0_groups) != 36:
        raise RuntimeError(
            f"Expected 36 T0 systems, got {len(t0_groups)}"
        )

    if len(t1_pairs) != 95:
        raise RuntimeError(
            f"Expected 95 T1 pairs, got {len(t1_pairs)}"
        )

    train_systems = {
        str(x["system_id"]) for x in train
    }
    val_systems = {
        str(x["system_id"]) for x in val
    }

    overlap = train_systems & val_systems
    if overlap:
        raise RuntimeError(
            f"TRAIN/VAL system leakage: {sorted(overlap)[:5]}"
        )

    print("=== SENSOR RANK-FIRST V2 ===", flush=True)
    print("T0 rows/systems:", len(t0), len(t0_groups), flush=True)
    print("T1 rows/pairs   :", len(t1), len(t1_pairs), flush=True)
    print("VAL rows/systems:", len(val), len(val_systems), flush=True)
    print("Internal TEST is NOT loaded.", flush=True)

    print("\nBuilding FULL geometry...", flush=True)

    X0_full = make_matrix(t0, "full")
    X1_full = make_matrix(t1, "full")
    Xv_full = make_matrix(val, "full")

    shuf_val = shuffled_output_rows(val)
    Xv_shuf = make_matrix(shuf_val, "full")

    print("\nBuilding QUERY-ONLY geometry...", flush=True)

    X0_q = make_matrix(t0, "query_only")
    X1_q = make_matrix(t1, "query_only")
    Xv_q = make_matrix(val, "query_only")

    if X0_full.shape[1] != 20:
        raise RuntimeError(
            f"Expected 20 geom features, got {X0_full.shape}"
        )

    print(
        "embedding cache entries:",
        len(embedding_cache),
        flush=True,
    )

    # ----------------------------------------------------------
    # 1. Linear system-rank baseline.
    # ----------------------------------------------------------

    print("\n=== LINEAR SYSTEM-RANK BASELINE ===", flush=True)

    linear_scaler, linear_clf = fit_linear_rank(
        X0_full,
        t0,
        t0_groups,
    )

    linear_val_scores = linear_clf.decision_function(
        linear_scaler.transform(Xv_full)
    )

    linear_metrics = evaluate(
        val,
        linear_val_scores,
    )

    print(json.dumps(
        linear_metrics,
        indent=2,
    ), flush=True)

    # ----------------------------------------------------------
    # 2. MLP rank only.
    # ----------------------------------------------------------

    Xall_full = np.concatenate(
        [X0_full, X1_full],
        axis=0,
    )

    rank_only = train_ensemble(
        "mlp_rank_only",
        Xall_full,
        X0_full,
        X1_full,
        Xv_full,
        Xv_shuf,
        t0,
        t0_groups,
        t1_pairs,
        use_cf=False,
        use_bce=False,
    )

    # ----------------------------------------------------------
    # 3. MLP rank + matched CF.
    # ----------------------------------------------------------

    rank_cf = train_ensemble(
        "mlp_rank_cf",
        Xall_full,
        X0_full,
        X1_full,
        Xv_full,
        Xv_shuf,
        t0,
        t0_groups,
        t1_pairs,
        use_cf=True,
        use_bce=False,
    )

    # ----------------------------------------------------------
    # 4. MLP full objective.
    # ----------------------------------------------------------

    full = train_ensemble(
        "mlp_full",
        Xall_full,
        X0_full,
        X1_full,
        Xv_full,
        Xv_shuf,
        t0,
        t0_groups,
        t1_pairs,
        use_cf=True,
        use_bce=True,
    )

    # ----------------------------------------------------------
    # 5. Query-only control, same architecture/objective.
    # ----------------------------------------------------------

    Xall_q = np.concatenate(
        [X0_q, X1_q],
        axis=0,
    )

    query_only = train_ensemble(
        "mlp_query_only_full_objective",
        Xall_q,
        X0_q,
        X1_q,
        Xv_q,
        Xv_q,  # no response exists to shuffle
        t0,
        t0_groups,
        t1_pairs,
        use_cf=True,
        use_bce=True,
    )

    results = {
        "linear_rank_full_geom":
            linear_metrics,

        "mlp_rank_only":
            evaluate(val, rank_only["val_scores"]),

        "mlp_rank_plus_cf":
            evaluate(val, rank_cf["val_scores"]),

        "mlp_full":
            evaluate(val, full["val_scores"]),

        "query_only_control":
            evaluate(val, query_only["val_scores"]),

        "full_model_shuffled_output_control":
            evaluate(val, full["shuffled_scores"]),
    }

    summary = {
        "version": "sensor_rankfirst_v2",
        "internal_test_used": False,

        "data": {
            "train_T0_rows": len(t0),
            "train_T0_systems": len(t0_groups),
            "train_T1_rows": len(t1),
            "train_T1_pairs": len(t1_pairs),
            "val_rows": len(val),
            "val_systems": len(val_systems),
        },

        "representation": {
            "type": "exact_20D_geometry_from_generic_fingerprint_v2",
            "features": FEATURE_NAMES,
        },

        "objective": {
            "system_rank":
                "mean_system mean_pos_neg softplus(s_neg-s_pos)",
            "matched_cf":
                "mean_pair softplus(s_negative-s_positive)",
            "balanced_bce":
                "mean_system 0.5*(positive BCE + negative BCE)",
            "cf_weight": CF_WEIGHT,
            "bce_weight": BCE_WEIGHT,
        },

        "mlp": {
            "shape": "20->64->32->1",
            "activation": "GELU",
            "dropout": DROPOUT,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "seeds": SEEDS,
            "ensemble": "mean logits across seeds",
        },

        "results": results,

        "gradient_diagnostics": {
            "rank_only": rank_only["gradient_diagnostics"],
            "rank_plus_cf": rank_cf["gradient_diagnostics"],
            "full": full["gradient_diagnostics"],
            "query_only": query_only["gradient_diagnostics"],
        },

        "histories": {
            "rank_only": rank_only["histories"],
            "rank_plus_cf": rank_cf["histories"],
            "full": full["histories"],
            "query_only": query_only["histories"],
        },
    }

    write_json(OUT / "summary.json", summary)

    score_rows = []

    for i, row in enumerate(val):
        score_rows.append({
            "system_id": row["system_id"],
            "target_role": row["target_role"],
            "label": int(row["label"]),
            "linear_rank": float(linear_val_scores[i]),
            "mlp_rank_only": float(rank_only["val_scores"][i]),
            "mlp_rank_plus_cf": float(rank_cf["val_scores"][i]),
            "mlp_full": float(full["val_scores"][i]),
            "query_only": float(query_only["val_scores"][i]),
            "full_shuffled_output": float(full["shuffled_scores"][i]),
        })

    with (OUT / "val_scores.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for row in score_rows:
            f.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    print("\n" + "=" * 90)
    print("FINAL VAL COMPARISON")
    print("=" * 90)

    for name, m in results.items():
        print(
            f"{name:<36} "
            f"sysAUC={m['mean_within_system_auc']:.4f} "
            f"roleAUC={m['mean_role_auc']:.4f} "
            f"pair={m['mean_system_pair_accuracy']:.4f} "
            f"perfect={m['perfect_order_rate']:.4f} "
            f"topK*={m['exact_topK_oracleK_rate']:.4f} "
            f"margin={m['mean_full_separation_margin']:+.4f}"
        )

    print()
    print("* topK uses TRUE K and is therefore role-ranking evaluation only.")
    print("Internal TEST untouched.")
    print("Summary:", OUT / "summary.json")


if __name__ == "__main__":
    main()
