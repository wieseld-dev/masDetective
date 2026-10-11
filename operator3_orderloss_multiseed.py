#!/usr/bin/env python3

import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

import operator3_poc as op


CACHE = Path("artifacts/operator3_poc/api_runs.jsonl")

NSEEDS = 20
DIM = 16
TRAIN_FRAC = 0.70

STEPS = 3000
LR = 3e-3
WEIGHT_DECAY = 1e-3

# Explicit weight of the structural/order loss.
ORDER_ALPHA = 5.0


def l2(x):
    return x / np.maximum(
        np.linalg.norm(x, axis=1, keepdims=True),
        1e-12
    )


def fit_model(
    q,
    targets,
    train_idx,
    device,
    use_order_loss=False,
):

    qtr = torch.tensor(
        q[train_idx],
        dtype=torch.float32,
        device=device
    )

    pair_targets = {
        c: torch.tensor(
            targets[c][train_idx],
            dtype=torch.float32,
            device=device
        )
        for c in op.PAIRS
    }

    triple_targets = {
        c: torch.tensor(
            targets[c][train_idx],
            dtype=torch.float32,
            device=device
        )
        for c in op.TRIPLES
    }

    model = op.Operator3(q.shape[1]).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best = float("inf")
    best_state = None

    for step in range(STEPS):

        optimizer.zero_grad(set_to_none=True)

        # ---------------------------------------
        # 1. Ordinary pairwise response prediction
        # ---------------------------------------
        pred_loss = 0.0

        for c in op.PAIRS:
            pred = model.predict(qtr, c)

            pred_loss = pred_loss + (
                (pred - pair_targets[c]) ** 2
            ).mean()

        pred_loss = pred_loss / len(op.PAIRS)

        loss = pred_loss

        # ---------------------------------------
        # 2. Explicit execution-order loss
        #
        # Example:
        #
        # predicted:
        # W_C(W_BW_A - W_AW_B)q
        #
        # actual:
        # z_ABC - z_BAC
        # ---------------------------------------
        if use_order_loss:

            order_loss = 0.0

            for c1, c2 in op.SAME_TERMINAL:

                pred1 = model.predict(qtr, c1)
                pred2 = model.predict(qtr, c2)

                predicted_delta = pred1 - pred2

                actual_delta = (
                    triple_targets[c1]
                    - triple_targets[c2]
                )

                order_loss = order_loss + (
                    (predicted_delta - actual_delta) ** 2
                ).mean()

            order_loss = (
                order_loss / len(op.SAME_TERMINAL)
            )

            loss = (
                pred_loss
                + ORDER_ALPHA * order_loss
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0
        )

        optimizer.step()

        lv = float(loss.detach().cpu())

        if lv < best:
            best = lv
            best_state = {
                k: v.detach().cpu().clone()
                for k, v
                in model.state_dict().items()
            }

    model.load_state_dict(best_state)

    return model, best


def comm_metrics(model, q, z, idx):

    device = next(model.parameters()).device
    W = model.W()

    predicted = []
    actual = []

    # Same final agent, only earlier order changes.
    tests = [
        (
            "ABC",
            "BAC",
            W["C"] @ (
                W["B"] @ W["A"]
                - W["A"] @ W["B"]
            )
        ),
        (
            "ACB",
            "CAB",
            W["B"] @ (
                W["C"] @ W["A"]
                - W["A"] @ W["C"]
            )
        ),
        (
            "BCA",
            "CBA",
            W["A"] @ (
                W["C"] @ W["B"]
                - W["B"] @ W["C"]
            )
        ),
    ]

    with torch.no_grad():

        for c1, c2, K in tests:

            K = K.detach().cpu().numpy()

            for i in idx:

                qi = q[i]

                predicted.append(
                    float(
                        np.sum((K @ qi) ** 2)
                    )
                )

                actual.append(
                    float(
                        np.sum(
                            (
                                z[c1][i]
                                - z[c2][i]
                            ) ** 2
                        )
                    )
                )

    predicted = np.asarray(predicted)
    actual = np.asarray(actual)

    rho, p = spearmanr(
        predicted,
        actual
    )

    k = max(
        1,
        math.ceil(0.20 * len(actual))
    )

    top = np.argsort(predicted)[-k:]

    lift = float(
        actual[top].mean()
        / max(actual.mean(), 1e-12)
    )

    return float(rho), float(p), lift


def same_terminal_accuracy(model, q, z, idx):

    device = next(model.parameters()).device

    tq = torch.tensor(
        q,
        dtype=torch.float32,
        device=device
    )

    pred = {}

    with torch.no_grad():

        for c in op.TRIPLES:

            pred[c] = (
                model.predict(tq, c)
                .cpu()
                .numpy()
            )

    correct = 0
    total = 0

    for c1, c2 in op.SAME_TERMINAL:

        for i in idx:

            # Test both possible truths.
            for truth in [c1, c2]:

                target = z[truth][i]

                d1 = np.sum(
                    (target - pred[c1][i]) ** 2
                )

                d2 = np.sum(
                    (target - pred[c2][i]) ** 2
                )

                guess = c1 if d1 < d2 else c2

                correct += int(
                    guess == truth
                )

                total += 1

    return correct / total


# =========================================================
# LOAD CACHE
# =========================================================

rows = []

with CACHE.open(encoding="utf-8") as f:

    for line in f:

        if line.strip():
            rows.append(json.loads(line))

rows.sort(
    key=lambda r: int(r["id"])
)

print("Cached probes:", len(rows))

tasks = [
    r["task"]
    for r in rows
]

all_comps = (
    ["A", "B", "C"]
    + op.PAIRS
    + op.TRIPLES
)

texts = {
    c: [
        r["outputs"][c]
        for r in rows
    ]
    for c in all_comps
}


# =========================================================
# EMBED ONCE
# =========================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

encoder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=device
)

print("Encoding tasks...")

q0 = encoder.encode(
    tasks,
    normalize_embeddings=True,
    convert_to_numpy=True,
    batch_size=64
)

z0 = {}

for c in all_comps:

    print("Encoding", c)

    z0[c] = encoder.encode(
        texts[c],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=64
    )


# =========================================================
# MULTI-SEED
# =========================================================

results = []

for seed in range(1, NSEEDS + 1):

    np.random.seed(seed)
    torch.manual_seed(seed)

    ids = np.arange(len(rows))

    tr, te = train_test_split(
        ids,
        train_size=TRAIN_FRAC,
        random_state=seed,
        shuffle=True
    )

    # PCA fitted ONLY on train probes + PAIR outputs.
    pca_data = [q0[tr]]

    for c in op.PAIRS:
        pca_data.append(
            z0[c][tr]
        )

    pca_data = np.vstack(
        pca_data
    )

    pca = PCA(
        n_components=DIM,
        random_state=seed
    )

    pca.fit(pca_data)

    q = l2(
        pca.transform(q0)
    )

    z = {
        c: l2(
            pca.transform(z0[c])
        )
        for c in all_comps
    }

    # ---------------------------------------
    # BASELINE
    # ---------------------------------------

    base_model, _ = fit_model(
        q,
        z,
        tr,
        device,
        use_order_loss=False
    )

    brho, bp, blift = comm_metrics(
        base_model,
        q,
        z,
        te
    )

    bacc = same_terminal_accuracy(
        base_model,
        q,
        z,
        te
    )

    # ---------------------------------------
    # ORDER LOSS
    # ---------------------------------------

    order_model, _ = fit_model(
        q,
        z,
        tr,
        device,
        use_order_loss=True
    )

    orho, opval, olift = comm_metrics(
        order_model,
        q,
        z,
        te
    )

    oacc = same_terminal_accuracy(
        order_model,
        q,
        z,
        te
    )

    results.append([
        brho,
        blift,
        bacc,
        orho,
        olift,
        oacc
    ])

    print(
        f"seed={seed:02d} | "
        f"BASE rho={brho:+.3f} "
        f"lift={blift:.3f}x "
        f"acc={bacc*100:.1f}% | "
        f"ORDER rho={orho:+.3f} "
        f"lift={olift:.3f}x "
        f"acc={oacc*100:.1f}%"
    )


R = np.asarray(results)

print()
print(
    "========================================"
)
print(
    "MULTI-SEED SUMMARY"
)
print(
    "========================================"
)

print(f"Seeds: {NSEEDS}")
print(f"Order-loss alpha: {ORDER_ALPHA}")
print()

print(
    f"BASE rho  = "
    f"{R[:,0].mean():+.3f} "
    f"± {R[:,0].std(ddof=1):.3f}"
)

print(
    f"BASE lift = "
    f"{R[:,1].mean():.3f} "
    f"± {R[:,1].std(ddof=1):.3f}x"
)

print(
    f"BASE same-terminal accuracy = "
    f"{R[:,2].mean()*100:.1f}% "
    f"± {R[:,2].std(ddof=1)*100:.1f}%"
)

print()

print(
    f"ORDER rho  = "
    f"{R[:,3].mean():+.3f} "
    f"± {R[:,3].std(ddof=1):.3f}"
)

print(
    f"ORDER lift = "
    f"{R[:,4].mean():.3f} "
    f"± {R[:,4].std(ddof=1):.3f}x"
)

print(
    f"ORDER same-terminal accuracy = "
    f"{R[:,5].mean()*100:.1f}% "
    f"± {R[:,5].std(ddof=1)*100:.1f}%"
)

print()

print(
    "ORDER - BASE:"
)

print(
    f"rho improvement  = "
    f"{(R[:,3]-R[:,0]).mean():+.3f}"
)

print(
    f"lift improvement = "
    f"{(R[:,4]-R[:,1]).mean():+.3f}x"
)

print(
    f"accuracy improvement = "
    f"{(R[:,5]-R[:,2]).mean()*100:+.1f} pp"
)

print()

print(
    "ORDER better rho in "
    f"{np.mean(R[:,3] > R[:,0])*100:.1f}% "
    "of seeds"
)

print(
    "ORDER better lift in "
    f"{np.mean(R[:,4] > R[:,1])*100:.1f}% "
    "of seeds"
)

print(
    "ORDER better accuracy in "
    f"{np.mean(R[:,5] > R[:,2])*100:.1f}% "
    "of seeds"
)
