#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
)
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset


SEED = 20261008

GROUP_ROOT = Path("artifacts/q38_selector_v41_data")
OUT = Path("artifacts/activation_predictor_v1")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH = 256
LR = 2e-4
EPOCHS = 40
PATIENCE = 6


# ------------------------------------------------------------------
# Reuse EXACT alignment code from the activation-mediator audit.
# ------------------------------------------------------------------

SPEC = importlib.util.spec_from_file_location(
    "med",
    "analysis/oracle_activation_mediator_audit_v1.py",
)
med = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(med)

SPEC2 = importlib.util.spec_from_file_location(
    "fwd",
    "analysis/forward_evidence_eig_v1.py",
)
fwd = importlib.util.module_from_spec(SPEC2)
SPEC2.loader.exec_module(fwd)

ROLES = list(med.ROLES)
R = len(ROLES)


def read_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


# ------------------------------------------------------------------
# Features
# ------------------------------------------------------------------

def make_feature(
    row,
    qmap,
    hmap,
    *,
    use_z,
):
    parts = [
        qmap[row["query"]],
        hmap[row["history_text"]],
        row["belief"],
        np.asarray(
            [row["round"] / 8.0],
            dtype=np.float32,
        ),
    ]

    if use_z:
        parts.append(row["z"])

    return np.concatenate(parts).astype(np.float32)


def make_xy(
    rows,
    qmap,
    hmap,
    *,
    use_z,
):
    X = np.stack([
        make_feature(
            r,
            qmap,
            hmap,
            use_z=use_z,
        )
        for r in rows
    ]).astype(np.float32)

    Y = np.stack([
        r["a"]
        for r in rows
    ]).astype(np.float32)

    return X, Y


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

class ActivationNet(nn.Module):
    def __init__(self, d_in, n_roles):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_in, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(.12),

            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(.08),

            nn.Linear(256, n_roles),
        )

    def forward(self, x):
        return self.net(x)


def macro_ap(y, p):
    vals = []

    for j in range(y.shape[1]):
        yy = y[:, j]

        if len(np.unique(yy)) < 2:
            continue

        vals.append(
            average_precision_score(
                yy,
                p[:, j],
            )
        )

    return float(np.mean(vals)) if vals else float("nan")


def train_model(
    name,
    Xtr,
    Ytr,
    Xdev,
    Ydev,
):

    pos = Ytr.sum(axis=0)
    neg = len(Ytr) - pos

    pos_weight = (
        neg
        / np.maximum(pos, 1)
    )

    # Avoid a very rare role dominating gradients.
    pos_weight = np.clip(
        pos_weight,
        1.0,
        12.0,
    ).astype(np.float32)

    model = ActivationNet(
        Xtr.shape[1],
        Ytr.shape[1],
    ).to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight,
            device=DEVICE,
        )
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-3,
    )

    ds = TensorDataset(
        torch.tensor(Xtr),
        torch.tensor(Ytr),
    )

    dl = DataLoader(
        ds,
        batch_size=BATCH,
        shuffle=True,
        num_workers=0,
    )

    Xd = torch.tensor(
        Xdev,
        device=DEVICE,
    )

    best_state = None
    best_ap = -1.0
    bad = 0

    for epoch in range(1, EPOCHS + 1):

        model.train()
        losses = []

        for xb, yb in dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            opt.zero_grad(set_to_none=True)

            logits = model(xb)

            loss = loss_fn(
                logits,
                yb,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            opt.step()

            losses.append(
                float(loss.detach().cpu())
            )

        model.eval()

        with torch.no_grad():
            pd = torch.sigmoid(
                model(Xd)
            ).cpu().numpy()

        dap = macro_ap(
            Ydev,
            pd,
        )

        print(
            f"[{name}] "
            f"epoch={epoch:02d} "
            f"loss={np.mean(losses):.5f} "
            f"dev_macro_AP={dap:.5f}",
            flush=True,
        )

        if dap > best_ap + 1e-5:
            best_ap = dap

            best_state = {
                k:
                    v.detach()
                    .cpu()
                    .clone()
                for k, v in model.state_dict().items()
            }

            bad = 0

        else:
            bad += 1

        if bad >= PATIENCE:
            break

    model.load_state_dict(
        best_state
    )

    model.eval()

    return model, best_ap


@torch.no_grad()
def predict(model, X, batch=1024):

    out = []

    for i in range(0, len(X), batch):

        xb = torch.tensor(
            X[i:i+batch],
            device=DEVICE,
        )

        out.append(
            torch.sigmoid(
                model(xb)
            ).cpu().numpy()
        )

    return np.concatenate(
        out,
        axis=0,
    )


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def choose_threshold(y, p):

    best = None

    for t in np.arange(
        .05,
        .951,
        .01,
    ):
        pred = (
            p >= t
        ).astype(int)

        f = f1_score(
            y.reshape(-1),
            pred.reshape(-1),
            zero_division=0,
        )

        candidate = (
            float(f),
            float(t),
        )

        if (
            best is None
            or candidate[0] > best[0]
        ):
            best = candidate

    return best[1]


def evaluate(
    y,
    p,
    threshold,
):

    per_role = {}

    aucs = []
    aps = []
    prevalences = []

    for j, role in enumerate(ROLES):

        yy = y[:, j]
        pp = p[:, j]

        prevalences.append(
            float(np.mean(yy))
        )

        if len(np.unique(yy)) < 2:
            continue

        auc = float(
            roc_auc_score(
                yy,
                pp,
            )
        )

        ap = float(
            average_precision_score(
                yy,
                pp,
            )
        )

        aucs.append(auc)
        aps.append(ap)

        per_role[role] = {
            "auc": auc,
            "ap": ap,
            "prevalence":
                float(np.mean(yy)),
        }

    pred = (
        p >= threshold
    ).astype(int)

    micro_f1 = float(
        f1_score(
            y.reshape(-1),
            pred.reshape(-1),
            zero_division=0,
        )
    )

    role_f1 = []

    for j in range(R):
        role_f1.append(
            f1_score(
                y[:, j],
                pred[:, j],
                zero_division=0,
            )
        )

    try:
        micro_auc = float(
            roc_auc_score(
                y.reshape(-1),
                p.reshape(-1),
            )
        )
    except Exception:
        micro_auc = None

    micro_ap = float(
        average_precision_score(
            y.reshape(-1),
            p.reshape(-1),
        )
    )

    prevalence = float(
        np.mean(y)
    )

    return {
        "macro_auroc":
            float(np.mean(aucs)),

        "macro_ap":
            float(np.mean(aps)),

        "micro_auroc":
            micro_auc,

        "micro_ap":
            micro_ap,

        "micro_f1":
            micro_f1,

        "macro_f1":
            float(
                np.mean(role_f1)
            ),

        "threshold":
            float(threshold),

        "positive_prevalence":
            prevalence,

        "macro_role_prevalence":
            float(
                np.mean(prevalences)
            ),

        "ap_lift_over_prevalence":
            float(
                np.mean(aps)
                - np.mean(prevalences)
            ),

        "per_role":
            per_role,
    }


def system_bootstrap_brier(
    rows,
    y,
    p,
    B=20000,
):

    per_system = defaultdict(list)

    row_brier = np.mean(
        (p - y) ** 2,
        axis=1,
    )

    for r, b in zip(
        rows,
        row_brier,
    ):
        per_system[
            r["system_id"]
        ].append(
            float(b)
        )

    systems = sorted(
        per_system
    )

    system_means = {
        sid:
            float(
                np.mean(
                    per_system[sid]
                )
            )
        for sid in systems
    }

    observed = float(
        np.mean(
            list(
                system_means.values()
            )
        )
    )

    rng = np.random.default_rng(
        SEED + 917
    )

    bs = np.empty(
        B,
        dtype=float,
    )

    vals = np.asarray(
        [
            system_means[s]
            for s in systems
        ],
        dtype=float,
    )

    for i in range(B):
        idx = rng.integers(
            0,
            len(vals),
            size=len(vals),
        )

        bs[i] = float(
            np.mean(
                vals[idx]
            )
        )

    lo, hi = np.quantile(
        bs,
        [.025, .975],
    )

    return {
        "system_mean_brier":
            observed,

        "cluster_ci95":
            [
                float(lo),
                float(hi),
            ],

        "systems":
            len(systems),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("DEVICE =", DEVICE)
    print("ROLES =", R)
    print("NO API CALLS")
    print()

    train_groups = read_jsonl(
        GROUP_ROOT / "train_groups.jsonl"
    )

    val_groups = read_jsonl(
        GROUP_ROOT / "val_groups.jsonl"
    )

    # Same exact raw alignment logic as mediator experiment.
    index = med.build_index()

    train_rows = med.attach(
        train_groups,
        index,
    )

    val_rows = med.attach(
        val_groups,
        index,
    )

    # --------------------------------------------
    # Sanity: actual activations must be installed.
    # --------------------------------------------

    violations = 0

    for r in (
        train_rows
        + val_rows
    ):
        if np.any(
            r["a"]
            > r["z"]
            + 1e-8
        ):
            violations += 1

    print(
        "ACTIVATION_NOT_SUBSET_OF_Z =",
        violations,
    )

    if violations:
        raise RuntimeError(
            "Activation labels violate installed-role truth."
        )

    # --------------------------------------------
    # EXACT same system split used before.
    # --------------------------------------------

    train_sids = sorted({
        r["system_id"]
        for r in train_rows
    })

    rng = random.Random(
        SEED
    )

    rng.shuffle(
        train_sids
    )

    n_dev = max(
        1,
        int(
            .15
            * len(train_sids)
        ),
    )

    dev_sids = set(
        train_sids[:n_dev]
    )

    fit_sids = set(
        train_sids[n_dev:]
    )

    fit_rows = [
        r
        for r in train_rows
        if r["system_id"]
        in fit_sids
    ]

    dev_rows = [
        r
        for r in train_rows
        if r["system_id"]
        in dev_sids
    ]

    print(
        "FIT systems/rows =",
        len(fit_sids),
        len(fit_rows),
    )

    print(
        "DEV systems/rows =",
        len(dev_sids),
        len(dev_rows),
    )

    print(
        "FINAL VAL systems/rows =",
        len({
            r["system_id"]
            for r in val_rows
        }),
        len(val_rows),
    )

    # --------------------------------------------
    # Text embeddings.
    # --------------------------------------------

    encoder = SentenceTransformer(
        fwd.ENCODER_NAME,
        device=DEVICE,
    )

    qmap, hmap = fwd.encode_maps(
        encoder,
        fit_rows
        + dev_rows
        + val_rows,
    )

    del encoder

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    configs = {
        "A0_PUBLIC": False,
        "A1_PLUS_TRUE_INSTALLED_Z": True,
    }

    results = {}
    saved = {}

    for name, use_z in configs.items():

        print()
        print("=" * 100)
        print(name)
        print("=" * 100)

        Xfit, Yfit = make_xy(
            fit_rows,
            qmap,
            hmap,
            use_z=use_z,
        )

        Xdev, Ydev = make_xy(
            dev_rows,
            qmap,
            hmap,
            use_z=use_z,
        )

        Xval, Yval = make_xy(
            val_rows,
            qmap,
            hmap,
            use_z=use_z,
        )

        model, best_ap = train_model(
            name,
            Xfit,
            Yfit,
            Xdev,
            Ydev,
        )

        pdev = predict(
            model,
            Xdev,
        )

        pval = predict(
            model,
            Xval,
        )

        threshold = choose_threshold(
            Ydev,
            pdev,
        )

        dev_metrics = evaluate(
            Ydev,
            pdev,
            threshold,
        )

        val_metrics = evaluate(
            Yval,
            pval,
            threshold,
        )

        val_metrics["brier_bootstrap"] = (
            system_bootstrap_brier(
                val_rows,
                Yval,
                pval,
            )
        )

        results[name] = {
            "best_internal_dev_macro_ap":
                float(best_ap),

            "dev":
                dev_metrics,

            "final_val":
                val_metrics,
        }

        torch.save(
            {
                "state_dict":
                    model.state_dict(),

                "input_dim":
                    Xfit.shape[1],

                "roles":
                    ROLES,

                "use_installed_z":
                    use_z,

                "threshold":
                    threshold,

                "encoder":
                    fwd.ENCODER_NAME,
            },
            OUT / f"{name}.pt",
        )

        saved[name] = {
            "pval": pval,
            "yval": Yval,
        }

        print(
            "\nFINAL VAL:"
        )

        print(
            json.dumps(
                val_metrics,
                indent=2,
            )
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    a0 = results[
        "A0_PUBLIC"
    ]["final_val"]

    a1 = results[
        "A1_PLUS_TRUE_INSTALLED_Z"
    ]["final_val"]

    comparison = {
        "A1_minus_A0": {
            "delta_macro_auroc":
                a1["macro_auroc"]
                - a0["macro_auroc"],

            "delta_macro_ap":
                a1["macro_ap"]
                - a0["macro_ap"],

            "delta_micro_f1":
                a1["micro_f1"]
                - a0["micro_f1"],

            "delta_micro_ap":
                a1["micro_ap"]
                - a0["micro_ap"],
        }
    }

    # --------------------------------------------
    # Decision.
    # --------------------------------------------

    a0_auc = a0[
        "macro_auroc"
    ]

    a1_auc = a1[
        "macro_auroc"
    ]

    delta_auc = (
        a1_auc
        - a0_auc
    )

    if (
        a0_auc >= .75
        and
        a0["macro_ap"]
        >= (
            a0["macro_role_prevalence"]
            + .15
        )
    ):
        verdict = (
            "PUBLIC_ACTIVATION_PREDICTABLE"
        )

    elif (
        a1_auc >= .75
        and
        delta_auc >= .03
    ):
        verdict = (
            "Z_CONDITIONING_NEEDED"
        )

    elif (
        a0_auc >= .68
        or
        a1_auc >= .72
    ):
        verdict = (
            "MODERATE_SIGNAL"
        )

    else:
        verdict = (
            "WEAK_CROSS_SYSTEM_ACTIVATION_SIGNAL"
        )

    summary = {
        "experiment":
            "activation_predictor_v1",

        "selector_truth_usage":
            False,

        "A0_test_inputs":
            [
                "candidate query",
                "public history",
                "current role belief",
                "round",
            ],

        "A1_test_inputs":
            [
                "candidate query",
                "public history",
                "current role belief",
                "round",
                "TRUE installed Z (oracle diagnostic only)",
            ],

        "fit_systems":
            len(fit_sids),

        "internal_dev_systems":
            len(dev_sids),

        "final_val_systems":
            len({
                r["system_id"]
                for r in val_rows
            }),

        "fit_rows":
            len(fit_rows),

        "internal_dev_rows":
            len(dev_rows),

        "final_val_rows":
            len(val_rows),

        "models":
            results,

        "comparison":
            comparison,

        "decision": {
            "verdict":
                verdict,

            "interpretation":
                (
                    "A0 strong => activation is predictable from "
                    "public state/query. A1 much stronger than A0 => "
                    "installed-role hypothesis Z is needed before "
                    "routing prediction."
                ),
        },
    }

    (
        OUT / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("FINAL ACTIVATION PREDICTOR AUDIT")
    print("=" * 100)

    for name in configs:
        m = results[
            name
        ]["final_val"]

        print(
            f"{name:30} "
            f"macroAUC={m['macro_auroc']:.4f} "
            f"macroAP={m['macro_ap']:.4f} "
            f"microAP={m['micro_ap']:.4f} "
            f"microF1={m['micro_f1']:.4f}"
        )

    print()
    print(
        "A1 - A0 =",
        json.dumps(
            comparison[
                "A1_minus_A0"
            ],
            indent=2,
        )
    )

    print()
    print(
        "VERDICT =",
        verdict,
    )

    print(
        "SAVED =",
        OUT / "summary.json",
    )


if __name__ == "__main__":
    main()
