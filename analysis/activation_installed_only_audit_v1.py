#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
)


# ============================================================
# LOAD EXISTING IMPLEMENTATIONS
# ============================================================

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


act = load_module(
    "act",
    "analysis/activation_predictor_v1.py",
)

med = load_module(
    "med",
    "analysis/oracle_activation_mediator_audit_v1.py",
)

fwd = load_module(
    "fwd",
    "analysis/forward_evidence_eig_v1.py",
)


GROUP_ROOT = Path(
    "artifacts/q38_selector_v41_data"
)

MODEL_ROOT = Path(
    "artifacts/activation_predictor_v1"
)

OUT = Path(
    "artifacts/activation_installed_only_audit_v1"
)

ROLES = list(act.ROLES)
R = len(ROLES)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# METRICS
# ============================================================

def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return None

    return float(
        roc_auc_score(
            y,
            p,
        )
    )


def safe_ap(y, p):
    if len(np.unique(y)) < 2:
        return None

    return float(
        average_precision_score(
            y,
            p,
        )
    )


def installed_only_metrics(
    y,
    p,
    z,
    threshold,
):

    per_role = {}

    aucs = []
    aps = []
    f1s = []

    all_y = []
    all_p = []

    for j, role in enumerate(ROLES):

        mask = (
            z[:, j] > .5
        )

        yy = y[
            mask,
            j
        ]

        pp = p[
            mask,
            j
        ]

        if len(yy) == 0:
            continue

        auc = safe_auc(
            yy,
            pp,
        )

        ap = safe_ap(
            yy,
            pp,
        )

        pred = (
            pp >= threshold
        ).astype(int)

        f1 = float(
            f1_score(
                yy,
                pred,
                zero_division=0,
            )
        )

        if auc is not None:
            aucs.append(auc)

        if ap is not None:
            aps.append(ap)

        f1s.append(f1)

        all_y.extend(
            yy.tolist()
        )

        all_p.extend(
            pp.tolist()
        )

        per_role[role] = {
            "n_installed":
                int(len(yy)),

            "activation_prevalence_given_installed":
                float(np.mean(yy)),

            "auc":
                auc,

            "ap":
                ap,

            "f1":
                f1,
        }

    all_y = np.asarray(
        all_y,
        dtype=float,
    )

    all_p = np.asarray(
        all_p,
        dtype=float,
    )

    micro_auc = safe_auc(
        all_y,
        all_p,
    )

    micro_ap = safe_ap(
        all_y,
        all_p,
    )

    micro_f1 = float(
        f1_score(
            all_y,
            (
                all_p
                >= threshold
            ).astype(int),
            zero_division=0,
        )
    )

    return {
        "macro_auc":
            float(np.mean(aucs))
            if aucs
            else None,

        "macro_ap":
            float(np.mean(aps))
            if aps
            else None,

        "macro_f1":
            float(np.mean(f1s))
            if f1s
            else None,

        "micro_auc":
            micro_auc,

        "micro_ap":
            micro_ap,

        "micro_f1":
            micro_f1,

        "installed_pair_count":
            int(len(all_y)),

        "activation_prevalence_given_installed":
            float(np.mean(all_y)),

        "per_role":
            per_role,
    }


def top1_routing_metric(
    y,
    p,
    z,
):

    hits = []
    random_probs = []

    for i in range(
        len(y)
    ):

        installed = np.flatnonzero(
            z[i] > .5
        )

        if len(installed) == 0:
            continue

        active = np.flatnonzero(
            y[i] > .5
        )

        active_installed = np.intersect1d(
            installed,
            active,
        )

        if len(active_installed) == 0:
            continue

        best = installed[
            np.argmax(
                p[
                    i,
                    installed
                ]
            )
        ]

        hits.append(
            float(
                best
                in set(
                    active_installed.tolist()
                )
            )
        )

        random_probs.append(
            len(active_installed)
            / len(installed)
        )

    return {
        "states_with_activation":
            len(hits),

        "predicted_top1_hit":
            float(np.mean(hits)),

        "random_within_installed_expected":
            float(np.mean(random_probs)),

        "lift":
            float(
                np.mean(hits)
                - np.mean(random_probs)
            ),
    }


def membership_only_baseline(
    y,
    z,
):

    # Score = 1 if installed, 0 otherwise.
    # This shows how inflated the ordinary all-pair metric
    # can become from membership knowledge alone.

    flat_y = y.reshape(-1)
    flat_z = z.reshape(-1)

    return {
        "micro_auc":
            safe_auc(
                flat_y,
                flat_z,
            ),

        "micro_ap":
            safe_ap(
                flat_y,
                flat_z,
            ),

        "activation_prevalence":
            float(
                np.mean(
                    flat_y
                )
            ),

        "installed_prevalence":
            float(
                np.mean(
                    flat_z
                )
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "DEVICE =",
        DEVICE,
    )

    print(
        "NO API CALLS"
    )

    # --------------------------------------------------------
    # Same 84 held-out systems used in previous experiments.
    # --------------------------------------------------------

    val_groups = fwd.read_jsonl(
        GROUP_ROOT
        / "val_groups.jsonl"
    )

    index = med.build_index()

    val_rows = med.attach(
        val_groups,
        index,
    )

    print(
        "VAL rows =",
        len(val_rows),
    )

    print(
        "VAL systems =",
        len({
            r["system_id"]
            for r in val_rows
        }),
    )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    encoder = SentenceTransformer(
        fwd.ENCODER_NAME,
        device=DEVICE,
    )

    qmap, hmap = fwd.encode_maps(
        encoder,
        val_rows,
    )

    del encoder

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    Y = np.stack([
        r["a"]
        for r in val_rows
    ]).astype(
        np.float32
    )

    Z = np.stack([
        r["z"]
        for r in val_rows
    ]).astype(
        np.float32
    )

    results = {}

    for name, use_z in [
        (
            "A0_PUBLIC",
            False,
        ),
        (
            "A1_PLUS_TRUE_INSTALLED_Z",
            True,
        ),
    ]:

        ckpt_path = (
            MODEL_ROOT
            / f"{name}.pt"
        )

        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
        )

        model = act.ActivationNet(
            ckpt[
                "input_dim"
            ],
            R,
        ).to(
            DEVICE
        )

        model.load_state_dict(
            ckpt[
                "state_dict"
            ]
        )

        model.eval()

        X, Y2 = act.make_xy(
            val_rows,
            qmap,
            hmap,
            use_z=use_z,
        )

        if not np.array_equal(
            Y,
            Y2,
        ):
            raise RuntimeError(
                "Activation target mismatch"
            )

        P = act.predict(
            model,
            X,
        )

        threshold = float(
            ckpt[
                "threshold"
            ]
        )

        conditional = (
            installed_only_metrics(
                Y,
                P,
                Z,
                threshold,
            )
        )

        routing = (
            top1_routing_metric(
                Y,
                P,
                Z,
            )
        )

        absent_mask = (
            Z < .5
        )

        installed_mask = (
            Z > .5
        )

        results[name] = {
            "threshold":
                threshold,

            "installed_only":
                conditional,

            "within_installed_top1":
                routing,

            "mean_prediction_absent_roles":
                float(
                    np.mean(
                        P[
                            absent_mask
                        ]
                    )
                ),

            "mean_prediction_installed_roles":
                float(
                    np.mean(
                        P[
                            installed_mask
                        ]
                    )
                ),
        }

        print()
        print(
            "=" * 100
        )

        print(name)

        print(
            "=" * 100
        )

        print(
            json.dumps(
                results[name],
                indent=2,
            )
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    baseline = (
        membership_only_baseline(
            Y,
            Z,
        )
    )

    a0 = results[
        "A0_PUBLIC"
    ][
        "installed_only"
    ]

    a1 = results[
        "A1_PLUS_TRUE_INSTALLED_Z"
    ][
        "installed_only"
    ]

    comparison = {
        "delta_installed_macro_auc":
            (
                a1["macro_auc"]
                - a0["macro_auc"]
            )
            if (
                a1["macro_auc"]
                is not None
                and
                a0["macro_auc"]
                is not None
            )
            else None,

        "delta_installed_macro_ap":
            (
                a1["macro_ap"]
                - a0["macro_ap"]
            )
            if (
                a1["macro_ap"]
                is not None
                and
                a0["macro_ap"]
                is not None
            )
            else None,

        "delta_installed_micro_auc":
            (
                a1["micro_auc"]
                - a0["micro_auc"]
            )
            if (
                a1["micro_auc"]
                is not None
                and
                a0["micro_auc"]
                is not None
            )
            else None,

        "delta_installed_micro_ap":
            (
                a1["micro_ap"]
                - a0["micro_ap"]
            )
            if (
                a1["micro_ap"]
                is not None
                and
                a0["micro_ap"]
                is not None
            )
            else None,
    }

    top1 = results[
        "A1_PLUS_TRUE_INSTALLED_Z"
    ][
        "within_installed_top1"
    ]

    a1_auc = (
        a1["macro_auc"]
        if a1["macro_auc"]
        is not None
        else 0.0
    )

    if (
        a1_auc >= .80
        and
        top1["lift"] >= .15
    ):
        verdict = (
            "STRONG_ROUTING_SIGNAL_BEYOND_MEMBERSHIP"
        )

    elif (
        a1_auc >= .70
        and
        top1["lift"] >= .05
    ):
        verdict = (
            "MODERATE_ROUTING_SIGNAL_BEYOND_MEMBERSHIP"
        )

    else:
        verdict = (
            "A1_GAIN_MOSTLY_MEMBERSHIP_FILTERING"
        )

    summary = {
        "experiment":
            "activation_installed_only_audit_v1",

        "final_val_systems":
            len({
                r["system_id"]
                for r in val_rows
            }),

        "final_val_rows":
            len(val_rows),

        "models":
            results,

        "membership_only_baseline":
            baseline,

        "comparison":
            comparison,

        "decision": {
            "verdict":
                verdict,

            "question":
                (
                    "Can A1 predict routing among roles "
                    "known to be installed, rather than merely "
                    "eliminate absent roles?"
                ),
        },
    }

    (
        OUT
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 100
    )

    print(
        "INSTALLED-ONLY SUMMARY"
    )

    print(
        "=" * 100
    )

    for name in (
        "A0_PUBLIC",
        "A1_PLUS_TRUE_INSTALLED_Z",
    ):

        x = results[
            name
        ][
            "installed_only"
        ]

        t = results[
            name
        ][
            "within_installed_top1"
        ]

        print(
            f"{name:30} "
            f"macroAUC={x['macro_auc']:.4f} "
            f"macroAP={x['macro_ap']:.4f} "
            f"microAUC={x['micro_auc']:.4f} "
            f"microAP={x['micro_ap']:.4f} "
            f"top1={t['predicted_top1_hit']:.4f} "
            f"randomTop1={t['random_within_installed_expected']:.4f}"
        )

    print()
    print(
        "MEMBERSHIP-ONLY ALL-PAIR BASELINE =",
        baseline,
    )

    print()
    print(
        "VERDICT =",
        verdict,
    )

    print(
        "SAVED =",
        OUT
        / "summary.json",
    )


if __name__ == "__main__":
    main()
