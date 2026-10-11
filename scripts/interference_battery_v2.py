from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, average_precision_score

# ---------------------------------------------------------------------
# Reuse the already-validated V1 loader and vector parsing.
# ---------------------------------------------------------------------

SPEC = importlib.util.spec_from_file_location(
    "ib1",
    "scripts/interference_battery_v1.py",
)
ib1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ib1)

ROLES = list(ib1.ROLES)
PAIR_TAGS = dict(ib1.PAIR_TAGS)
PAIR_NAMES = list(PAIR_TAGS)
STYLES = ["numbered", "requirements"]
EPS = 1e-12


def unit_center(v):
    v = np.asarray(v, dtype=float)
    v = v - np.mean(v)
    n = np.linalg.norm(v)
    if n < EPS:
        return np.zeros_like(v)
    return v / n


def equal_concat(a, b):
    """
    Each Judge channel receives exactly equal L2 weight.
    Prevents V13.4 or V13.5 scale from dominating cosine geometry.
    """
    return np.concatenate([
        unit_center(a),
        unit_center(b),
    ]) / np.sqrt(2.0)


def raw_concat(a, b):
    return np.concatenate([
        np.asarray(a, float),
        np.asarray(b, float),
    ])


def cosine(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na < EPS or nb < EPS:
        return np.nan

    return float(np.dot(a, b) / (na * nb))


def exact_signflip(vals):
    return ib1.exact_signflip(vals)


def tfidf_predict(train_texts, train_y, test_texts, alpha=30.0):
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=5000,
        sublinear_tf=True,
        stop_words="english",
    )

    Xtr = vec.fit_transform(train_texts)
    Xte = vec.transform(test_texts)

    model = Ridge(
        alpha=alpha,
        solver="lsqr",
    )
    model.fit(Xtr, train_y)

    return (
        model.predict(Xtr),
        model.predict(Xte),
    )


def route_vec(sel):
    ss = set(sel or ())
    return np.asarray(
        [float(r in ss) for r in ROLES],
        dtype=float,
    )


def build_rows(data, pair_records):
    rec = {
        (r["channel"], r["sid"], r["style"], r["tag"]): r
        for r in pair_records
    }

    design = data["design"]
    selected = data["selected"]

    basis = design["basis_queries"]

    muxq = {}
    for e in design["entries"]:
        muxq[
            (
                str(e["session_id"]),
                str(e["style"]),
                str(e["tag"]),
                str(e["order"]),
            )
        ] = str(e["query"])

    rows = []

    for sid in sorted(basis):
        for style in STYLES:
            for tag, (i, j) in PAIR_TAGS.items():

                k135 = ("v135", sid, style, tag)
                k134 = ("v134", sid, style, tag)

                if k135 not in rec or k134 not in rec:
                    continue

                a = rec[k135]
                b = rec[k134]

                qi = str(basis[sid][str(i)])
                qj = str(basis[sid][str(j)])

                pair_text = (
                    f"FIRST PROBE:\n{qi}\n\n"
                    f"SECOND PROBE:\n{qj}"
                )

                row = {
                    "sid": sid,
                    "style": style,
                    "tag": tag,
                    "i": i,
                    "j": j,
                    "pair_text": pair_text,

                    "anti_v135": np.asarray(a["anti"], float),
                    "anti_v134": np.asarray(b["anti"], float),
                    "resid_v135": np.asarray(a["resid"], float),
                    "resid_v134": np.asarray(b["resid"], float),
                }

                row["anti_concat_raw"] = raw_concat(
                    row["anti_v135"],
                    row["anti_v134"],
                )
                row["resid_concat_raw"] = raw_concat(
                    row["resid_v135"],
                    row["resid_v134"],
                )

                row["anti_concat_equal"] = equal_concat(
                    row["anti_v135"],
                    row["anti_v134"],
                )
                row["resid_concat_equal"] = equal_concat(
                    row["resid_v135"],
                    row["resid_v134"],
                )

                qf = muxq.get((sid, style, tag, "fwd"))
                qr = muxq.get((sid, style, tag, "rev"))

                si = selected.get((sid, qi))
                sj = selected.get((sid, qj))
                sf = selected.get((sid, qf)) if qf else None
                sr = selected.get((sid, qr)) if qr else None

                if (
                    si is not None
                    and sj is not None
                    and sf is not None
                    and sr is not None
                ):
                    zi = route_vec(si)
                    zj = route_vec(sj)
                    zf = route_vec(sf)
                    zr = route_vec(sr)

                    # Directional routing response to swapping task order.
                    row["route_order_delta"] = zf - zr

                    # Non-additive routing interaction.
                    row["route_sym_delta"] = (
                        0.5 * (zf + zr)
                        - 0.5 * (zi + zj)
                    )

                    row["route_labels_ok"] = True
                else:
                    row["route_labels_ok"] = False

                rows.append(row)

    return rows


# ---------------------------------------------------------------------
# Cross-wrapper signature invariance
# ---------------------------------------------------------------------

def cross_wrapper(rows, feature):
    N = {
        (r["sid"], r["tag"]): r
        for r in rows
        if r["style"] == "numbered"
    }

    R = {
        (r["sid"], r["tag"]): r
        for r in rows
        if r["style"] == "requirements"
    }

    common = sorted(set(N) & set(R))

    observed = [
        cosine(N[k][feature], R[k][feature])
        for k in common
    ]

    obs = float(np.nanmean(observed))

    rng = np.random.default_rng(22091)
    sids = sorted({s for s, _ in common})

    null = []

    for _ in range(20000):
        vals = []

        for sid in sids:
            tags = [
                t for t in PAIR_NAMES
                if (sid, t) in N and (sid, t) in R
            ]

            perm = list(rng.permutation(tags))

            for a, b in zip(tags, perm):
                vals.append(
                    cosine(
                        N[(sid, a)][feature],
                        R[(sid, b)][feature],
                    )
                )

        null.append(np.nanmean(vals))

    null = np.asarray(null)

    p = float(
        (1 + np.sum(null >= obs - 1e-12))
        / (1 + len(null))
    )

    ranks = []

    for sid in sids:
        tags = [
            t for t in PAIR_NAMES
            if (sid, t) in N and (sid, t) in R
        ]

        if len(tags) < 4:
            continue

        a = np.asarray([
            np.linalg.norm(N[(sid, t)][feature])
            for t in tags
        ])
        b = np.asarray([
            np.linalg.norm(R[(sid, t)][feature])
            for t in tags
        ])

        rho, _ = ib1.safe_spearman(a, b)
        ranks.append(rho)

    return {
        "n_pairs": len(common),
        "same_pair_cosine": obs,
        "pair_permutation_p": p,
        "mean_pair_magnitude_rank_rho": float(
            np.nanmean(ranks)
        ),
    }


# ---------------------------------------------------------------------
# Can semantics alone predict the interference vector?
# Strict leave-one-system-out.
# ---------------------------------------------------------------------

def query_predictability(rows, style, feature):
    rr = [r for r in rows if r["style"] == style]
    sids = sorted({r["sid"] for r in rr})

    cos_by_system = []
    all_sse = 0.0
    all_base_sse = 0.0

    for held in sids:
        tr = [r for r in rr if r["sid"] != held]
        te = [r for r in rr if r["sid"] == held]

        if not tr or not te:
            continue

        Ytr = np.stack([r[feature] for r in tr])
        Yte = np.stack([r[feature] for r in te])

        _, pred = tfidf_predict(
            [r["pair_text"] for r in tr],
            Ytr,
            [r["pair_text"] for r in te],
            alpha=30.0,
        )

        c = [
            cosine(p, y)
            for p, y in zip(pred, Yte)
        ]

        cos_by_system.append(float(np.nanmean(c)))

        mean_train = np.mean(Ytr, axis=0)

        all_sse += float(np.sum((pred - Yte) ** 2))
        all_base_sse += float(
            np.sum((Yte - mean_train) ** 2)
        )

    return {
        "n_systems": len(cos_by_system),
        "mean_heldout_cosine": float(
            np.nanmean(cos_by_system)
        ),
        "cosine_signflip_p": exact_signflip(
            cos_by_system
        ),
        "heldout_R2_vs_train_mean": float(
            1.0 - all_sse / max(all_base_sse, EPS)
        ),
    }


# ---------------------------------------------------------------------
# Remove query-predictable component with LOSO models.
# ---------------------------------------------------------------------

def crossfit_semantic_residuals(rows, style, feature):
    rr = [r for r in rows if r["style"] == style]
    sids = sorted({r["sid"] for r in rr})

    out = {}

    for held in sids:
        tr = [r for r in rr if r["sid"] != held]
        te = [r for r in rr if r["sid"] == held]

        if not tr or not te:
            continue

        Ytr = np.stack([r[feature] for r in tr])

        _, pred = tfidf_predict(
            [r["pair_text"] for r in tr],
            Ytr,
            [r["pair_text"] for r in te],
            alpha=30.0,
        )

        for r, p in zip(te, pred):
            out[(r["sid"], r["tag"])] = (
                np.asarray(r[feature]) - p
            )

    return out


def residual_cross_wrapper(rows, feature):
    N = crossfit_semantic_residuals(
        rows,
        "numbered",
        feature,
    )

    R = crossfit_semantic_residuals(
        rows,
        "requirements",
        feature,
    )

    common = sorted(set(N) & set(R))

    obsvals = [
        cosine(N[k], R[k])
        for k in common
    ]

    obs = float(np.nanmean(obsvals))

    rng = np.random.default_rng(22777)

    sids = sorted({s for s, _ in common})
    null = []

    for _ in range(20000):
        vals = []

        for sid in sids:
            tags = [
                t for t in PAIR_NAMES
                if (sid, t) in N and (sid, t) in R
            ]

            perm = list(rng.permutation(tags))

            for a, b in zip(tags, perm):
                vals.append(
                    cosine(
                        N[(sid, a)],
                        R[(sid, b)],
                    )
                )

        null.append(np.nanmean(vals))

    null = np.asarray(null)

    p = float(
        (1 + np.sum(null >= obs - 1e-12))
        / (1 + len(null))
    )

    return {
        "n_pairs": len(common),
        "semantic_residual_same_pair_cosine": obs,
        "pair_permutation_p": p,
    }


# ---------------------------------------------------------------------
# Equal-channel higher-order closure
# ---------------------------------------------------------------------

def equalized_closure(data):
    orig = data["orig"]
    mux = data["mux"]

    out = {}

    for style in STYLES:
        vals_first = []
        vals_pair = []

        for sid in sorted(orig["v135"]):
            if not (
                all(i in orig["v135"][sid] for i in range(4))
                and all(i in orig["v134"][sid] for i in range(4))
            ):
                continue

            b135 = mux["v135"][sid].get(style, {})
            b134 = mux["v134"][sid].get(style, {})

            if not (
                "all" in b135
                and "all" in b134
                and all(
                    t in b135 and t in b134
                    for t in PAIR_NAMES
                )
            ):
                continue

            try:
                all135 = 0.5 * (
                    b135["all"]["fwd"]
                    + b135["all"]["rev"]
                )
                all134 = 0.5 * (
                    b134["all"]["fwd"]
                    + b134["all"]["rev"]
                )

                first135 = sum(
                    orig["v135"][sid][i]
                    for i in range(4)
                )
                first134 = sum(
                    orig["v134"][sid][i]
                    for i in range(4)
                )

                pair135 = sum(
                    0.5 * (
                        b135[t]["fwd"]
                        + b135[t]["rev"]
                    )
                    for t in PAIR_NAMES
                ) / 3.0

                pair134 = sum(
                    0.5 * (
                        b134[t]["fwd"]
                        + b134[t]["rev"]
                    )
                    for t in PAIR_NAMES
                ) / 3.0

            except Exception:
                continue

            A = equal_concat(all135, all134)
            F = equal_concat(first135, first134)
            P = equal_concat(pair135, pair134)

            vals_first.append(cosine(A, F))
            vals_pair.append(cosine(A, P))

        out[style] = {
            "n_systems": len(vals_first),
            "all4_vs_first_order_equalized": float(
                np.nanmean(vals_first)
            ),
            "all4_vs_pair_closure_equalized": float(
                np.nanmean(vals_pair)
            ),
        }

    return out


# ---------------------------------------------------------------------
# Route decoder.
#
# Train on all OTHER systems.
# Test only on held system.
# Can train wrapper A and evaluate wrapper B.
# ---------------------------------------------------------------------

def standardize_train_test(Xtr, Xte):
    mu = np.mean(Xtr, axis=0)
    sd = np.std(Xtr, axis=0)

    sd[sd < 1e-8] = 1.0

    return (
        (Xtr - mu) / sd,
        (Xte - mu) / sd,
    )


def evaluate_predictions(Y, P):
    """
    Y/P: rows x 20 roles.
    """
    labels = (np.abs(Y) > 0.25).astype(int)
    scores = np.abs(P)

    auc = np.nan
    ap = np.nan

    yf = labels.reshape(-1)
    pf = scores.reshape(-1)

    if len(np.unique(yf)) >= 2:
        auc = float(roc_auc_score(yf, pf))
        ap = float(average_precision_score(yf, pf))

    cosines = []
    signs = []

    for y, p in zip(Y, P):
        if np.linalg.norm(y) > EPS:
            cosines.append(cosine(y, p))

        nz = np.abs(y) > 0.25
        if np.any(nz):
            signs.extend(
                (
                    np.sign(p[nz])
                    == np.sign(y[nz])
                ).astype(float).tolist()
            )

    return {
        "auc": auc,
        "ap": ap,
        "cos": (
            float(np.nanmean(cosines))
            if cosines else np.nan
        ),
        "sign_acc": (
            float(np.mean(signs))
            if signs else np.nan
        ),
    }


def route_decode(
    rows,
    train_style,
    test_style,
    feature,
    target,
    mode,
):
    usable = [
        r for r in rows
        if r["route_labels_ok"]
    ]

    tr_style_rows = [
        r for r in usable
        if r["style"] == train_style
    ]

    te_style_rows = [
        r for r in usable
        if r["style"] == test_style
    ]

    sids = sorted(
        {
            r["sid"]
            for r in te_style_rows
        }
    )

    per_system = []

    for held in sids:
        train = [
            r for r in tr_style_rows
            if r["sid"] != held
        ]

        test = [
            r for r in te_style_rows
            if r["sid"] == held
        ]

        if not train or not test:
            continue

        Ytr = np.stack([
            r[target]
            for r in train
        ])

        Yte = np.stack([
            r[target]
            for r in test
        ])

        if mode == "query_only":

            _, P = tfidf_predict(
                [r["pair_text"] for r in train],
                Ytr,
                [r["pair_text"] for r in test],
                alpha=30.0,
            )

        else:

            Xtr = np.stack([
                r[feature]
                for r in train
            ])

            Xte = np.stack([
                r[feature]
                for r in test
            ])

            if mode == "semantic_residual":

                qtr_pred, qte_pred = tfidf_predict(
                    [r["pair_text"] for r in train],
                    Xtr,
                    [r["pair_text"] for r in test],
                    alpha=30.0,
                )

                Xtr = Xtr - qtr_pred
                Xte = Xte - qte_pred

            Xtr, Xte = standardize_train_test(
                Xtr,
                Xte,
            )

            model = Ridge(
                alpha=20.0,
            )
            model.fit(Xtr, Ytr)

            P = model.predict(Xte)

        m = evaluate_predictions(
            Yte,
            P,
        )

        m["sid"] = held
        per_system.append(m)

    aucs = [
        x["auc"]
        for x in per_system
        if np.isfinite(x["auc"])
    ]

    aps = [
        x["ap"]
        for x in per_system
        if np.isfinite(x["ap"])
    ]

    cos = [
        x["cos"]
        for x in per_system
        if np.isfinite(x["cos"])
    ]

    sign = [
        x["sign_acc"]
        for x in per_system
        if np.isfinite(x["sign_acc"])
    ]

    return {
        "n_systems": len(per_system),

        "mean_system_auc": (
            float(np.mean(aucs))
            if aucs else None
        ),

        "auc_exact_signflip_p_vs_0.5": (
            exact_signflip([
                a - 0.5
                for a in aucs
            ])
            if aucs else None
        ),

        "mean_system_ap": (
            float(np.mean(aps))
            if aps else None
        ),

        "mean_signed_delta_cosine": (
            float(np.mean(cos))
            if cos else None
        ),

        "mean_nonzero_sign_accuracy": (
            float(np.mean(sign))
            if sign else None
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument(
        "--source-root",
        default="artifacts/adaptive_v11_role_only_v1",
    )
    args = ap.parse_args()

    root = Path(args.root)
    src = Path(args.source_root)

    outdir = root / "interference_battery_v2"
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("[V2] loading V1-compatible experiment", flush=True)

    data = ib1.load_everything(
        root,
        src,
    )

    pair_records = ib1.make_pair_records(
        data
    )

    rows = build_rows(
        data,
        pair_records,
    )

    print(
        f"[V2] pair rows={len(rows)} "
        f"routing-labeled={sum(r['route_labels_ok'] for r in rows)}",
        flush=True,
    )

    # -------------------------------------------------------------
    # A. Equalization control
    # -------------------------------------------------------------

    cross = {}

    for feat in (
        "anti_concat_raw",
        "anti_concat_equal",
        "resid_concat_raw",
        "resid_concat_equal",
    ):
        cross[feat] = cross_wrapper(
            rows,
            feat,
        )

    # -------------------------------------------------------------
    # B. Can query semantics alone predict interference?
    # -------------------------------------------------------------

    query_pred = {}

    for style in STYLES:
        query_pred[style] = {}

        for feat in (
            "anti_v135",
            "anti_v134",
            "anti_concat_equal",
            "resid_v135",
            "resid_v134",
            "resid_concat_equal",
        ):
            query_pred[style][feat] = (
                query_predictability(
                    rows,
                    style,
                    feat,
                )
            )

    # -------------------------------------------------------------
    # C. Cross-wrapper invariance after removing query semantics.
    # -------------------------------------------------------------

    residual_invariance = {}

    for feat in (
        "anti_v135",
        "anti_v134",
        "anti_concat_equal",
        "resid_v135",
        "resid_v134",
        "resid_concat_equal",
    ):
        residual_invariance[feat] = (
            residual_cross_wrapper(
                rows,
                feat,
            )
        )

    # -------------------------------------------------------------
    # D. Higher-order closure after equalizing channels.
    # -------------------------------------------------------------

    closure = equalized_closure(
        data
    )

    # -------------------------------------------------------------
    # E. Actual vector-level routing prediction.
    # -------------------------------------------------------------

    directions = {
        "N_to_N": (
            "numbered",
            "numbered",
        ),
        "R_to_R": (
            "requirements",
            "requirements",
        ),
        "N_to_R": (
            "numbered",
            "requirements",
        ),
        "R_to_N": (
            "requirements",
            "numbered",
        ),
    }

    route_results = {
        "order_direction": {},
        "symmetric_route_interaction": {},
    }

    targets = {
        "order_direction": (
            "route_order_delta",
            "anti",
        ),
        "symmetric_route_interaction": (
            "route_sym_delta",
            "resid",
        ),
    }

    for target_name, (
        target_key,
        prefix,
    ) in targets.items():

        route_results[target_name] = {}

        for direction, (
            tr_style,
            te_style,
        ) in directions.items():

            route_results[target_name][
                direction
            ] = {}

            configs = [
                (
                    "query_only",
                    None,
                    "query_only",
                ),
                (
                    "V135",
                    f"{prefix}_v135",
                    "raw",
                ),
                (
                    "V134",
                    f"{prefix}_v134",
                    "raw",
                ),
                (
                    "equal_concat",
                    f"{prefix}_concat_equal",
                    "raw",
                ),
                (
                    "equal_concat_SEMANTIC_RESIDUAL",
                    f"{prefix}_concat_equal",
                    "semantic_residual",
                ),
            ]

            for name, feature, mode in configs:

                route_results[
                    target_name
                ][direction][name] = (
                    route_decode(
                        rows,
                        tr_style,
                        te_style,
                        feature,
                        target_key,
                        mode,
                    )
                )

    # -------------------------------------------------------------
    # Decision gate.
    # -------------------------------------------------------------

    candidate_targets = []

    for target_name in route_results:
        a = route_results[
            target_name
        ]["N_to_R"][
            "equal_concat_SEMANTIC_RESIDUAL"
        ]

        b = route_results[
            target_name
        ]["R_to_N"][
            "equal_concat_SEMANTIC_RESIDUAL"
        ]

        qa = route_results[
            target_name
        ]["N_to_R"]["query_only"]

        qb = route_results[
            target_name
        ]["R_to_N"]["query_only"]

        good = (
            a["mean_system_auc"] is not None
            and b["mean_system_auc"] is not None
            and a["mean_system_auc"] >= 0.60
            and b["mean_system_auc"] >= 0.60
            and a[
                "auc_exact_signflip_p_vs_0.5"
            ] <= 0.05
            and b[
                "auc_exact_signflip_p_vs_0.5"
            ] <= 0.05
            and a["mean_system_auc"]
            >= qa["mean_system_auc"] + 0.05
            and b["mean_system_auc"]
            >= qb["mean_system_auc"] + 0.05
        )

        candidate_targets.append({
            "target": target_name,
            "passes": bool(good),
            "N_to_R_residual_auc":
                a["mean_system_auc"],
            "R_to_N_residual_auc":
                b["mean_system_auc"],
            "N_to_R_query_auc":
                qa["mean_system_auc"],
            "R_to_N_query_auc":
                qb["mean_system_auc"],
        })

    structural_go = any(
        x["passes"]
        for x in candidate_targets
    )

    summary = {
        "experiment":
            "semantic_interference_battery_v2",

        "pair_rows": len(rows),

        "routing_labeled_rows": sum(
            r["route_labels_ok"]
            for r in rows
        ),

        "equalization_control":
            cross,

        "query_only_interference_predictability":
            query_pred,

        "semantic_residual_cross_wrapper":
            residual_invariance,

        "equalized_higher_order_closure":
            closure,

        "vector_route_decoding":
            route_results,

        "decision": {
            "exploratory_structural_GO":
                structural_go,

            "criteria":
                "Cross-wrapper semantic-residualized equal-channel "
                "routing AUC >= .60 in BOTH directions, exact "
                "sign-flip p <= .05 in BOTH, and >= .05 AUC above "
                "query-only in BOTH directions.",

            "targets":
                candidate_targets,
        },
    }

    (
        outdir / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # Human-readable report
    # -------------------------------------------------------------

    L = []

    L.append("=" * 96)
    L.append(
        "SEMANTIC INTERFERENCE BATTERY V2 — STRUCTURAL SIGNAL TEST"
    )
    L.append("=" * 96)

    L.append(
        f"pair rows: {len(rows)} | "
        f"routing-labeled rows: "
        f"{sum(r['route_labels_ok'] for r in rows)}"
    )

    L.append("")
    L.append(
        "A) CHANNEL EQUALIZATION CONTROL"
    )

    for feat, x in cross.items():
        L.append(
            f"  {feat:22s} "
            f"same-pair-cos={x['same_pair_cosine']:.4f} "
            f"perm-p={x['pair_permutation_p']:.5f} "
            f"magnitude-rho="
            f"{x['mean_pair_magnitude_rank_rho']:.4f}"
        )

    L.append("")
    L.append(
        "B) CAN QUERY SEMANTICS ALONE PREDICT INTERFERENCE?"
    )

    for style in STYLES:
        L.append(f"  [{style}]")

        for feat, x in query_pred[
            style
        ].items():

            L.append(
                f"    {feat:22s} "
                f"heldout-cos="
                f"{x['mean_heldout_cosine']:.4f} "
                f"signflip-p="
                f"{x['cosine_signflip_p']:.5f} "
                f"R2={x['heldout_R2_vs_train_mean']:+.4f}"
            )

    L.append("")
    L.append(
        "C) CROSS-WRAPPER INTERFERENCE AFTER REMOVING QUERY-PREDICTABLE COMPONENT"
    )

    for feat, x in residual_invariance.items():

        L.append(
            f"  {feat:22s} "
            f"residual-cos="
            f"{x['semantic_residual_same_pair_cosine']:.4f} "
            f"perm-p="
            f"{x['pair_permutation_p']:.5f}"
        )

    L.append("")
    L.append(
        "D) EQUAL-CHANNEL HIGHER-ORDER CLOSURE"
    )

    for style, x in closure.items():

        L.append(
            f"  {style:12s} "
            f"all4~first="
            f"{x['all4_vs_first_order_equalized']:.4f} "
            f"all4~pairclosure="
            f"{x['all4_vs_pair_closure_equalized']:.4f}"
        )

    L.append("")
    L.append(
        "E) VECTOR-LEVEL PRIVILEGED ROUTING DECODING"
    )

    for target_name, directions_obj in (
        route_results.items()
    ):

        L.append("")
        L.append(
            f"  TARGET: {target_name}"
        )

        for direction, models in (
            directions_obj.items()
        ):

            L.append(
                f"    [{direction}]"
            )

            for model_name, x in models.items():

                auc = x[
                    "mean_system_auc"
                ]

                p = x[
                    "auc_exact_signflip_p_vs_0.5"
                ]

                ap = x[
                    "mean_system_ap"
                ]

                co = x[
                    "mean_signed_delta_cosine"
                ]

                sa = x[
                    "mean_nonzero_sign_accuracy"
                ]

                L.append(
                    f"      {model_name:36s} "
                    f"AUC={auc if auc is not None else float('nan'):.4f} "
                    f"p={p if p is not None else float('nan'):.5f} "
                    f"AP={ap if ap is not None else float('nan'):.4f} "
                    f"signed-cos="
                    f"{co if co is not None else float('nan'):.4f} "
                    f"sign-acc="
                    f"{sa if sa is not None else float('nan'):.4f}"
                )

    L.append("")
    L.append("=" * 96)
    L.append(
        "PRE-REGISTERED EXPLORATORY DECISION"
    )

    for x in candidate_targets:

        L.append(
            f"  {x['target']}: "
            f"N→R residual={x['N_to_R_residual_auc']:.4f} "
            f"vs query={x['N_to_R_query_auc']:.4f}; "
            f"R→N residual={x['R_to_N_residual_auc']:.4f} "
            f"vs query={x['R_to_N_query_auc']:.4f}; "
            f"PASS={x['passes']}"
        )

    L.append("")

    if structural_go:
        L.append(
            "OVERALL: EXPLORATORY GO — interference contains "
            "cross-wrapper routing information beyond query semantics."
        )
    else:
        L.append(
            "OVERALL: NO-GO FOR STRUCTURAL INTERFERENCE — "
            "do not spend more API calls on this mechanism unless "
            "a diagnostic above reveals a specific correctable failure."
        )

    L.append("=" * 96)

    report = "\n".join(L)

    (
        outdir / "REPORT.txt"
    ).write_text(
        report + "\n",
        encoding="utf-8",
    )

    print(report)
    print()
    print(
        f"[done] {outdir / 'REPORT.txt'}"
    )


if __name__ == "__main__":
    main()
