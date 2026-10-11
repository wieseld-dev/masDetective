from __future__ import annotations

import copy
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

import dag_operator_active_poc as base


ROOT = Path("artifacts/dag_operator_active_poc")
OUT = Path("artifacts/fixed_dag_rank_v1")
OUT.mkdir(parents=True, exist_ok=True)

TOPOS = list(base.TOPOLOGY_NAMES)
PAIR_LIST = list(itertools.combinations(TOPOS, 2))

TOP_FRAC = 0.20
N_SHUFFLE = 5000
N_RANDOM_K = 2000
SEED = 9172026


def numeric_pid(pid: str):
    try:
        return int(pid.split("_")[-1])
    except Exception:
        return pid


def original_defaults():
    """
    Get the exact hyperparameter defaults from the original Fixed-DAG script
    without allowing this wrapper's argv to affect argparse.
    """
    old = sys.argv[:]
    try:
        sys.argv = ["dag_operator_active_poc.py"]
        args = base.parse_args()
    finally:
        sys.argv = old
    return args


def load_data():
    report = json.loads(
        (ROOT / "report.json").read_text(encoding="utf-8")
    )

    rows = [
        json.loads(x)
        for x in (ROOT / "api_runs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if x.strip()
    ]

    obs = {}

    for r in rows:
        pid = str(r["probe_id"])
        topo = str(r["topology"])

        if pid not in obs:
            obs[pid] = {
                "query": str(r["query"]),
                "final": {},
            }

        obs[pid]["final"][topo] = str(r["final"])

    if len(obs) != 60:
        raise RuntimeError(f"Expected 60 probes, got {len(obs)}")

    for pid, x in obs.items():
        missing = set(TOPOS) - set(x["final"])
        if missing:
            raise RuntimeError(f"{pid} missing {missing}")

    test_ids = list(map(str, report["heldout_probe_ids"]))

    if len(test_ids) != 18:
        raise RuntimeError(
            f"Expected 18 frozen held-out IDs, got {len(test_ids)}"
        )

    test_set = set(test_ids)

    train_ids = sorted(
        [p for p in obs if p not in test_set],
        key=numeric_pid,
    )

    if len(train_ids) != 42:
        raise RuntimeError(
            f"Expected 42 train IDs, got {len(train_ids)}"
        )

    # Order does not change fitting; keep frozen held-out ordering from report.
    all_ids = train_ids + test_ids

    return report, obs, train_ids, test_ids, all_ids


def metric(score, actual):
    score = np.asarray(score, float)
    actual = np.asarray(actual, float)

    rho = float(spearmanr(score, actual).statistic)

    n = len(actual)
    k = max(1, int(math.ceil(TOP_FRAC * n)))

    top = np.argsort(score)[::-1][:k]
    oracle = np.argsort(actual)[::-1][:k]

    baseline = float(np.mean(actual))
    selected = float(np.mean(actual[top]))
    oracle_val = float(np.mean(actual[oracle]))

    lift = selected / max(baseline, 1e-12)
    oracle_lift = oracle_val / max(baseline, 1e-12)

    denom = oracle_val - baseline

    capture = (
        (selected - baseline) / denom
        if denom > 1e-12
        else 0.0
    )

    best = int(np.argmax(score))
    top1_lift = float(actual[best] / max(baseline, 1e-12))

    return {
        "rho": rho,
        "k": k,
        "baseline_actual": baseline,
        "selected_actual": selected,
        "lift20": float(lift),
        "oracle_actual": oracle_val,
        "oracle_lift20": float(oracle_lift),
        "oracle_opportunity_capture": float(capture),
        "top1_lift": top1_lift,
    }


def bootstrap_mean_ci(vals, B=30000):
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(SEED + 991)

    means = np.empty(B, dtype=float)

    for b in range(B):
        means[b] = np.mean(
            rng.choice(vals, size=len(vals), replace=True)
        )

    return [
        float(np.quantile(means, .025)),
        float(np.quantile(means, .975)),
    ]


def pair_scores(qte, zte, mats, pred):
    out = {}

    for gi, gj in PAIR_LIST:
        K = (
            np.asarray(mats[gi], float)
            -
            np.asarray(mats[gj], float)
        )

        spectral = np.einsum(
            "ni,ij,nj->n",
            qte,
            K.T @ K,
            qte,
        )

        actual = np.sum(
            (zte[gi] - zte[gj]) ** 2,
            axis=1,
        )

        # Non-spectral model prediction baseline.
        pred_disagreement = np.sum(
            (pred[gi] - pred[gj]) ** 2,
            axis=1,
        )

        out[(gi, gj)] = {
            "spectral_score": spectral,
            "pred_disagreement": pred_disagreement,
            "actual": actual,
            "spectral_metric": metric(spectral, actual),
            "disagreement_metric":
                metric(pred_disagreement, actual),
            "K_norm": float(np.linalg.norm(K, ord="fro")),
        }

    return out


def global_multi_hypothesis(qte, zte, mats):
    """
    Uniform prior = exact first-step version of the posterior-weighted
    multi-hypothesis spectral acquisition.
    """
    p = np.ones(len(TOPOS), dtype=float) / len(TOPOS)

    d = qte.shape[1]

    C = np.zeros((d, d), dtype=float)
    actual = np.zeros(len(qte), dtype=float)

    for i, gi in enumerate(TOPOS):
        for j in range(i + 1, len(TOPOS)):
            gj = TOPOS[j]

            w = float(p[i] * p[j])

            K = (
                np.asarray(mats[gi], float)
                -
                np.asarray(mats[gj], float)
            )

            C += w * (K.T @ K)

            actual += w * np.sum(
                (zte[gi] - zte[gj]) ** 2,
                axis=1,
            )

    score = np.einsum(
        "ni,ij,nj->n",
        qte,
        C,
        qte,
    )

    return {
        "metric": metric(score, actual),
        "score": score,
        "actual": actual,
    }


def summarize_pairwise(pair_result):
    srho = [
        x["spectral_metric"]["rho"]
        for x in pair_result.values()
    ]

    slift = [
        x["spectral_metric"]["lift20"]
        for x in pair_result.values()
    ]

    scap = [
        x["spectral_metric"]["oracle_opportunity_capture"]
        for x in pair_result.values()
    ]

    drho = [
        x["disagreement_metric"]["rho"]
        for x in pair_result.values()
    ]

    dlift = [
        x["disagreement_metric"]["lift20"]
        for x in pair_result.values()
    ]

    return {
        "macro_spectral_rho": float(np.mean(srho)),
        "macro_spectral_rho_ci95": bootstrap_mean_ci(srho),
        "macro_spectral_lift20": float(np.mean(slift)),
        "macro_spectral_lift20_ci95": bootstrap_mean_ci(slift),
        "macro_spectral_capture": float(np.mean(scap)),
        "positive_spectral_rho_pairs":
            int(sum(x > 0 for x in srho)),
        "lift_gt_1_pairs":
            int(sum(x > 1 for x in slift)),
        "macro_disagreement_rho": float(np.mean(drho)),
        "macro_disagreement_lift20": float(np.mean(dlift)),
    }


def shuffle_null(pair_result):
    rng = np.random.default_rng(SEED + 123)

    rho_null = np.empty(N_SHUFFLE)
    lift_null = np.empty(N_SHUFFLE)

    pair_items = list(pair_result.values())

    for b in range(N_SHUFFLE):
        rr = []
        ll = []

        for x in pair_items:
            s = np.asarray(x["spectral_score"]).copy()
            rng.shuffle(s)

            m = metric(
                s,
                x["actual"],
            )

            rr.append(m["rho"])
            ll.append(m["lift20"])

        rho_null[b] = np.mean(rr)
        lift_null[b] = np.mean(ll)

    return {
        "rho_mean": float(rho_null.mean()),
        "rho_q95": float(np.quantile(rho_null, .95)),
        "lift_mean": float(lift_null.mean()),
        "lift_q95": float(np.quantile(lift_null, .95)),
    }


def random_k_null(qte, zte, mats):
    rng = np.random.default_rng(SEED + 456)

    rho_null = np.empty(N_RANDOM_K)
    lift_null = np.empty(N_RANDOM_K)

    for b in range(N_RANDOM_K):
        rr = []
        ll = []

        for gi, gj in PAIR_LIST:
            Ktrue = (
                np.asarray(mats[gi], float)
                -
                np.asarray(mats[gj], float)
            )

            target_norm = np.linalg.norm(
                Ktrue,
                ord="fro",
            )

            R = rng.normal(
                size=Ktrue.shape
            )

            R *= (
                target_norm
                /
                max(
                    np.linalg.norm(R, ord="fro"),
                    1e-12,
                )
            )

            score = np.einsum(
                "ni,ij,nj->n",
                qte,
                R.T @ R,
                qte,
            )

            actual = np.sum(
                (zte[gi] - zte[gj]) ** 2,
                axis=1,
            )

            m = metric(score, actual)

            rr.append(m["rho"])
            ll.append(m["lift20"])

        rho_null[b] = np.mean(rr)
        lift_null[b] = np.mean(ll)

    return {
        "rho_mean": float(rho_null.mean()),
        "rho_q95": float(np.quantile(rho_null, .95)),
        "lift_mean": float(lift_null.mean()),
        "lift_q95": float(np.quantile(lift_null, .95)),
    }


def serializable_pairs(x):
    out = {}

    for (a, b), v in x.items():
        out[f"{a}__vs__{b}"] = {
            "K_norm": v["K_norm"],
            "spectral_metric": v["spectral_metric"],
            "disagreement_metric": v["disagreement_metric"],
        }

    return out


def main():
    args = original_defaults()

    (
        old_report,
        obs,
        train_ids,
        test_ids,
        all_ids,
    ) = load_data()

    print("=" * 100)
    print("FIXED-DAG HELD-OUT SPECTRAL RANKING V1")
    print("=" * 100)

    print("Original defaults:")
    for name in (
        "embed_model",
        "seed",
        "dim",
        "ridge_alpha",
        "steps",
        "lr",
        "weight_decay",
        "identity_reg",
    ):
        print(
            f"  {name}={getattr(args, name)}"
        )

    print()
    print(
        f"frozen split: "
        f"train={len(train_ids)} "
        f"test={len(test_ids)}"
    )

    q_text = [
        obs[pid]["query"]
        for pid in all_ids
    ]

    by_topo = {
        g: [
            obs[pid]["final"][g]
            for pid in all_ids
        ]
        for g in TOPOS
    }

    ntr = len(train_ids)

    tr_idx = np.arange(ntr)
    te_idx = np.arange(
        ntr,
        len(all_ids),
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("device:", device)

    enc = base.SentenceTransformer(
        args.embed_model,
        device=device,
    )

    print("Encoding queries...")

    q_full = enc.encode(
        q_text,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    z_full = {}

    for g in TOPOS:
        print("Encoding", g)

        z_full[g] = enc.encode(
            by_topo[g],
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    # EXACT SAME PCA PRINCIPLE AS ORIGINAL:
    # fit only on train queries + train topology outputs.
    pca_train = [
        q_full[tr_idx]
    ] + [
        z_full[g][tr_idx]
        for g in TOPOS
    ]

    pca_train = np.vstack(
        pca_train
    )

    d = min(
        args.dim,
        pca_train.shape[0] - 1,
        pca_train.shape[1],
    )

    pca = PCA(
        n_components=d,
        random_state=args.seed,
    ).fit(pca_train)

    q = base.l2_rows(
        pca.transform(q_full)
    )

    z = {
        g: base.l2_rows(
            pca.transform(
                z_full[g]
            )
        )
        for g in TOPOS
    }

    qtr = q[tr_idx]
    qte = q[te_idx]

    ztr = {
        g: z[g][tr_idx]
        for g in TOPOS
    }

    zte = {
        g: z[g][te_idx]
        for g in TOPOS
    }

    # --------------------------------------------------------
    # DIRECT GRAPH-SPECIFIC MAPS
    # --------------------------------------------------------

    direct_maps = base.fit_direct_maps(
        qtr,
        ztr,
        args.ridge_alpha,
    )

    direct_pred = {
        g: base.direct_predict(
            qte,
            direct_maps[g],
        )
        for g in TOPOS
    }

    # --------------------------------------------------------
    # COMPOSED PATH-OPERATOR MODEL
    # --------------------------------------------------------

    op_model, train_loss = (
        base.fit_operator_model(
            qtr,
            ztr,
            device=device,
            steps=args.steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            identity_reg=args.identity_reg,
            seed=args.seed,
        )
    )

    with torch.no_grad():
        tqte = torch.tensor(
            qte,
            dtype=torch.float32,
            device=device,
        )

        op_pred = {
            g:
                op_model.predict(
                    tqte,
                    g,
                ).cpu().numpy()
            for g in TOPOS
        }

        FG = {
            g:
                op_model.graph_matrix(
                    g
                ).cpu().numpy()
            for g in TOPOS
        }

    # --------------------------------------------------------
    # EXACT CHAIN-LIKE QUESTION:
    # predicted spectral sensitivity vs ACTUAL heldout separation
    # --------------------------------------------------------

    op_pairs = pair_scores(
        qte,
        zte,
        FG,
        op_pred,
    )

    direct_pairs = pair_scores(
        qte,
        zte,
        direct_maps,
        direct_pred,
    )

    op_summary = summarize_pairwise(
        op_pairs
    )

    direct_summary = summarize_pairwise(
        direct_pairs
    )

    op_global = global_multi_hypothesis(
        qte,
        zte,
        FG,
    )

    direct_global = global_multi_hypothesis(
        qte,
        zte,
        direct_maps,
    )

    op_shuffle = shuffle_null(
        op_pairs
    )

    direct_shuffle = shuffle_null(
        direct_pairs
    )

    print("Computing matched random-K nulls...")

    op_random_k = random_k_null(
        qte,
        zte,
        FG,
    )

    direct_random_k = random_k_null(
        qte,
        zte,
        direct_maps,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    result = {
        "experiment":
            "fixed_dag_heldout_spectral_ranking_v1",

        "train_probe_ids":
            train_ids,

        "test_probe_ids":
            test_ids,

        "latent_dim":
            d,

        "operator_train_loss":
            float(train_loss),

        "operator": {
            "summary":
                op_summary,

            "multi_hypothesis_uniform":
                op_global["metric"],

            "shuffle_null":
                op_shuffle,

            "random_K_null":
                op_random_k,

            "pairs":
                serializable_pairs(
                    op_pairs
                ),
        },

        "direct": {
            "summary":
                direct_summary,

            "multi_hypothesis_uniform":
                direct_global["metric"],

            "shuffle_null":
                direct_shuffle,

            "random_K_null":
                direct_random_k,

            "pairs":
                serializable_pairs(
                    direct_pairs
                ),
        },
    }

    (
        OUT / "summary.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    L = []

    L.append("=" * 108)
    L.append(
        "FIXED-DAG HELD-OUT SPECTRAL PROBE RANKING"
    )
    L.append("=" * 108)

    L.append(
        f"42 frozen train probes | "
        f"18 frozen held-out probes | "
        f"5 DAGs | 10 DAG contrasts | "
        f"latent_dim={d}"
    )

    for title, pairs, summ, glob, shuf, rnd in [
        (
            "COMPOSED PATH OPERATOR",
            op_pairs,
            op_summary,
            op_global["metric"],
            op_shuffle,
            op_random_k,
        ),
        (
            "DIRECT GRAPH MAP",
            direct_pairs,
            direct_summary,
            direct_global["metric"],
            direct_shuffle,
            direct_random_k,
        ),
    ]:

        L.append("")
        L.append("-" * 108)
        L.append(title)
        L.append("-" * 108)

        for gi, gj in PAIR_LIST:
            m = pairs[
                (gi, gj)
            ]["spectral_metric"]

            L.append(
                f"{gi:12s} vs {gj:12s} | "
                f"rho={m['rho']:+.3f} | "
                f"lift20={m['lift20']:.3f}x | "
                f"capture={m['oracle_opportunity_capture']:.3f} | "
                f"oracle={m['oracle_lift20']:.3f}x"
            )

        ci_r = summ[
            "macro_spectral_rho_ci95"
        ]

        ci_l = summ[
            "macro_spectral_lift20_ci95"
        ]

        L.append("")
        L.append(
            f"MACRO spectral rho       = "
            f"{summ['macro_spectral_rho']:+.4f} "
            f"CI=[{ci_r[0]:+.4f},{ci_r[1]:+.4f}]"
        )

        L.append(
            f"MACRO spectral lift@20   = "
            f"{summ['macro_spectral_lift20']:.4f}x "
            f"CI=[{ci_l[0]:.4f},{ci_l[1]:.4f}]"
        )

        L.append(
            f"positive-rho pairs       = "
            f"{summ['positive_spectral_rho_pairs']}/10"
        )

        L.append(
            f"lift>1 pairs             = "
            f"{summ['lift_gt_1_pairs']}/10"
        )

        L.append(
            f"oracle-opportunity capture = "
            f"{summ['macro_spectral_capture']:.3f}"
        )

        L.append(
            f"prediction-disagreement rho = "
            f"{summ['macro_disagreement_rho']:+.4f}"
        )

        L.append(
            f"prediction-disagreement lift = "
            f"{summ['macro_disagreement_lift20']:.4f}x"
        )

        L.append("")
        L.append(
            f"UNIFORM MULTI-HYPOTHESIS: "
            f"rho={glob['rho']:+.4f} "
            f"lift20={glob['lift20']:.4f}x "
            f"capture={glob['oracle_opportunity_capture']:.3f}"
        )

        L.append(
            f"SHUFFLE NULL: "
            f"rho mean={shuf['rho_mean']:+.4f}, "
            f"95th={shuf['rho_q95']:+.4f}; "
            f"lift mean={shuf['lift_mean']:.4f}x, "
            f"95th={shuf['lift_q95']:.4f}x"
        )

        L.append(
            f"RANDOM-K NULL: "
            f"rho mean={rnd['rho_mean']:+.4f}, "
            f"95th={rnd['rho_q95']:+.4f}; "
            f"lift mean={rnd['lift_mean']:.4f}x, "
            f"95th={rnd['lift_q95']:.4f}x"
        )

    # Decision is intentionally about ranking, NOT topology accuracy.
    op_rho = op_summary[
        "macro_spectral_rho"
    ]

    op_lift = op_summary[
        "macro_spectral_lift20"
    ]

    dir_rho = direct_summary[
        "macro_spectral_rho"
    ]

    dir_lift = direct_summary[
        "macro_spectral_lift20"
    ]

    if (
        op_rho > op_shuffle["rho_q95"]
        and op_lift > op_shuffle["lift_q95"]
        and op_summary[
            "positive_spectral_rho_pairs"
        ] >= 7
    ):
        verdict = (
            "COMPOSED_DAG_SPECTRAL_GO"
        )

    elif (
        dir_rho > direct_shuffle["rho_q95"]
        and dir_lift > direct_shuffle["lift_q95"]
    ):
        verdict = (
            "SPECTRAL_GO_FACTORIZATION_NO_GO"
        )

    else:
        verdict = (
            "FIXED_DAG_RANKING_NO_GO"
        )

    L.append("")
    L.append("=" * 108)
    L.append(
        f"RANKING VERDICT: {verdict}"
    )

    if verdict == "COMPOSED_DAG_SPECTRAL_GO":
        L.append(
            "The same spectral principle that ranked order-sensitive "
            "Chain probes also ranks informative probes across fixed DAGs."
        )

    elif verdict == "SPECTRAL_GO_FACTORIZATION_NO_GO":
        L.append(
            "Fixed-DAG probes are spectrally rankable with graph-specific "
            "maps, but the current sum-of-path-products factorization "
            "does not generalize well enough."
        )

    else:
        L.append(
            "Even the direct graph-map upper baseline does not reliably "
            "rank held-out DAG-separating probes. The current latent "
            "representation / candidate bank lacks sufficient DAG signal."
        )

    L.append("=" * 108)

    report = "\n".join(L)

    (
        OUT / "REPORT.txt"
    ).write_text(
        report + "\n",
        encoding="utf-8",
    )

    print()
    print(report)


if __name__ == "__main__":
    main()
