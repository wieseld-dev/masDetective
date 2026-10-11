from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge
from sentence_transformers import SentenceTransformer

from v11.role_universe import TRAIN_ROLES

ROLES = list(TRAIN_ROLES)
SEED = 94173
EPS = 1e-12


def read_jsonl(path):
    p = Path(path)
    return [
        json.loads(x)
        for x in p.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def norm(s):
    return re.sub(r"\s+", " ", str(s or "").strip()).lower()


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > EPS else np.zeros_like(v)


def cos(a, b):
    return float(np.dot(unit(a), unit(b)))


def load_source(path, source):
    out = []
    seen = set()

    for ri, r in enumerate(read_jsonl(path)):
        if r.get("phase") != "train":
            continue
        if r.get("status") not in (None, "ok"):
            continue

        target = str(r.get("target_role") or "").strip()
        query = str(r.get("query") or "").strip()

        if target not in ROLES or not query:
            continue

        branches = list(r.get("branches_training_only") or [])
        branches += list(r.get("confirmation_branches_training_only") or [])

        for bi, b in enumerate(branches):
            plus = str(b.get("plus_output") or "").strip()
            minus = str(b.get("minus_output") or "").strip()
            sham = str(b.get("sham_output") or "").strip()

            if not plus or not minus or not sham:
                continue

            pid = str(b.get("pair_id") or f"{ri}-{bi}")

            key = (
                source,
                target,
                norm(query),
                hashlib.sha1(plus.encode()).hexdigest(),
                hashlib.sha1(minus.encode()).hexdigest(),
                hashlib.sha1(sham.encode()).hexdigest(),
            )

            if key in seen:
                continue
            seen.add(key)

            out.append({
                "source": source,
                "target": target,
                "query": query,
                "query_norm": norm(query),
                "pair_id": pid,
                "plus": plus,
                "minus": minus,
                "sham": sham,
            })

    return out


def encode_unique(model, texts):
    uniq = list(dict.fromkeys(texts))

    E = model.encode(
        uniq,
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return {
        t: np.asarray(e, float)
        for t, e in zip(uniq, E)
    }


def mean_ci(x, B=5000):
    x = np.asarray(x, float)
    rng = np.random.default_rng(SEED)

    vals = [
        np.mean(rng.choice(x, len(x), replace=True))
        for _ in range(B)
    ]

    return (
        float(np.mean(x)),
        float(np.quantile(vals, .025)),
        float(np.quantile(vals, .975)),
    )


def signflip_p(x, B=20000):
    x = np.asarray(x, float)
    obs = abs(np.mean(x))

    rng = np.random.default_rng(SEED + 1)

    null = [
        abs(np.mean(
            x * rng.choice([-1., 1.], size=len(x))
        ))
        for _ in range(B)
    ]

    return float(
        (1 + np.sum(np.asarray(null) >= obs - 1e-12))
        / (B + 1)
    )


def make_tangents(rows, emb, qemb):
    for r in rows:
        ep = emb[r["plus"]]
        em = emb[r["minus"]]
        es = emb[r["sham"]]

        r["pm"] = unit(ep - em)
        r["sm"] = unit(es - em)
        r["qvec"] = qemb[r["query"]]

    return rows


def replicate_test(rows):
    """
    Does PLUS-MINUS agree with SHAM-MINUS for the same
    intervention more than another tangent for same role?
    """
    by = defaultdict(list)

    for i, r in enumerate(rows):
        by[(r["source"], r["target"])].append(i)

    paired = []
    same_role_wrong_pair = []
    diffs = []

    for key, inds in by.items():
        if len(inds) < 2:
            continue

        for pos, i in enumerate(inds):
            j = inds[(pos + 1) % len(inds)]

            a = cos(rows[i]["pm"], rows[i]["sm"])
            b = cos(rows[i]["pm"], rows[j]["sm"])

            paired.append(a)
            same_role_wrong_pair.append(b)
            diffs.append(a - b)

    return {
        "n": len(diffs),
        "paired_pm_sm": mean_ci(paired),
        "same_role_wrong_pair": mean_ci(same_role_wrong_pair),
        "paired_minus_wrong_delta": mean_ci(diffs),
        "signflip_p": signflip_p(diffs),
    }


def cross_source_pair_auc(rows):
    A = [r for r in rows if r["source"] == "v11"]
    B = [r for r in rows if r["source"] == "v13"]

    y = []
    tangent = []
    query = []

    exact_query_overlap_pairs = 0

    for a in A:
        for b in B:
            # Strict: do not let exact same diagnostic wording create
            # an artificially easy cross-source pair.
            if a["query_norm"] == b["query_norm"]:
                exact_query_overlap_pairs += 1
                continue

            y.append(int(a["target"] == b["target"]))
            tangent.append(cos(a["pm"], b["pm"]))
            query.append(cos(a["qvec"], b["qvec"]))

    y = np.asarray(y)
    tangent = np.asarray(tangent)
    query = np.asarray(query)

    auc_t = float(roc_auc_score(y, tangent))
    auc_q = float(roc_auc_score(y, query))

    # Remove the component of tangent similarity linearly predictable
    # from query similarity. No structural labels are used here.
    X = np.column_stack([np.ones(len(query)), query])
    beta = np.linalg.lstsq(X, tangent, rcond=None)[0]
    residual = tangent - X @ beta

    auc_res = float(roc_auc_score(y, residual))

    return {
        "n_pairs": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "exact_query_pairs_excluded": exact_query_overlap_pairs,
        "query_similarity_auc": auc_q,
        "tangent_similarity_auc": auc_t,
        "tangent_residualized_on_query_auc": auc_res,
    }


def centroid_transfer(rows, train_source, test_source):
    train = [r for r in rows if r["source"] == train_source]
    test = [r for r in rows if r["source"] == test_source]

    train_queries = {r["query_norm"] for r in train}

    roles = sorted(
        set(r["target"] for r in train)
        & set(r["target"] for r in test)
    )

    centroids = {}

    for role in roles:
        rr = [
            r["pm"]
            for r in train
            if r["target"] == role
        ]

        centroids[role] = unit(np.mean(rr, axis=0))

    correct = []
    rranks = []
    margins = []

    strict_correct = []
    strict_rranks = []

    for r in test:
        if r["target"] not in centroids:
            continue

        scores = {
            role: cos(r["pm"], c)
            for role, c in centroids.items()
        }

        order = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )

        rank = order.index(r["target"]) + 1
        correct.append(rank == 1)
        rranks.append(1. / rank)

        true = scores[r["target"]]
        wrong = max(
            v for k, v in scores.items()
            if k != r["target"]
        )
        margins.append(true - wrong)

        if r["query_norm"] not in train_queries:
            strict_correct.append(rank == 1)
            strict_rranks.append(1. / rank)

    return {
        "roles": len(roles),
        "chance_top1": 1. / max(1, len(roles)),
        "n_test": len(correct),
        "top1": float(np.mean(correct)),
        "mrr": float(np.mean(rranks)),
        "mean_true_minus_best_wrong": float(np.mean(margins)),
        "n_strict_unseen_query": len(strict_correct),
        "strict_unseen_query_top1": float(np.mean(strict_correct))
            if strict_correct else None,
        "strict_unseen_query_mrr": float(np.mean(strict_rranks))
            if strict_rranks else None,
    }


def within_source_loso(rows, source):
    rr = [r for r in rows if r["source"] == source]

    groups = defaultdict(list)

    for i, r in enumerate(rr):
        groups[r["query_norm"]].append(i)

    roles = sorted(set(r["target"] for r in rr))

    correct = []
    ranks = []

    for q, test_idx in groups.items():
        test_set = set(test_idx)
        train = [
            r for i, r in enumerate(rr)
            if i not in test_set
        ]

        cents = {}

        for role in roles:
            vals = [
                r["pm"]
                for r in train
                if r["target"] == role
            ]

            if vals:
                cents[role] = unit(
                    np.mean(vals, axis=0)
                )

        for i in test_idx:
            r = rr[i]

            if r["target"] not in cents:
                continue

            scores = {
                role: cos(r["pm"], c)
                for role, c in cents.items()
            }

            order = sorted(
                scores,
                key=scores.get,
                reverse=True,
            )

            rank = order.index(r["target"]) + 1
            correct.append(rank == 1)
            ranks.append(1. / rank)

    return {
        "n": len(correct),
        "roles": len(roles),
        "chance": 1. / len(roles),
        "top1": float(np.mean(correct)),
        "mrr": float(np.mean(ranks)),
    }


def tangent_geometry(rows):
    cents = []

    for role in ROLES:
        vals = [
            r["pm"]
            for r in rows
            if r["target"] == role
        ]

        if vals:
            cents.append(
                unit(np.mean(vals, axis=0))
            )

    C = np.stack(cents)

    C = C - C.mean(axis=0, keepdims=True)

    s = np.linalg.svd(
        C,
        compute_uv=False,
    )

    stable_rank = float(
        np.sum(s**2) / (s[0]**2 + EPS)
    )

    p = (s**2) / np.sum(s**2)
    effective_rank = float(
        np.exp(
            -np.sum(
                p * np.log(p + EPS)
            )
        )
    )

    return {
        "n_role_centroids": len(C),
        "singular_values": s.tolist(),
        "stable_rank": stable_rank,
        "effective_rank": effective_rank,
    }


def main():
    rows = []

    rows += load_source(
        "artifacts/v11_probe_discovery.jsonl",
        "v11",
    )

    rows += load_source(
        "artifacts/v13_probe_disambiguation.jsonl",
        "v13",
    )

    print(
        f"[load] rows={len(rows)} "
        f"v11={sum(r['source']=='v11' for r in rows)} "
        f"v13={sum(r['source']=='v13' for r in rows)}",
        flush=True,
    )

    all_outputs = []

    for r in rows:
        all_outputs += [
            r["plus"],
            r["minus"],
            r["sham"],
        ]

    queries = [r["query"] for r in rows]

    print("[embed] loading all-MiniLM-L6-v2", flush=True)

    model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device="cpu",
    )

    print("[embed] outputs", flush=True)
    emb = encode_unique(model, all_outputs)

    print("[embed] queries", flush=True)
    qemb = encode_unique(model, queries)

    rows = make_tangents(
        rows,
        emb,
        qemb,
    )

    print("[test] replicate", flush=True)
    replicate = replicate_test(rows)

    print("[test] cross-source pair AUC", flush=True)
    pair_auc = cross_source_pair_auc(rows)

    print("[test] centroid V11 -> V13", flush=True)
    v11_v13 = centroid_transfer(
        rows,
        "v11",
        "v13",
    )

    print("[test] centroid V13 -> V11", flush=True)
    v13_v11 = centroid_transfer(
        rows,
        "v13",
        "v11",
    )

    print("[test] within-source leave-query-out", flush=True)
    loso_v11 = within_source_loso(rows, "v11")
    loso_v13 = within_source_loso(rows, "v13")

    print("[test] tangent spectrum", flush=True)
    geometry = tangent_geometry(rows)

    # Exploratory continuation gate, not a theorem.
    gate = (
        v11_v13["strict_unseen_query_top1"] is not None
        and v13_v11["strict_unseen_query_top1"] is not None
        and v11_v13["strict_unseen_query_top1"] >= .20
        and v13_v11["strict_unseen_query_top1"] >= .20
        and pair_auc["tangent_similarity_auc"] >= .60
        and pair_auc[
            "tangent_residualized_on_query_auc"
        ] >= .57
        and replicate["paired_minus_wrong_delta"][0] >= .05
        and replicate["signflip_p"] <= .05
    )

    summary = {
        "experiment": "structural_edit_tangent_v0",
        "rows": len(rows),
        "replicate": replicate,
        "cross_source_pair_auc": pair_auc,
        "centroid_transfer": {
            "v11_to_v13": v11_v13,
            "v13_to_v11": v13_v11,
        },
        "within_source_leave_query_out": {
            "v11": loso_v11,
            "v13": loso_v13,
        },
        "geometry": geometry,
        "decision": {
            "pass": bool(gate),
            "note":
                "Exploratory continuation gate only; "
                "not a claim of statistical identifiability.",
        },
    }

    out = Path("artifacts/edit_tangent_v0")
    out.mkdir(parents=True, exist_ok=True)

    (out/"summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    L = []
    L.append("="*92)
    L.append("STRUCTURAL EDIT TANGENT V0")
    L.append("="*92)

    L.append(
        f"rows={len(rows)} | "
        f"V11={sum(r['source']=='v11' for r in rows)} | "
        f"V13={sum(r['source']=='v13' for r in rows)}"
    )

    L.append("")
    L.append("A) SAME-EDIT REPLICATION")
    L.append(
        f"  PM~SM same pair           = "
        f"{replicate['paired_pm_sm'][0]:+.4f} "
        f"[{replicate['paired_pm_sm'][1]:+.4f}, "
        f"{replicate['paired_pm_sm'][2]:+.4f}]"
    )
    L.append(
        f"  PM~SM same-role wrong pair= "
        f"{replicate['same_role_wrong_pair'][0]:+.4f}"
    )
    L.append(
        f"  paired advantage          = "
        f"{replicate['paired_minus_wrong_delta'][0]:+.4f} "
        f"p={replicate['signflip_p']:.5f}"
    )

    L.append("")
    L.append("B) CROSS-SOURCE SAME-EDIT DETECTION")
    L.append(
        f"  query-sim AUC             = "
        f"{pair_auc['query_similarity_auc']:.4f}"
    )
    L.append(
        f"  tangent-sim AUC           = "
        f"{pair_auc['tangent_similarity_auc']:.4f}"
    )
    L.append(
        f"  tangent | query residual AUC = "
        f"{pair_auc['tangent_residualized_on_query_auc']:.4f}"
    )
    L.append(
        f"  exact-query pairs excluded= "
        f"{pair_auc['exact_query_pairs_excluded']}"
    )

    L.append("")
    L.append("C) CROSS-SOURCE EDIT IDENTIFICATION")

    for name, x in (
        ("V11 -> V13", v11_v13),
        ("V13 -> V11", v13_v11),
    ):
        L.append(
            f"  {name:12s}: "
            f"top1={x['top1']:.4f} "
            f"MRR={x['mrr']:.4f} "
            f"margin={x['mean_true_minus_best_wrong']:+.4f} "
            f"chance={x['chance_top1']:.4f}"
        )
        L.append(
            f"                 strict unseen-query "
            f"top1={x['strict_unseen_query_top1']} "
            f"MRR={x['strict_unseen_query_mrr']} "
            f"n={x['n_strict_unseen_query']}"
        )

    L.append("")
    L.append("D) WITHIN-SOURCE LEAVE-QUERY-OUT")
    L.append(
        f"  V11 top1={loso_v11['top1']:.4f} "
        f"MRR={loso_v11['mrr']:.4f} "
        f"chance={loso_v11['chance']:.4f}"
    )
    L.append(
        f"  V13 top1={loso_v13['top1']:.4f} "
        f"MRR={loso_v13['mrr']:.4f} "
        f"chance={loso_v13['chance']:.4f}"
    )

    L.append("")
    L.append("E) EDIT-TANGENT GEOMETRY")
    L.append(
        f"  role centroids={geometry['n_role_centroids']}"
    )
    L.append(
        f"  stable rank={geometry['stable_rank']:.3f}"
    )
    L.append(
        f"  effective rank={geometry['effective_rank']:.3f}"
    )
    L.append(
        "  top singular values="
        + ", ".join(
            f"{x:.3f}"
            for x in geometry["singular_values"][:10]
        )
    )

    L.append("")
    L.append("="*92)

    if gate:
        L.append(
            "OVERALL: GO — local role edits induce a reproducible "
            "observable tangent geometry. Next test: edge/topology edits."
        )
    else:
        L.append(
            "OVERALL: NO-GO — role-edit tangents are not sufficiently "
            "stable/transferable to justify Structural Fisher Geometry."
        )

    L.append("="*92)

    report = "\n".join(L)

    (out/"REPORT.txt").write_text(
        report+"\n",
        encoding="utf-8",
    )

    print(report)


if __name__ == "__main__":
    main()
