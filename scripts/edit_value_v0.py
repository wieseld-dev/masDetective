from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import rankdata, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder

from v11.role_universe import TRAIN_ROLES, ROLE_DESCRIPTIONS

ROLES = list(TRAIN_ROLES)
RIDX = {r:i for i,r in enumerate(ROLES)}
SEED = 7717


def read_jsonl(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(p)
    return [
        json.loads(x)
        for x in p.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def arr(x):
    if x is None:
        return ()
    if isinstance(x, str):
        return (x,)
    try:
        return tuple(map(str, x))
    except Exception:
        return ()


def load_source(path, source, spec_key):
    rows = read_jsonl(path)
    out = []
    seen = set()

    for ri, r in enumerate(rows):
        if r.get("phase") != "train":
            continue
        if r.get("status") not in (None, "ok"):
            continue

        target = str(r.get("target_role") or "").strip()
        query = str(r.get("query") or "").strip()

        if target not in RIDX or not query:
            continue

        desc = str(
            r.get("candidate_description_public")
            or ROLE_DESCRIPTIONS.get(target, target)
        )

        branches = list(r.get("branches_training_only") or [])
        branches += list(r.get("confirmation_branches_training_only") or [])

        for bi, b in enumerate(branches):
            pid = str(b.get("pair_id") or f"{ri}-{bi}")

            try:
                y = float(b.get(spec_key))
            except Exception:
                continue

            if not np.isfinite(y):
                continue

            plus = arr(b.get("plus_roles_training_only"))
            minus = arr(b.get("minus_roles_training_only"))

            # V11 may use non-_training_only fields.
            if not plus:
                plus = arr(b.get("plus_roles"))
            if not minus:
                minus = arr(b.get("minus_roles"))

            key = (source, target, norm(query), pid)
            if key in seen:
                continue
            seen.add(key)

            # Structural context = all known surrounding roles excluding edit target.
            context = sorted(
                (set(plus) | set(minus)) - {target}
            )

            # Group exact queries together so paraphrase/query leakage cannot
            # cross train/test.
            group = hashlib.sha1(
                norm(query).encode()
            ).hexdigest()[:16]

            out.append({
                "source": source,
                "target": target,
                "target_desc": desc,
                "query": query,
                "pair_id": pid,
                "group": group,
                "context": context,
                "y_raw": y,
            })

    return out


def source_rank_targets(rows):
    """
    Different discovery generators may have different score scales.
    Evaluate prediction of within-source discriminability percentile.
    """
    by = defaultdict(list)
    for i,r in enumerate(rows):
        by[r["source"]].append(i)

    y = np.zeros(len(rows), dtype=float)

    for src, inds in by.items():
        vals = np.asarray([rows[i]["y_raw"] for i in inds])
        ranks = rankdata(vals, method="average")
        ranks = (ranks - 1.0) / max(1.0, len(ranks)-1.0)

        for i,v in zip(inds, ranks):
            y[i] = v

    return y


def role_matrix(rows):
    R = np.zeros((len(rows), len(ROLES)), dtype=float)
    for i,r in enumerate(rows):
        R[i, RIDX[r["target"]]] = 1.0
    return sparse.csr_matrix(R)


def context_matrix(rows):
    C = np.zeros((len(rows), len(ROLES)), dtype=float)
    for i,r in enumerate(rows):
        for role in r["context"]:
            if role in RIDX:
                C[i, RIDX[role]] = 1.0
    return sparse.csr_matrix(C)


def interaction_blocks(Xq, indicators):
    """
    Fixed-width interaction design.
    Every role always receives one TF-IDF interaction block,
    even when that role is absent from this particular split.
    """
    A = indicators.toarray()

    blocks = [
        Xq.multiply(A[:, j][:, None])
        for j in range(A.shape[1])
    ]

    if not blocks:
        return sparse.csr_matrix((Xq.shape[0], 0))

    return sparse.hstack(blocks, format="csr")


def make_features(train_rows, test_rows, mode):
    vec = TfidfVectorizer(
        ngram_range=(1,2),
        min_df=2,
        max_features=5000,
        sublinear_tf=True,
        stop_words="english",
    )

    Xq_tr = vec.fit_transform([r["query"] for r in train_rows])
    Xq_te = vec.transform([r["query"] for r in test_rows])

    Rt = role_matrix(train_rows)
    Re = role_matrix(test_rows)

    Ct = context_matrix(train_rows)
    Ce = context_matrix(test_rows)

    if mode == "target_only":
        return Rt, Re

    if mode == "q_only":
        return Xq_tr, Xq_te

    if mode == "q_edit":
        It = interaction_blocks(Xq_tr, Rt)
        Ie = interaction_blocks(Xq_te, Re)

        return (
            sparse.hstack([Xq_tr, Rt, It], format="csr"),
            sparse.hstack([Xq_te, Re, Ie], format="csr"),
        )

    if mode == "q_edit_context":
        Irt = interaction_blocks(Xq_tr, Rt)
        Ire = interaction_blocks(Xq_te, Re)

        Ict = interaction_blocks(Xq_tr, Ct)
        Ice = interaction_blocks(Xq_te, Ce)

        return (
            sparse.hstack(
                [Xq_tr, Rt, Ct, Irt, Ict],
                format="csr",
            ),
            sparse.hstack(
                [Xq_te, Re, Ce, Ire, Ice],
                format="csr",
            ),
        )

    raise ValueError(mode)


def rho(y,p):
    if len(y) < 4:
        return np.nan
    return float(spearmanr(y,p).statistic)


def top20_gain(y,p):
    n = max(1, int(np.ceil(.20*len(y))))
    idx = np.argsort(p)[-n:]
    return float(np.mean(y[idx]) - np.mean(y))


def fit_predict(train_rows, ytr, test_rows, mode):
    Xtr, Xte = make_features(train_rows, test_rows, mode)

    model = Ridge(
        alpha=20.0,
        solver="lsqr",
    )
    model.fit(Xtr, ytr)

    return model.predict(Xte)


def grouped_oof(rows, y, mode):
    groups = np.asarray([r["group"] for r in rows])
    unique = np.unique(groups)

    nsplit = min(5, len(unique))
    if nsplit < 3:
        raise RuntimeError("Too few groups")

    gkf = GroupKFold(n_splits=nsplit)
    pred = np.full(len(rows), np.nan)

    dummy = np.zeros(len(rows))

    for tr,te in gkf.split(dummy, y, groups):
        train_rows = [rows[i] for i in tr]
        test_rows = [rows[i] for i in te]

        pred[te] = fit_predict(
            train_rows,
            y[tr],
            test_rows,
            mode,
        )

    return pred


def paired_perm_delta(y, pa, pb, groups, B=10000):
    """
    Null: the two models are exchangeable within held-out query groups.
    Statistic: rho(pb)-rho(pa).
    """
    obs = rho(y,pb) - rho(y,pa)

    uniq = sorted(set(groups))
    by = defaultdict(list)

    for i,g in enumerate(groups):
        by[g].append(i)

    rng = np.random.default_rng(SEED)
    vals = []

    for _ in range(B):
        a = pa.copy()
        b = pb.copy()

        for g in uniq:
            if rng.random() < .5:
                idx = by[g]
                tmp = a[idx].copy()
                a[idx] = b[idx]
                b[idx] = tmp

        vals.append(rho(y,b)-rho(y,a))

    vals = np.asarray(vals)

    p = float(
        (1 + np.sum(vals >= obs - 1e-12))
        / (1 + len(vals))
    )

    return float(obs), p


def cross_source(rows, y, train_source, test_source, mode):
    tr = np.asarray([
        i for i,r in enumerate(rows)
        if r["source"] == train_source
    ])

    te = np.asarray([
        i for i,r in enumerate(rows)
        if r["source"] == test_source
    ])

    if len(tr) < 20 or len(te) < 20:
        return None

    pred = fit_predict(
        [rows[i] for i in tr],
        y[tr],
        [rows[i] for i in te],
        mode,
    )

    return {
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "rho": rho(y[te], pred),
        "top20_gain": top20_gain(y[te], pred),
    }


def main():
    rows = []

    rows += load_source(
        "artifacts/v13_probe_disambiguation.jsonl",
        "v13",
        "specific_pairwise_training_only",
    )

    rows += load_source(
        "artifacts/v11_probe_discovery.jsonl",
        "v11",
        "specific_pairwise",
    )

    if len(rows) < 100:
        raise RuntimeError(
            f"Only {len(rows)} usable edit-probe cases"
        )

    y = source_rank_targets(rows)
    groups = [r["group"] for r in rows]

    modes = [
        "target_only",
        "q_only",
        "q_edit",
        "q_edit_context",
    ]

    preds = {}
    results = {}

    for mode in modes:
        print(f"[OOF] {mode}", flush=True)
        p = grouped_oof(rows, y, mode)
        preds[mode] = p

        results[mode] = {
            "rho": rho(y,p),
            "top20_gain": top20_gain(y,p),
        }

    d_edit, p_edit = paired_perm_delta(
        y,
        preds["q_only"],
        preds["q_edit"],
        groups,
    )

    d_ctx, p_ctx = paired_perm_delta(
        y,
        preds["q_only"],
        preds["q_edit_context"],
        groups,
    )

    d_ctx_vs_edit, p_ctx_vs_edit = paired_perm_delta(
        y,
        preds["q_edit"],
        preds["q_edit_context"],
        groups,
    )

    transfers = {}

    for a,b in (
        ("v11","v13"),
        ("v13","v11"),
    ):
        key = f"{a}_to_{b}"
        transfers[key] = {}

        for mode in modes:
            transfers[key][mode] = cross_source(
                rows,
                y,
                a,
                b,
                mode,
            )

    q_rho = results["q_only"]["rho"]
    qe_rho = results["q_edit"]["rho"]
    qec_rho = results["q_edit_context"]["rho"]

    cross_improvements = []

    for direction in transfers.values():
        q = direction["q_only"]
        c = direction["q_edit_context"]

        if q and c:
            cross_improvements.append(
                c["rho"] - q["rho"]
            )

    gate = (
        qec_rho >= q_rho + .10
        and p_ctx <= .05
        and len(cross_improvements) == 2
        and min(cross_improvements) >= .03
    )

    out = {
        "experiment":
            "edit_conditioned_probe_value_v0",

        "n_cases": len(rows),

        "source_counts": {
            s: sum(r["source"]==s for r in rows)
            for s in ("v11","v13")
        },

        "unique_query_groups":
            len(set(groups)),

        "oof": results,

        "paired_tests": {
            "q_edit_minus_q_only": {
                "rho_delta": d_edit,
                "p": p_edit,
            },

            "q_edit_context_minus_q_only": {
                "rho_delta": d_ctx,
                "p": p_ctx,
            },

            "q_edit_context_minus_q_edit": {
                "rho_delta": d_ctx_vs_edit,
                "p": p_ctx_vs_edit,
            },
        },

        "cross_source": transfers,

        "decision": {
            "pass": bool(gate),

            "criterion":
                "q+edit+context grouped-OOF rho >= "
                "q-only rho + .10, paired p<=.05, and "
                "cross-source improvement >=.03 in both "
                "V11→V13 and V13→V11.",
        },
    }

    od = Path("artifacts/edit_value_v0")
    od.mkdir(parents=True,exist_ok=True)

    (od/"summary.json").write_text(
        json.dumps(out,indent=2),
        encoding="utf-8",
    )

    L = []
    L.append("="*88)
    L.append("EDIT-CONDITIONED PROBE VALUE V0")
    L.append("="*88)
    L.append(
        f"cases={len(rows)} "
        f"groups={len(set(groups))} "
        f"V11={out['source_counts']['v11']} "
        f"V13={out['source_counts']['v13']}"
    )
    L.append("")
    L.append("GROUPED OUT-OF-FOLD")

    for m in modes:
        x = results[m]
        L.append(
            f"  {m:20s} "
            f"rho={x['rho']:+.4f} "
            f"top20-gain={x['top20_gain']:+.4f}"
        )

    L.append("")
    L.append("PAIRED MODEL IMPROVEMENTS")
    L.append(
        f"  q+edit - q: "
        f"delta-rho={d_edit:+.4f} p={p_edit:.5f}"
    )
    L.append(
        f"  q+edit+context - q: "
        f"delta-rho={d_ctx:+.4f} p={p_ctx:.5f}"
    )
    L.append(
        f"  q+edit+context - q+edit: "
        f"delta-rho={d_ctx_vs_edit:+.4f} "
        f"p={p_ctx_vs_edit:.5f}"
    )

    L.append("")
    L.append("CROSS-SOURCE TRANSFER")

    for name,d in transfers.items():
        L.append(f"  [{name}]")

        for m in modes:
            x = d[m]

            if x is None:
                L.append(f"    {m:20s} unavailable")
            else:
                L.append(
                    f"    {m:20s} "
                    f"rho={x['rho']:+.4f} "
                    f"top20-gain={x['top20_gain']:+.4f}"
                )

    L.append("")
    L.append("="*88)

    if gate:
        L.append(
            "OVERALL: GO — probe informativeness is materially "
            "conditional on the structural edit/context. "
            "Proceed to Graph-Edit Information Geometry."
        )
    else:
        L.append(
            "OVERALL: NO-GO — edit/context conditioning does not "
            "materially outperform intrinsic q-only informativeness."
        )

    L.append("="*88)

    report = "\n".join(L)

    (od/"REPORT.txt").write_text(
        report+"\n",
        encoding="utf-8",
    )

    print(report)


if __name__ == "__main__":
    main()
