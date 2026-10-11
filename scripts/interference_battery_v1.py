from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from v11.role_universe import TRAIN_ROLES


ROLES = list(TRAIN_ROLES)
EPS = 1e-12

PAIR_TAGS = {
    "13": (0, 2),
    "24": (1, 3),
    "12": (0, 1),
    "34": (2, 3),
    "14": (0, 3),
    "23": (1, 2),
}
PAIR_NAMES = list(PAIR_TAGS)
TAGS7 = ["all", "13", "24", "12", "34", "14", "23"]
STYLES = ["numbered", "requirements"]
CHANNELS = ["v135", "v134", "concat"]


# ---------------------------------------------------------------------
# IO / generic utilities
# ---------------------------------------------------------------------

def read_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def sid_of(r):
    return str(r.get("system_id") or r.get("session_id") or "")


def cosine(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < EPS or nb < EPS:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def center(v):
    v = np.asarray(v, dtype=float)
    return v - np.mean(v)


def zscore(v):
    v = np.asarray(v, dtype=float)
    s = np.std(v)
    if s < EPS:
        return np.zeros_like(v)
    return (v - np.mean(v)) / s


def safe_spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return np.nan, np.nan
    r, p = spearmanr(x[mask], y[mask])
    return float(r), float(p)


def bootstrap_ci(values, B=10000, seed=7341):
    x = np.asarray([v for v in values if np.isfinite(v)], float)
    if len(x) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    vals = np.empty(B)
    for b in range(B):
        vals[b] = np.mean(rng.choice(x, len(x), replace=True))
    return [
        float(np.quantile(vals, .025)),
        float(np.quantile(vals, .975)),
    ]


def exact_signflip(values):
    x = np.asarray([v for v in values if np.isfinite(v)], float)
    n = len(x)
    if n == 0:
        return np.nan
    obs = abs(float(np.mean(x)))

    if n <= 16:
        vals = []
        for signs in itertools.product([-1.0, 1.0], repeat=n):
            vals.append(abs(float(np.mean(x * np.asarray(signs)))))
        vals = np.asarray(vals)
        return float(np.mean(vals >= obs - 1e-12))

    rng = np.random.default_rng(7391)
    vals = []
    for _ in range(20000):
        signs = rng.choice([-1.0, 1.0], size=n)
        vals.append(abs(np.mean(x * signs)))
    return float(np.mean(np.asarray(vals) >= obs - 1e-12))


def scalar_value(x):
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if isinstance(x, dict):
        for k in (
            "logit", "score", "value", "z",
            "raw_logit", "evidence", "probability", "prob",
        ):
            if k in x:
                try:
                    return float(x[k])
                except Exception:
                    pass

        nums = [
            float(v)
            for v in x.values()
            if isinstance(v, (int, float, np.integer, np.floating))
        ]
        if len(nums) == 1:
            return nums[0]
    return None


def mapping_to_vec(obj):
    if obj is None:
        return None

    if isinstance(obj, (list, tuple)):
        if len(obj) == len(ROLES):
            try:
                return np.asarray(obj, dtype=float)
            except Exception:
                pass

    if not isinstance(obj, dict):
        return None

    hits = sum(r in obj for r in ROLES)
    if hits >= max(3, int(.7 * len(ROLES))):
        vals = []
        for r in ROLES:
            v = scalar_value(obj.get(r))
            if v is None:
                return None
            vals.append(v)
        return np.asarray(vals, dtype=float)

    for k in (
        "role_scores", "scores", "logits",
        "evidence", "role_logits", "values", "roles",
    ):
        if k in obj:
            v = mapping_to_vec(obj[k])
            if v is not None:
                return v

    return None


def channel_vec(row, channel):
    md = row.get("metadata") or {}

    if channel == "v135":
        choices = [
            row.get("evidence"),
            row.get("evidence_v135"),
            md.get("evidence_v135"),
            md.get("causal_evidence"),
        ]
    elif channel == "v134":
        choices = [
            row.get("evidence_v134"),
            md.get("evidence_v134"),
            md.get("presence_evidence"),
            row.get("presence_evidence"),
        ]
    else:
        raise ValueError(channel)

    for obj in choices:
        v = mapping_to_vec(obj)
        if v is not None:
            return v
    return None


def deep_selected(obj):
    if isinstance(obj, dict):
        for k in ("selected_roles", "selected_roles_training_only"):
            if k in obj:
                x = obj[k]
                if isinstance(x, str):
                    return (x,)
                try:
                    return tuple(map(str, x or []))
                except Exception:
                    pass
        for v in obj.values():
            ans = deep_selected(v)
            if ans is not None:
                return ans
    elif isinstance(obj, list):
        for v in obj:
            ans = deep_selected(v)
            if ans is not None:
                return ans
    return None


def deep_truth_roles(obj):
    if isinstance(obj, dict):
        for k in ("true_roles", "truth_roles", "truth_roles_training_only"):
            if k in obj:
                x = obj[k]
                if isinstance(x, str):
                    return (x,)
                try:
                    return tuple(map(str, x or []))
                except Exception:
                    pass
        for v in obj.values():
            ans = deep_truth_roles(v)
            if ans is not None:
                return ans
    return None


def jaccard(a, b):
    a, b = set(a or ()), set(b or ())
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


# ---------------------------------------------------------------------
# Load experiment
# ---------------------------------------------------------------------

def load_everything(root: Path, src: Path):
    design = json.loads((root / "design.json").read_text(encoding="utf-8"))

    src_cand = read_jsonl(src / "candidates_train_round_01.jsonl")
    src_score = read_jsonl(src / "scored_train_round_01.jsonl")
    mux_score = read_jsonl(root / "scored_train_round_01.jsonl")

    src_resp = read_jsonl(src / "responses_train_round_01.jsonl")
    mux_resp = read_jsonl(root / "responses_train_round_01.jsonl")

    source_qidx = {}
    for r in src_cand:
        sid = str(r.get("session_id") or "")
        q = str(r.get("query") or "")
        try:
            idx = int(r.get("candidate_index"))
        except Exception:
            continue
        if sid and q and 0 <= idx < 4:
            source_qidx[(sid, q)] = idx

    design_q = {
        (str(e["session_id"]), str(e["query"])): e
        for e in design["entries"]
    }

    orig = {c: defaultdict(dict) for c in CHANNELS}
    truth_roles = {}

    for r in src_score:
        sid = sid_of(r)
        q = str(r.get("query") or "")
        idx = source_qidx.get((sid, q))
        if idx is None:
            continue

        a = channel_vec(r, "v135")
        b = channel_vec(r, "v134")

        vals = {
            "v135": a,
            "v134": b,
            "concat": (
                np.concatenate([a, b])
                if a is not None and b is not None else None
            ),
        }
        for c, v in vals.items():
            if v is not None:
                orig[c][sid][idx] = v

        tr = deep_truth_roles(r)
        if tr:
            truth_roles[sid] = tuple(tr)

    mux = {
        c: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for c in CHANNELS
    }

    for r in mux_score:
        sid = sid_of(r)
        q = str(r.get("query") or "")
        spec = design_q.get((sid, q))
        if spec is None:
            continue

        a = channel_vec(r, "v135")
        b = channel_vec(r, "v134")

        vals = {
            "v135": a,
            "v134": b,
            "concat": (
                np.concatenate([a, b])
                if a is not None and b is not None else None
            ),
        }

        for c, v in vals.items():
            if v is not None:
                mux[c][sid][spec["style"]][spec["tag"]][spec["order"]] = v

        tr = deep_truth_roles(r)
        if tr:
            truth_roles[sid] = tuple(tr)

    # Public outputs.
    outputs = {}
    for r in src_resp + mux_resp:
        sid = sid_of(r)
        q = str(r.get("query") or "")
        out = str(r.get("output") or r.get("response") or "")
        if sid and q:
            outputs[(sid, q)] = out

    # Manifest: mode / K / true role set.
    metadata = {}
    manifest_path = Path(
        "artifacts/adaptive_v11_active_v1/manifest_privileged.json"
    )
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for s in manifest.get("systems", []):
            sid = str(s.get("session_id") or "")
            rr = (
                s.get("roles")
                or s.get("worker_roles")
                or s.get("true_roles")
                or []
            )
            mode = (
                s.get("execution_mode")
                or s.get("mode")
                or s.get("routing_mode")
            )
            metadata[sid] = {
                "roles": tuple(map(str, rr)),
                "mode": str(mode) if mode is not None else None,
                "K": len(rr) if rr else None,
            }
            if rr:
                truth_roles[sid] = tuple(map(str, rr))

    # Scan likely cache files for selected-role truth.
    scan_dirs = [
        root,
        src,
        Path("artifacts/adaptive_v11_active_v1"),
    ]

    rows_scan = []
    for d in scan_dirs:
        if not d.exists():
            continue
        for p in d.glob("*.jsonl"):
            lname = p.name.lower()
            if any(x in lname for x in (
                "response", "truth", "cache", "scored"
            )):
                rows_scan.extend(read_jsonl(p))

    # Resolve rows lacking query through public key.
    key_to_query = {}
    for r in rows_scan:
        sid = sid_of(r)
        q = str(r.get("query") or "")
        for kname in ("key", "query_hash"):
            kval = str(r.get(kname) or "")
            if sid and q and kval:
                key_to_query[(sid, kval)] = q

    selected = {}
    for r in rows_scan:
        sel = deep_selected(r)
        if sel is None:
            continue
        sid = sid_of(r)
        q = str(r.get("query") or "")
        if not q:
            for kname in ("key", "query_hash"):
                kval = str(r.get(kname) or "")
                if (sid, kval) in key_to_query:
                    q = key_to_query[(sid, kval)]
                    break
        if sid and q:
            selected[(sid, q)] = tuple(sel)

    return {
        "design": design,
        "design_q": design_q,
        "source_qidx": source_qidx,
        "orig": orig,
        "mux": mux,
        "truth_roles": truth_roles,
        "metadata": metadata,
        "outputs": outputs,
        "selected": selected,
        "src_cand": src_cand,
    }


# ---------------------------------------------------------------------
# Pairwise interference
# ---------------------------------------------------------------------

def make_pair_records(data):
    orig = data["orig"]
    mux = data["mux"]
    records = []

    for channel in CHANNELS:
        for sid in sorted(orig[channel]):
            if not all(i in orig[channel][sid] for i in range(4)):
                continue

            for style in STYLES:
                block = mux[channel][sid].get(style, {})

                for tag, (i, j) in PAIR_TAGS.items():
                    if (
                        tag not in block
                        or "fwd" not in block[tag]
                        or "rev" not in block[tag]
                    ):
                        continue

                    ei = orig[channel][sid][i]
                    ej = orig[channel][sid][j]
                    f = block[tag]["fwd"]
                    r = block[tag]["rev"]

                    cf, cr = center(f), center(r)
                    sym = .5 * (f + r)
                    anti = .5 * (f - r)

                    add = ei + ej
                    csym = center(sym)
                    cadd = center(add)

                    alpha = (
                        float(np.dot(csym, cadd) / (np.dot(cadd, cadd) + EPS))
                    )
                    resid = csym - alpha * cadd

                    order_cos = cosine(cf, cr)
                    anti_ratio = (
                        np.linalg.norm(cf - cr)
                        / (np.linalg.norm(cf) + np.linalg.norm(cr) + EPS)
                    )
                    orth_resid_ratio = (
                        np.linalg.norm(resid)
                        / (np.linalg.norm(csym) + EPS)
                    )

                    records.append({
                        "sid": sid,
                        "channel": channel,
                        "style": style,
                        "tag": tag,
                        "i": i,
                        "j": j,
                        "f": f,
                        "r": r,
                        "sym": sym,
                        "anti": anti,
                        "resid": resid,
                        "order_cos_centered": order_cos,
                        "anti_ratio": float(anti_ratio),
                        "orth_resid_ratio": float(orth_resid_ratio),
                        "best_additive_scale": alpha,
                    })
    return records


def aggregate_pairwise(records):
    out = {}

    for style in STYLES:
        out[style] = {}
        for channel in CHANNELS:
            rr = [
                x for x in records
                if x["style"] == style and x["channel"] == channel
            ]

            bysid = defaultdict(list)
            for x in rr:
                bysid[x["sid"]].append(x)

            system_anti = [
                np.mean([x["anti_ratio"] for x in xs])
                for xs in bysid.values()
            ]
            system_resid = [
                np.mean([x["orth_resid_ratio"] for x in xs])
                for xs in bysid.values()
            ]
            system_cos = [
                np.mean([x["order_cos_centered"] for x in xs])
                for xs in bysid.values()
            ]

            out[style][channel] = {
                "n_pairs": len(rr),
                "n_systems": len(bysid),
                "anti_ratio_mean": float(np.mean(system_anti)) if system_anti else None,
                "anti_ratio_ci95": bootstrap_ci(system_anti),
                "nonadditive_orth_ratio_mean": float(np.mean(system_resid)) if system_resid else None,
                "nonadditive_orth_ratio_ci95": bootstrap_ci(system_resid),
                "centered_order_cosine_mean": float(np.mean(system_cos)) if system_cos else None,
            }

    # Direct paired causal-vs-absolute comparisons.
    contrasts = {}
    for style in STYLES:
        by = {}
        for channel in ("v135", "v134"):
            tmp = defaultdict(list)
            for r in records:
                if r["style"] == style and r["channel"] == channel:
                    tmp[r["sid"]].append(r)
            by[channel] = {
                sid: {
                    "anti": np.mean([x["anti_ratio"] for x in xs]),
                    "resid": np.mean([x["orth_resid_ratio"] for x in xs]),
                }
                for sid, xs in tmp.items()
            }

        common = sorted(set(by["v135"]) & set(by["v134"]))
        anti_diff = [
            by["v135"][s]["anti"] - by["v134"][s]["anti"]
            for s in common
        ]
        resid_diff = [
            by["v135"][s]["resid"] - by["v134"][s]["resid"]
            for s in common
        ]

        contrasts[style] = {
            "n_systems": len(common),
            "v135_minus_v134_anti_ratio": float(np.mean(anti_diff)) if anti_diff else None,
            "anti_exact_signflip_p": exact_signflip(anti_diff),
            "v135_minus_v134_nonadditive_ratio": float(np.mean(resid_diff)) if resid_diff else None,
            "nonadditive_exact_signflip_p": exact_signflip(resid_diff),
        }

    return out, contrasts


# ---------------------------------------------------------------------
# Cross-wrapper invariance
# ---------------------------------------------------------------------

def cross_wrapper(records, channel, field):
    num = {
        (r["sid"], r["tag"]): r
        for r in records
        if r["channel"] == channel and r["style"] == "numbered"
    }
    req = {
        (r["sid"], r["tag"]): r
        for r in records
        if r["channel"] == channel and r["style"] == "requirements"
    }

    common = sorted(set(num) & set(req))
    if not common:
        return {}

    vec_name = "anti" if field == "anti" else "resid"

    obs_vals = [
        cosine(center(num[k][vec_name]), center(req[k][vec_name]))
        for k in common
    ]
    obs = float(np.nanmean(obs_vals))

    # Pair-label permutation within each system.
    rng = np.random.default_rng(9137)
    sids = sorted({s for s, _ in common})
    null = []

    for _ in range(20000):
        vals = []
        for sid in sids:
            tags = [
                t for t in PAIR_NAMES
                if (sid, t) in num and (sid, t) in req
            ]
            perm = list(rng.permutation(tags))
            for t, tp in zip(tags, perm):
                vals.append(
                    cosine(
                        center(num[(sid, t)][vec_name]),
                        center(req[(sid, tp)][vec_name]),
                    )
                )
        null.append(np.nanmean(vals))

    null = np.asarray(null)
    p = float(
        (1 + np.sum(null >= obs - 1e-12))
        / (1 + len(null))
    )

    # Also compare rankings of interaction magnitudes across six pairs.
    rank_rhos = []
    for sid in sids:
        tags = [
            t for t in PAIR_NAMES
            if (sid, t) in num and (sid, t) in req
        ]
        if len(tags) < 4:
            continue

        if field == "anti":
            a = [num[(sid, t)]["anti_ratio"] for t in tags]
            b = [req[(sid, t)]["anti_ratio"] for t in tags]
        else:
            a = [num[(sid, t)]["orth_resid_ratio"] for t in tags]
            b = [req[(sid, t)]["orth_resid_ratio"] for t in tags]

        rho, _ = safe_spearman(a, b)
        rank_rhos.append(rho)

    return {
        "n_pairs": len(common),
        "mean_same_pair_cosine": obs,
        "pair_label_permutation_p": p,
        "mean_within_system_pair_rank_spearman": float(np.nanmean(rank_rhos)),
        "per_system_rank_spearman": rank_rhos,
    }


# ---------------------------------------------------------------------
# Higher-order closure
# ---------------------------------------------------------------------

def higher_order(data):
    orig, mux = data["orig"], data["mux"]
    out = {}

    for style in STYLES:
        out[style] = {}
        for channel in CHANNELS:
            rows = []

            for sid in sorted(orig[channel]):
                if not all(i in orig[channel][sid] for i in range(4)):
                    continue

                block = mux[channel][sid].get(style, {})
                if not (
                    "all" in block
                    and "fwd" in block["all"]
                    and "rev" in block["all"]
                    and all(
                        t in block and "fwd" in block[t] and "rev" in block[t]
                        for t in PAIR_NAMES
                    )
                ):
                    continue

                all4 = .5 * (
                    block["all"]["fwd"] + block["all"]["rev"]
                )
                first_order = sum(
                    orig[channel][sid][i] for i in range(4)
                )
                pair_syms = [
                    .5 * (block[t]["fwd"] + block[t]["rev"])
                    for t in PAIR_NAMES
                ]

                # Each primitive appears in exactly 3 of the six pairs.
                pair_closure = sum(pair_syms) / 3.0

                ca = center(all4)
                cf = center(first_order)
                cp = center(pair_closure)

                rows.append({
                    "sid": sid,
                    "all4_vs_firstorder_cos": cosine(ca, cf),
                    "all4_vs_pairclosure_cos": cosine(ca, cp),
                    "pairclosure_residual_ratio": float(
                        np.linalg.norm(ca - (
                            np.dot(ca, cp) / (np.dot(cp, cp) + EPS)
                        ) * cp)
                        / (np.linalg.norm(ca) + EPS)
                    ),
                })

            out[style][channel] = {
                "n": len(rows),
                "all4_vs_firstorder_cos_mean": (
                    float(np.nanmean([
                        x["all4_vs_firstorder_cos"] for x in rows
                    ])) if rows else None
                ),
                "all4_vs_pairclosure_cos_mean": (
                    float(np.nanmean([
                        x["all4_vs_pairclosure_cos"] for x in rows
                    ])) if rows else None
                ),
                "pairclosure_residual_ratio_mean": (
                    float(np.nanmean([
                        x["pairclosure_residual_ratio"] for x in rows
                    ])) if rows else None
                ),
            }

    return out


# ---------------------------------------------------------------------
# Role concentration and one-call compression
# ---------------------------------------------------------------------

def role_concentration(records, truth_roles):
    out = {}

    for style in STYLES:
        out[style] = {}

        for channel in ("v135", "v134"):
            sysvals = defaultdict(lambda: {"anti": [], "resid": []})

            for r in records:
                if r["style"] != style or r["channel"] != channel:
                    continue
                sid = r["sid"]
                truth = set(truth_roles.get(sid, ()))
                if not truth:
                    continue

                mask = np.asarray([role in truth for role in ROLES])
                if mask.all() or (~mask).all():
                    continue

                av = np.abs(r["anti"])
                rv = np.abs(r["resid"])

                sysvals[sid]["anti"].append(
                    np.mean(av[mask]) - np.mean(av[~mask])
                )
                sysvals[sid]["resid"].append(
                    np.mean(rv[mask]) - np.mean(rv[~mask])
                )

            anti = [
                np.mean(x["anti"])
                for x in sysvals.values() if x["anti"]
            ]
            resid = [
                np.mean(x["resid"])
                for x in sysvals.values() if x["resid"]
            ]

            out[style][channel] = {
                "n_systems": len(sysvals),
                "true_minus_false_abs_anti": float(np.mean(anti)) if anti else None,
                "anti_signflip_p": exact_signflip(anti),
                "true_minus_false_abs_residual": float(np.mean(resid)) if resid else None,
                "residual_signflip_p": exact_signflip(resid),
            }

    return out


def role_ranking_metrics(score_by_sid, truth_roles):
    ys, ps = [], []
    topk = []

    for sid, v in score_by_sid.items():
        truth = set(truth_roles.get(sid, ()))
        if not truth or len(v) != len(ROLES):
            continue

        zv = zscore(v)
        labels = np.asarray([int(r in truth) for r in ROLES])

        ys.extend(labels.tolist())
        ps.extend(zv.tolist())

        K = len(truth)
        idx = np.argsort(v)[::-1][:K]
        pred = {ROLES[i] for i in idx}
        topk.append(len(pred & truth) / K)

    if len(set(ys)) < 2:
        return {}

    return {
        "systems": len(topk),
        "auc_within_system_z": float(roc_auc_score(ys, ps)),
        "average_precision_within_system_z": float(
            average_precision_score(ys, ps)
        ),
        "oracle_K_role_recall": float(np.mean(topk)),
    }


def compression(data):
    orig, mux = data["orig"], data["mux"]
    truth = data["truth_roles"]
    out = {}

    for channel in ("v135", "v134"):
        out[channel] = {}

        baseline = {}
        for sid, d in orig[channel].items():
            if all(i in d for i in range(4)):
                baseline[sid] = np.mean(
                    np.stack([d[i] for i in range(4)]), axis=0
                )

        out[channel]["four_individual_calls_mean"] = role_ranking_metrics(
            baseline, truth
        )

        for style in STYLES:
            b_fwd, b_rev, b_avg = {}, {}, {}

            for sid, block0 in mux[channel].items():
                block = block0.get(style, {})
                if "all" not in block:
                    continue
                if "fwd" in block["all"]:
                    b_fwd[sid] = block["all"]["fwd"]
                if "rev" in block["all"]:
                    b_rev[sid] = block["all"]["rev"]
                if "fwd" in block["all"] and "rev" in block["all"]:
                    b_avg[sid] = .5 * (
                        block["all"]["fwd"] + block["all"]["rev"]
                    )

            out[channel][f"{style}_all4_ONE_CALL_fwd"] = role_ranking_metrics(
                b_fwd, truth
            )
            out[channel][f"{style}_all4_ONE_CALL_rev"] = role_ranking_metrics(
                b_rev, truth
            )
            out[channel][f"{style}_all4_TWO_CALL_avg"] = role_ranking_metrics(
                b_avg, truth
            )

    return out


# ---------------------------------------------------------------------
# Learned demultiplexing, strictly held-out by system
# ---------------------------------------------------------------------

def measurements_for(data, sid, channel, style):
    block = data["mux"][channel][sid].get(style, {})
    if not all(
        t in block and "fwd" in block[t] and "rev" in block[t]
        for t in TAGS7
    ):
        return None

    return {
        t: .5 * (block[t]["fwd"] + block[t]["rev"])
        for t in TAGS7
    }


def fixed_decode(meas):
    m0 = meas["all"]
    m1 = meas["13"] - meas["24"]
    m2 = meas["12"] - meas["34"]
    m3 = meas["14"] - meas["23"]

    return np.stack([
        (m0 + m1 + m2 + m3) / 4.,
        (m0 - m1 + m2 - m3) / 4.,
        (m0 + m1 - m2 - m3) / 4.,
        (m0 - m1 - m2 + m3) / 4.,
    ])


def component_metrics(pred, truth):
    pred = np.asarray(pred, float)   # 4 x D
    truth = np.asarray(truth, float)

    pc = pred - np.mean(pred, axis=0, keepdims=True)
    tc = truth - np.mean(truth, axis=0, keepdims=True)

    S = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            S[i, j] = cosine(pc[i], tc[j])

    top1 = float(np.mean(np.argmax(S, axis=1) == np.arange(4)))
    diag = float(np.nanmean(np.diag(S)))

    margin = []
    for i in range(4):
        wrong = [S[i, j] for j in range(4) if j != i]
        margin.append(S[i, i] - max(wrong))

    return {
        "top1": top1,
        "diag_cos": diag,
        "margin": float(np.mean(margin)),
    }


def decoder_loso(data, channel, train_style, test_style):
    orig = data["orig"][channel]

    sids = [
        sid for sid in sorted(orig)
        if all(i in orig[sid] for i in range(4))
        and measurements_for(data, sid, channel, train_style) is not None
        and measurements_for(data, sid, channel, test_style) is not None
    ]

    if len(sids) < 4:
        return {}

    learned, fixed = [], []

    for held in sids:
        X, Y = [], []

        for sid in sids:
            if sid == held:
                continue

            meas = measurements_for(data, sid, channel, train_style)
            D = len(next(iter(meas.values())))

            for d in range(D):
                X.append([meas[t][d] for t in TAGS7])
                Y.append([orig[sid][k][d] for k in range(4)])

        X = np.asarray(X, float)
        Y = np.asarray(Y, float)

        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-8] = 1.

        model = Ridge(alpha=1.0)
        model.fit((X - mu) / sd, Y)

        mt = measurements_for(data, held, channel, test_style)
        D = len(next(iter(mt.values())))

        Xt = np.asarray([
            [mt[t][d] for t in TAGS7]
            for d in range(D)
        ])
        pred = model.predict((Xt - mu) / sd).T

        truth = np.stack([orig[held][k] for k in range(4)])

        learned.append(component_metrics(pred, truth))
        fixed.append(component_metrics(fixed_decode(mt), truth))

    def agg(rows):
        return {
            "n_systems": len(rows),
            "top1_mean": float(np.mean([x["top1"] for x in rows])),
            "diag_cos_mean": float(np.mean([x["diag_cos"] for x in rows])),
            "margin_mean": float(np.mean([x["margin"] for x in rows])),
        }

    return {
        "learned_shared_linear_decoder": agg(learned),
        "fixed_hadamard_decoder": agg(fixed),
    }


# ---------------------------------------------------------------------
# Text confound
# ---------------------------------------------------------------------

def text_order_analysis(data, records):
    outputs = data["outputs"]
    design_q = data["design_q"]

    docs = []
    keys = []

    for (sid, q), spec in design_q.items():
        out = outputs.get((sid, q), "")
        if out:
            keys.append((sid, q))
            docs.append(out)

    if len(docs) < 10:
        return {"available_outputs": len(docs)}

    tfidf = TfidfVectorizer(
        min_df=1,
        max_features=6000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X = tfidf.fit_transform(docs)
    index = {k: i for i, k in enumerate(keys)}

    # design lookup by sid/style/tag/order
    qlookup = {}
    for (sid, q), spec in design_q.items():
        qlookup[
            (sid, spec["style"], spec["tag"], spec["order"])
        ] = q

    rows = []

    for r in records:
        if r["channel"] != "v135":
            continue

        kf = (r["sid"], r["style"], r["tag"], "fwd")
        kr = (r["sid"], r["style"], r["tag"], "rev")

        qf = qlookup.get(kf)
        qr = qlookup.get(kr)
        if not qf or not qr:
            continue

        if (r["sid"], qf) not in index or (r["sid"], qr) not in index:
            continue

        i = index[(r["sid"], qf)]
        j = index[(r["sid"], qr)]

        sim = cosine_similarity(X[i], X[j])[0, 0]
        of = outputs[(r["sid"], qf)]
        or_ = outputs[(r["sid"], qr)]

        rows.append({
            "sid": r["sid"],
            "style": r["style"],
            "tag": r["tag"],
            "anti_ratio": r["anti_ratio"],
            "text_distance": float(1. - sim),
            "length_distance": abs(len(of) - len(or_))
                / max(1., .5 * (len(of) + len(or_))),
        })

    out = {"n_pairs": len(rows)}

    for style in STYLES:
        rr = [x for x in rows if x["style"] == style]
        rho_t, p_t = safe_spearman(
            [x["anti_ratio"] for x in rr],
            [x["text_distance"] for x in rr],
        )
        rho_l, p_l = safe_spearman(
            [x["anti_ratio"] for x in rr],
            [x["length_distance"] for x in rr],
        )

        out[style] = {
            "anti_vs_tfidf_output_distance_spearman": rho_t,
            "p": p_t,
            "anti_vs_length_distance_spearman": rho_l,
            "length_p": p_l,
        }

    return out


# ---------------------------------------------------------------------
# Privileged-trace evaluation
# ---------------------------------------------------------------------

def trace_analysis(data, records):
    selected = data["selected"]
    src_cand = data["src_cand"]
    design_q = data["design_q"]

    src_query = {}
    for r in src_cand:
        sid = str(r.get("session_id") or "")
        q = str(r.get("query") or "")
        try:
            idx = int(r.get("candidate_index"))
        except Exception:
            continue
        if sid and q and idx in range(4):
            src_query[(sid, idx)] = q

    mux_query = {}
    for (sid, q), spec in design_q.items():
        mux_query[
            (sid, spec["style"], spec["tag"], spec["order"])
        ] = q

    rows = []

    for r in records:
        if r["channel"] != "v135":
            continue

        sid, style, tag = r["sid"], r["style"], r["tag"]
        i, j = r["i"], r["j"]

        qi = src_query.get((sid, i))
        qj = src_query.get((sid, j))
        qf = mux_query.get((sid, style, tag, "fwd"))
        qr = mux_query.get((sid, style, tag, "rev"))

        ti = selected.get((sid, qi), ()) if qi else ()
        tj = selected.get((sid, qj), ()) if qj else ()
        tf = selected.get((sid, qf), ()) if qf else ()
        tr = selected.get((sid, qr), ()) if qr else ()

        if not (ti or tj or tf or tr):
            continue

        rows.append({
            "sid": sid,
            "style": style,
            "tag": tag,
            "anti_ratio": r["anti_ratio"],
            "resid_ratio": r["orth_resid_ratio"],
            "basis_trace_overlap": jaccard(ti, tj)
                if ti and tj else np.nan,
            "forward_reverse_trace_change": 1. - jaccard(tf, tr)
                if tf and tr else np.nan,
            "selected_count_change": abs(len(tf) - len(tr))
                if tf and tr else np.nan,
        })

    out = {
        "selected_query_rows_found": len(selected),
        "usable_pair_rows": len(rows),
    }

    for style in STYLES:
        rr = [x for x in rows if x["style"] == style]

        r1, p1 = safe_spearman(
            [x["anti_ratio"] for x in rr],
            [x["forward_reverse_trace_change"] for x in rr],
        )
        r2, p2 = safe_spearman(
            [x["resid_ratio"] for x in rr],
            [1. - x["basis_trace_overlap"]
             if np.isfinite(x["basis_trace_overlap"]) else np.nan
             for x in rr],
        )

        out[style] = {
            "n": len(rr),
            "anti_vs_actual_route_change_spearman": r1,
            "anti_vs_actual_route_change_p": p1,
            "nonadditive_vs_basis_trace_difference_spearman": r2,
            "nonadditive_vs_basis_trace_difference_p": p2,
        }

    return out


# ---------------------------------------------------------------------
# Mode / K fingerprints
# ---------------------------------------------------------------------

def system_fingerprints(records, metadata):
    sids = sorted({
        r["sid"] for r in records
        if r["channel"] == "v135"
    })

    feats, labels_mode, labels_k, used = [], [], [], []

    for sid in sids:
        f = []
        ok = True

        for style in STYLES:
            rr = [
                r for r in records
                if r["sid"] == sid
                and r["channel"] == "v135"
                and r["style"] == style
            ]
            if len(rr) != 6:
                ok = False
                break

            a = np.asarray([r["anti_ratio"] for r in rr])
            n = np.asarray([r["orth_resid_ratio"] for r in rr])
            c = np.asarray([r["order_cos_centered"] for r in rr])

            f.extend([
                np.mean(a), np.std(a), np.max(a),
                np.mean(n), np.std(n), np.max(n),
                np.mean(c),
            ])

        md = metadata.get(sid, {})
        if not ok or not md:
            continue
        if md.get("mode") is None or md.get("K") is None:
            continue

        feats.append(f)
        labels_mode.append(md["mode"])
        labels_k.append(float(md["K"]))
        used.append(sid)

    if len(feats) < 5:
        return {"n": len(feats)}

    X = np.asarray(feats, float)
    ym = np.asarray(labels_mode, object)
    yk = np.asarray(labels_k, float)

    # LOOCV 1-NN mode classification with train-only standardization.
    preds = []

    for h in range(len(X)):
        tr = np.asarray([i for i in range(len(X)) if i != h])
        mu = X[tr].mean(axis=0)
        sd = X[tr].std(axis=0)
        sd[sd < 1e-8] = 1.

        ztr = (X[tr] - mu) / sd
        zh = (X[h] - mu) / sd

        dist = np.linalg.norm(ztr - zh, axis=1)
        preds.append(ym[tr[np.argmin(dist)]])

    acc = float(np.mean(np.asarray(preds) == ym))

    # Permutation null.
    rng = np.random.default_rng(6641)
    null = []
    for _ in range(10000):
        yp = rng.permutation(ym)
        pp = []
        for h in range(len(X)):
            tr = np.asarray([i for i in range(len(X)) if i != h])
            mu = X[tr].mean(axis=0)
            sd = X[tr].std(axis=0)
            sd[sd < 1e-8] = 1.
            ztr = (X[tr] - mu) / sd
            zh = (X[h] - mu) / sd
            dist = np.linalg.norm(ztr - zh, axis=1)
            pp.append(yp[tr[np.argmin(dist)]])
        null.append(np.mean(np.asarray(pp) == yp))

    p_mode = float(
        (1 + np.sum(np.asarray(null) >= acc - 1e-12))
        / (1 + len(null))
    )

    # Strict LOSO Ridge for K.
    kp = []
    baseline = []
    for h in range(len(X)):
        tr = np.asarray([i for i in range(len(X)) if i != h])
        mu = X[tr].mean(axis=0)
        sd = X[tr].std(axis=0)
        sd[sd < 1e-8] = 1.

        model = Ridge(alpha=10.)
        model.fit((X[tr] - mu) / sd, yk[tr])
        kp.append(float(model.predict(((X[h] - mu) / sd)[None])[0]))
        baseline.append(float(np.mean(yk[tr])))

    return {
        "n": len(X),
        "systems": used,
        "modes": labels_mode,
        "K": labels_k,
        "mode_1nn_loso_accuracy": acc,
        "mode_permutation_p": p_mode,
        "K_ridge_loso_MAE": float(np.mean(np.abs(np.asarray(kp) - yk))),
        "K_mean_baseline_MAE": float(
            np.mean(np.abs(np.asarray(baseline) - yk))
        ),
        "K_pred": kp,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

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
    outdir = root / "interference_battery_v1"
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/9] loading", flush=True)
    data = load_everything(root, src)

    print("[2/9] pairwise decomposition", flush=True)
    records = make_pair_records(data)
    pair_summary, channel_contrast = aggregate_pairwise(records)

    print("[3/9] cross-wrapper invariance", flush=True)
    wrapper = {}
    for ch in CHANNELS:
        wrapper[ch] = {
            "antisymmetric_order_component": cross_wrapper(
                records, ch, "anti"
            ),
            "nonadditive_symmetric_residual": cross_wrapper(
                records, ch, "resid"
            ),
        }

    print("[4/9] higher-order closure", flush=True)
    closure = higher_order(data)

    print("[5/9] truth-role localization", flush=True)
    concentration = role_concentration(
        records, data["truth_roles"]
    )

    print("[6/9] compressed role extraction", flush=True)
    comp = compression(data)

    print("[7/9] LOSO learned demultiplexing", flush=True)
    decoders = {}
    for ch in CHANNELS:
        decoders[ch] = {
            "numbered_to_numbered": decoder_loso(
                data, ch, "numbered", "numbered"
            ),
            "requirements_to_requirements": decoder_loso(
                data, ch, "requirements", "requirements"
            ),
            "numbered_to_REQUIREMENTS_cross_wrapper": decoder_loso(
                data, ch, "numbered", "requirements"
            ),
            "requirements_to_NUMBERED_cross_wrapper": decoder_loso(
                data, ch, "requirements", "numbered"
            ),
        }

    print("[8/9] text + privileged trace tests", flush=True)
    text = text_order_analysis(data, records)
    trace = trace_analysis(data, records)

    print("[9/9] structural fingerprint tests", flush=True)
    fingerprint = system_fingerprints(
        records, data["metadata"]
    )

    summary = {
        "experiment": "semantic_interference_battery_v1",
        "root": str(root),
        "n_pair_records": len(records),
        "roles": ROLES,
        "pairwise": pair_summary,
        "causal_vs_absolute_channel_contrasts": channel_contrast,
        "cross_wrapper": wrapper,
        "higher_order_closure": closure,
        "truth_role_localization": concentration,
        "compression": comp,
        "decoder_loso": decoders,
        "text_confound": text,
        "privileged_trace_evaluation": trace,
        "execution_structure_fingerprint": fingerprint,
    }

    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Flat pair metrics.
    serial = []
    for r in records:
        serial.append({
            "sid": r["sid"],
            "channel": r["channel"],
            "style": r["style"],
            "tag": r["tag"],
            "i": r["i"],
            "j": r["j"],
            "order_cos_centered": r["order_cos_centered"],
            "anti_ratio": r["anti_ratio"],
            "orth_resid_ratio": r["orth_resid_ratio"],
            "best_additive_scale": r["best_additive_scale"],
        })

    with (outdir / "pair_metrics.jsonl").open("w", encoding="utf-8") as f:
        for r in serial:
            f.write(json.dumps(r) + "\n")

    # Human-readable report.
    L = []
    L.append("=" * 90)
    L.append("SEMANTIC INTERFERENCE BATTERY V1")
    L.append("=" * 90)
    L.append(f"pair records: {len(records)}")
    L.append(
        f"privileged selected-role query rows discovered: "
        f"{trace.get('selected_query_rows_found', 0)}"
    )
    L.append("")

    L.append("A) CAUSAL INTERFERENCE VS ABSOLUTE COMPOSITION")
    for style in STYLES:
        a = pair_summary[style]["v135"]
        b = pair_summary[style]["v134"]
        c = channel_contrast[style]

        L.append(f"  [{style}]")
        L.append(
            f"    V135 anti={a['anti_ratio_mean']:.4f} "
            f"nonadd={a['nonadditive_orth_ratio_mean']:.4f} "
            f"centered-order-cos={a['centered_order_cosine_mean']:.4f}"
        )
        L.append(
            f"    V134 anti={b['anti_ratio_mean']:.4f} "
            f"nonadd={b['nonadditive_orth_ratio_mean']:.4f} "
            f"centered-order-cos={b['centered_order_cosine_mean']:.4f}"
        )
        L.append(
            f"    V135-V134 anti delta={c['v135_minus_v134_anti_ratio']:+.4f} "
            f"exact-p={c['anti_exact_signflip_p']:.5f}"
        )
        L.append(
            f"    V135-V134 nonadd delta="
            f"{c['v135_minus_v134_nonadditive_ratio']:+.4f} "
            f"exact-p={c['nonadditive_exact_signflip_p']:.5f}"
        )

    L.append("")
    L.append("B) CROSS-WRAPPER INVARIANCE")
    for ch in CHANNELS:
        aa = wrapper[ch]["antisymmetric_order_component"]
        ss = wrapper[ch]["nonadditive_symmetric_residual"]

        L.append(f"  [{ch}]")
        if aa:
            L.append(
                f"    anti same-pair cosine={aa['mean_same_pair_cosine']:.4f} "
                f"perm-p={aa['pair_label_permutation_p']:.5f} "
                f"rank-rho={aa['mean_within_system_pair_rank_spearman']:.4f}"
            )
        if ss:
            L.append(
                f"    residual same-pair cosine={ss['mean_same_pair_cosine']:.4f} "
                f"perm-p={ss['pair_label_permutation_p']:.5f} "
                f"rank-rho={ss['mean_within_system_pair_rank_spearman']:.4f}"
            )

    L.append("")
    L.append("C) HIGHER-ORDER / PAIRWISE CLOSURE")
    for style in STYLES:
        for ch in CHANNELS:
            x = closure[style][ch]
            if x["n"]:
                L.append(
                    f"  {style:12s} {ch:6s}: "
                    f"all4~first={x['all4_vs_firstorder_cos_mean']:.4f} "
                    f"all4~pairclosure={x['all4_vs_pairclosure_cos_mean']:.4f} "
                    f"closure-resid={x['pairclosure_residual_ratio_mean']:.4f}"
                )

    L.append("")
    L.append("D) INTERFERENCE LOCALIZATION ON TRUE ROLES")
    for style in STYLES:
        for ch in ("v135", "v134"):
            x = concentration[style][ch]
            L.append(
                f"  {style:12s} {ch}: "
                f"true-false anti={x['true_minus_false_abs_anti']} "
                f"p={x['anti_signflip_p']} "
                f"true-false residual={x['true_minus_false_abs_residual']} "
                f"p={x['residual_signflip_p']}"
            )

    L.append("")
    L.append("E) ONE-CALL COMPRESSED ROLE EXTRACTION")
    for ch in ("v135", "v134"):
        L.append(f"  [{ch}]")
        for name, x in comp[ch].items():
            if not x:
                continue
            L.append(
                f"    {name:38s} "
                f"AUC={x['auc_within_system_z']:.4f} "
                f"AP={x['average_precision_within_system_z']:.4f} "
                f"oracleK-recall={x['oracle_K_role_recall']:.4f}"
            )

    L.append("")
    L.append("F) STRICT LOSO DEMULTIPLEXING")
    for ch in CHANNELS:
        L.append(f"  [{ch}]")
        for direction, x in decoders[ch].items():
            if not x:
                continue
            learned = x["learned_shared_linear_decoder"]
            fixed = x["fixed_hadamard_decoder"]
            L.append(
                f"    {direction:44s} "
                f"learned-top1={learned['top1_mean']:.4f} "
                f"fixed-top1={fixed['top1_mean']:.4f} "
                f"learned-margin={learned['margin_mean']:+.4f}"
            )

    L.append("")
    L.append("G) OUTPUT-SURFACE CONFOUND CHECK")
    for style in STYLES:
        x = text.get(style, {})
        if x:
            L.append(
                f"  {style}: "
                f"anti~TFIDF-distance rho="
                f"{x.get('anti_vs_tfidf_output_distance_spearman')} "
                f"p={x.get('p')} | "
                f"anti~length rho="
                f"{x.get('anti_vs_length_distance_spearman')} "
                f"p={x.get('length_p')}"
            )

    L.append("")
    L.append("H) PRIVILEGED ROUTING-TRACE VALIDATION")
    L.append(
        f"  selected query rows found="
        f"{trace.get('selected_query_rows_found', 0)}, "
        f"usable pair rows={trace.get('usable_pair_rows', 0)}"
    )
    for style in STYLES:
        x = trace.get(style, {})
        if x:
            L.append(
                f"  {style}: "
                f"anti~actual-route-change rho="
                f"{x.get('anti_vs_actual_route_change_spearman')} "
                f"p={x.get('anti_vs_actual_route_change_p')} | "
                f"nonadd~basis-route-diff rho="
                f"{x.get('nonadditive_vs_basis_trace_difference_spearman')} "
                f"p={x.get('nonadditive_vs_basis_trace_difference_p')}"
            )

    L.append("")
    L.append("I) EXECUTION-STRUCTURE FINGERPRINT (EXPLORATORY)")
    if fingerprint.get("n", 0) >= 5:
        L.append(
            f"  n={fingerprint['n']} "
            f"mode-1NN-LOSO={fingerprint['mode_1nn_loso_accuracy']:.4f} "
            f"perm-p={fingerprint['mode_permutation_p']:.5f}"
        )
        L.append(
            f"  K-LOSO-MAE={fingerprint['K_ridge_loso_MAE']:.4f} "
            f"mean-baseline-MAE={fingerprint['K_mean_baseline_MAE']:.4f}"
        )
    else:
        L.append(f"  insufficient metadata: n={fingerprint.get('n', 0)}")

    L.append("")
    L.append("=" * 90)
    L.append("DECISION LOGIC")
    L.append("=" * 90)
    L.append(
        "Strongest evidence for real structural interference would be: "
        "(1) V135 anti/nonadditivity > V134 under paired system-level tests; "
        "(2) same signed interference survives wrapper change; "
        "(3) it localizes to true roles or correlates with privileged route changes; "
        "(4) cross-wrapper LOSO decoding remains > 0.25; and "
        "(5) the effect is not explained mainly by output length/text distance."
    )
    L.append(
        "If only numbered decoding survives while cross-wrapper anti/residual alignment is null, "
        "treat V0 as a wrapper-specific prompt-composition artifact rather than a new MAS mechanism."
    )
    L.append("=" * 90)

    report = "\n".join(L)
    (outdir / "REPORT.txt").write_text(report + "\n", encoding="utf-8")

    print()
    print(report)
    print()
    print(f"[done] {outdir / 'REPORT.txt'}")
    print(f"[done] {outdir / 'summary.json'}")
    print(f"[done] {outdir / 'pair_metrics.jsonl'}")


if __name__ == "__main__":
    main()
