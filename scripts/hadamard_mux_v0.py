from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
from pathlib import Path

import numpy as np


SRC_ROOT = Path("artifacts/adaptive_v11_role_only_v1")
SRC_CAND = SRC_ROOT / "candidates_train_round_01.jsonl"
SRC_SCORED = SRC_ROOT / "scored_train_round_01.jsonl"

TAGS = {
    "all": (0, 1, 2, 3),
    "13":  (0, 2),
    "24":  (1, 3),
    "12":  (0, 1),
    "34":  (2, 3),
    "14":  (0, 3),
    "23":  (1, 2),
}


def read_jsonl(path: Path):
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sid_of(row):
    return str(row.get("system_id") or row.get("session_id") or "")


def compose(parts, style):
    parts = [str(x).strip() for x in parts]

    if style == "numbered":
        body = "\n".join(
            f"{i+1}. {q}"
            for i, q in enumerate(parts)
        )
        return (
            "I have one request with several parts. "
            "Please answer every part in one coherent response and do not skip any part.\n\n"
            + body
        )

    if style == "requirements":
        body = "\n".join(
            f"- {q}"
            for q in parts
        )
        return (
            "Please help with the following combined request. "
            "Treat every item below as an equally important requirement and address each clearly.\n\n"
            + body
        )

    raise ValueError(style)


def build(root: Path):
    root.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(SRC_CAND)
    if not rows:
        raise RuntimeError(f"Missing source candidates: {SRC_CAND}")

    grouped = {}
    for r in rows:
        sid = str(r.get("session_id", ""))
        try:
            idx = int(r.get("candidate_index"))
        except Exception:
            continue

        q = str(r.get("query") or "").strip()
        if (
            sid
            and idx in (0, 1, 2, 3)
            and bool(r.get("valid"))
            and q
        ):
            grouped.setdefault(sid, {})[idx] = q

    eligible = {
        sid: qs
        for sid, qs in grouped.items()
        if all(i in qs for i in range(4))
    }

    print(f"[build] source candidate rows = {len(rows)}", flush=True)
    print(f"[build] sessions with 4/4 valid basis probes = {len(eligible)}", flush=True)

    if len(eligible) < 4:
        raise RuntimeError(
            f"Only {len(eligible)} systems have four valid basis probes; refusing API run."
        )

    out_rows = []
    design_entries = []

    styles = ("numbered", "requirements")

    for sid in sorted(eligible):
        basis = eligible[sid]
        candidate_index = 0
        seen_queries = set()

        for style in styles:
            for tag, inds in TAGS.items():
                for order_name, ordered_inds in (
                    ("fwd", inds),
                    ("rev", tuple(reversed(inds))),
                ):
                    query = compose(
                        [basis[i] for i in ordered_inds],
                        style,
                    )

                    if query in seen_queries:
                        raise RuntimeError(
                            f"Duplicate composite query in {sid}: "
                            f"{style}/{tag}/{order_name}"
                        )
                    seen_queries.add(query)

                    row = {
                        "session_id": sid,
                        "split": "train",
                        "round": 1,
                        "candidate_index": candidate_index,
                        "group_id": f"{sid}::hadamard_mux_v0::{style}",
                        "query": query,
                        "valid": True,
                        "invalid_reason": "",
                        "prompt": "",
                        "completion": "",
                        "old_logp": 0.0,
                        "ref_logp": 0.0,
                        "raw_completion": "",
                        "seed": 0,
                        "adapter": "hadamard_mux_v0",
                        "phase": "hadamard_mux_v0",
                        "mux_style": style,
                        "mux_tag": tag,
                        "mux_order": order_name,
                    }
                    out_rows.append(row)

                    design_entries.append({
                        "session_id": sid,
                        "candidate_index": candidate_index,
                        "style": style,
                        "tag": tag,
                        "order": order_name,
                        "source_indices": list(ordered_inds),
                        "query": query,
                    })

                    candidate_index += 1

    expected = len(eligible) * 28
    assert len(out_rows) == expected

    write_jsonl(root / "candidates_train_round_01.jsonl", out_rows)

    design = {
        "experiment": "hadamard_mux_v0",
        "source_candidates": str(SRC_CAND),
        "source_scored": str(SRC_SCORED),
        "eligible_sessions": sorted(eligible),
        "n_sessions": len(eligible),
        "styles": list(styles),
        "measurements_per_style": 14,
        "queries_per_system": 28,
        "total_new_queries": len(out_rows),
        "basis_queries": {
            sid: {str(i): eligible[sid][i] for i in range(4)}
            for sid in sorted(eligible)
        },
        "entries": design_entries,
        "hadamard": [
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [1, -1, -1, 1],
        ],
    }

    (root / "design.json").write_text(
        json.dumps(design, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 72, flush=True)
    print("HADAMARD MUX V0 BUILD COMPLETE", flush=True)
    print(f"sessions          : {len(eligible)}", flush=True)
    print(f"queries/system    : 28", flush=True)
    print(f"new MAS queries   : {len(out_rows)}", flush=True)
    print(f"root              : {root}", flush=True)
    print("=" * 72, flush=True)


def scalar_value(x):
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    if isinstance(x, dict):
        for k in (
            "logit", "score", "value", "z",
            "raw_logit", "evidence", "probability", "prob"
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


def mapping_to_vec(obj, roles):
    if obj is None:
        return None

    if isinstance(obj, (list, tuple)):
        if len(obj) == len(roles):
            try:
                return np.asarray(obj, dtype=float)
            except Exception:
                return None

    if not isinstance(obj, dict):
        return None

    hits = sum(1 for r in roles if r in obj)

    if hits >= max(3, int(0.7 * len(roles))):
        vals = []
        for role in roles:
            v = scalar_value(obj.get(role))
            if v is None:
                return None
            vals.append(v)
        return np.asarray(vals, dtype=float)

    for key in (
        "role_scores", "scores", "logits", "evidence",
        "role_logits", "values", "roles"
    ):
        if key in obj:
            v = mapping_to_vec(obj[key], roles)
            if v is not None:
                return v

    return None


def channel_vec(row, channel, roles):
    md = row.get("metadata") or {}

    if channel == "v135":
        candidates = [
            row.get("evidence"),
            row.get("evidence_v135"),
            md.get("evidence_v135"),
            md.get("causal_evidence"),
        ]
    elif channel == "v134":
        candidates = [
            row.get("evidence_v134"),
            md.get("evidence_v134"),
            md.get("presence_evidence"),
            row.get("presence_evidence"),
        ]
    else:
        raise ValueError(channel)

    for obj in candidates:
        v = mapping_to_vec(obj, roles)
        if v is not None:
            return v

    return None


def cosine(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def mean_vec(*xs):
    arr = [np.asarray(x, dtype=float) for x in xs]
    return np.mean(np.stack(arr), axis=0)


def ci_boot(values, seed=1337, B=10000):
    x = np.asarray(
        [v for v in values if np.isfinite(v)],
        dtype=float,
    )
    if len(x) == 0:
        return [None, None]

    rng = np.random.default_rng(seed)
    means = np.empty(B, dtype=float)

    for b in range(B):
        sample = rng.choice(x, size=len(x), replace=True)
        means[b] = np.mean(sample)

    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def summarize_style_channel(session_objs, seed=1337):
    if not session_objs:
        return {
            "n_sessions": 0,
            "identity_top1": None,
        }

    accs = []
    diag_cos = []
    raw_diag_cos = []
    margins = []
    order_cos = []
    pair_add_cos = []
    all_add_cos = []
    sim_mats = []

    for obj in session_objs:
        accs.append(obj["identity_top1"])
        diag_cos.append(obj["centered_diag_cosine"])
        raw_diag_cos.append(obj["raw_diag_cosine"])
        margins.append(obj["centered_true_vs_best_wrong_margin"])
        order_cos.append(obj["order_cosine"])
        pair_add_cos.append(obj["pair_additive_cosine"])
        all_add_cos.append(obj["all_additive_cosine"])
        sim_mats.append(obj["_sim_matrix"])

    obs = float(np.nanmean(diag_cos))

    rng = np.random.default_rng(seed)
    null = []

    for _ in range(10000):
        vals = []
        for S in sim_mats:
            perm = rng.permutation(4)
            vals.append(
                np.mean(S[np.arange(4), perm])
            )
        null.append(float(np.mean(vals)))

    null = np.asarray(null, dtype=float)
    p_perm = float(
        (1 + np.sum(null >= obs - 1e-12))
        / (1 + len(null))
    )

    def smry(vals):
        arr = np.asarray(
            [x for x in vals if np.isfinite(x)],
            dtype=float,
        )
        if len(arr) == 0:
            return {
                "mean": None,
                "median": None,
                "ci95": [None, None],
            }
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "ci95": ci_boot(arr, seed=seed),
        }

    return {
        "n_sessions": len(session_objs),
        "identity_top1": smry(accs),
        "centered_diag_cosine": smry(diag_cos),
        "raw_diag_cosine": smry(raw_diag_cos),
        "centered_true_vs_best_wrong_margin": smry(margins),
        "forward_reverse_order_cosine": smry(order_cos),
        "pair_additive_cosine": smry(pair_add_cos),
        "all4_additive_cosine": smry(all_add_cos),
        "exact_within_system_label_permutation_p": p_perm,
        "chance_identity_top1": 0.25,
    }


def analyze(root: Path):
    from v11.role_universe import TRAIN_ROLES

    roles = list(TRAIN_ROLES)

    design_path = root / "design.json"
    mux_scored_path = root / "scored_train_round_01.jsonl"

    if not design_path.exists():
        raise RuntimeError(f"Missing {design_path}")
    if not mux_scored_path.exists():
        raise RuntimeError(f"Missing {mux_scored_path}")
    if not SRC_SCORED.exists():
        raise RuntimeError(f"Missing original scores {SRC_SCORED}")

    design = json.loads(design_path.read_text(encoding="utf-8"))

    source_candidates = read_jsonl(SRC_CAND)
    original_scored = read_jsonl(SRC_SCORED)
    mux_scored = read_jsonl(mux_scored_path)

    src_query_index = {}
    for r in source_candidates:
        sid = str(r.get("session_id") or "")
        q = str(r.get("query") or "")
        try:
            idx = int(r.get("candidate_index"))
        except Exception:
            continue
        if sid and q and idx in range(4):
            src_query_index[(sid, q)] = idx

    design_by_query = {
        (e["session_id"], e["query"]): e
        for e in design["entries"]
    }

    channels = ("v135", "v134", "concat")

    original = {
        c: {} for c in channels
    }

    for r in original_scored:
        sid = sid_of(r)
        q = str(r.get("query") or "")
        idx = src_query_index.get((sid, q))

        if idx is None:
            continue

        v135 = channel_vec(r, "v135", roles)
        v134 = channel_vec(r, "v134", roles)

        vals = {
            "v135": v135,
            "v134": v134,
            "concat": (
                np.concatenate([v135, v134])
                if v135 is not None and v134 is not None
                else None
            ),
        }

        for channel, vec in vals.items():
            if vec is not None:
                original[channel].setdefault(sid, {})[idx] = vec

    mux = {
        c: {} for c in channels
    }

    matched_mux_rows = 0

    for r in mux_scored:
        sid = sid_of(r)
        q = str(r.get("query") or "")

        spec = design_by_query.get((sid, q))
        if spec is None:
            continue

        matched_mux_rows += 1

        v135 = channel_vec(r, "v135", roles)
        v134 = channel_vec(r, "v134", roles)

        vals = {
            "v135": v135,
            "v134": v134,
            "concat": (
                np.concatenate([v135, v134])
                if v135 is not None and v134 is not None
                else None
            ),
        }

        for channel, vec in vals.items():
            if vec is None:
                continue

            mux[channel].setdefault(sid, {}) \
                .setdefault(spec["style"], {}) \
                .setdefault(spec["tag"], {})[
                    spec["order"]
                ] = vec

    print(f"[analyze] original score rows = {len(original_scored)}")
    print(f"[analyze] mux score rows      = {len(mux_scored)}")
    print(f"[analyze] matched mux rows    = {matched_mux_rows}")

    all_session_records = []
    summary = {
        "experiment": "hadamard_mux_v0",
        "roles": roles,
        "design_sessions": design["n_sessions"],
        "total_new_queries": design["total_new_queries"],
        "original_scored_rows": len(original_scored),
        "mux_scored_rows": len(mux_scored),
        "matched_mux_rows": matched_mux_rows,
        "results": {},
        "preregistered_gate": {
            "required_for_both_styles_on_concat": {
                "identity_top1_mean_at_least": 0.45,
                "within_system_permutation_p_at_most": 0.05,
                "forward_reverse_order_cosine_mean_at_least": 0.70,
            }
        },
    }

    pair_indices = {
        "13": (0, 2),
        "24": (1, 3),
        "12": (0, 1),
        "34": (2, 3),
        "14": (0, 3),
        "23": (1, 2),
    }

    for style in ("numbered", "requirements"):
        summary["results"][style] = {}

        for channel in channels:
            session_objs = []

            common_sids = sorted(
                set(original[channel])
                & set(mux[channel])
            )

            for sid in common_sids:
                if not all(
                    i in original[channel][sid]
                    for i in range(4)
                ):
                    continue

                block = mux[channel][sid].get(style, {})

                complete = all(
                    tag in block
                    and "fwd" in block[tag]
                    and "rev" in block[tag]
                    for tag in TAGS
                )

                if not complete:
                    continue

                orig = [
                    original[channel][sid][i]
                    for i in range(4)
                ]

                E = {
                    tag: mean_vec(
                        block[tag]["fwd"],
                        block[tag]["rev"],
                    )
                    for tag in TAGS
                }

                m0 = E["all"]
                m1 = E["13"] - E["24"]
                m2 = E["12"] - E["34"]
                m3 = E["14"] - E["23"]

                dec = [
                    (m0 + m1 + m2 + m3) / 4.0,
                    (m0 - m1 + m2 - m3) / 4.0,
                    (m0 + m1 - m2 - m3) / 4.0,
                    (m0 - m1 - m2 + m3) / 4.0,
                ]

                O = np.stack(orig)
                D = np.stack(dec)

                raw_diag = float(np.nanmean([
                    cosine(D[i], O[i])
                    for i in range(4)
                ]))

                Oc = O - np.mean(O, axis=0, keepdims=True)
                Dc = D - np.mean(D, axis=0, keepdims=True)

                S = np.zeros((4, 4), dtype=float)
                for i in range(4):
                    for j in range(4):
                        S[i, j] = cosine(Dc[i], Oc[j])

                top1 = float(np.mean(
                    np.argmax(S, axis=1)
                    == np.arange(4)
                ))

                diag = float(np.mean(np.diag(S)))

                margins_i = []
                for i in range(4):
                    wrong = [
                        S[i, j]
                        for j in range(4)
                        if j != i
                    ]
                    margins_i.append(
                        S[i, i] - max(wrong)
                    )

                margin = float(np.mean(margins_i))

                ordcos = float(np.nanmean([
                    cosine(
                        block[tag]["fwd"],
                        block[tag]["rev"],
                    )
                    for tag in TAGS
                ]))

                padd = []
                for tag, inds in pair_indices.items():
                    expected = sum(
                        (orig[i] for i in inds),
                        np.zeros_like(orig[0]),
                    )
                    padd.append(
                        cosine(E[tag], expected)
                    )

                pair_add = float(np.nanmean(padd))

                all_expected = sum(
                    orig,
                    np.zeros_like(orig[0]),
                )
                all_add = cosine(
                    E["all"],
                    all_expected,
                )

                rec = {
                    "session_id": sid,
                    "style": style,
                    "channel": channel,
                    "identity_top1": top1,
                    "centered_diag_cosine": diag,
                    "raw_diag_cosine": raw_diag,
                    "centered_true_vs_best_wrong_margin": margin,
                    "order_cosine": ordcos,
                    "pair_additive_cosine": pair_add,
                    "all_additive_cosine": all_add,
                    "_sim_matrix": S,
                }

                session_objs.append(rec)

                all_session_records.append({
                    k: v
                    for k, v in rec.items()
                    if k != "_sim_matrix"
                })

            result = summarize_style_channel(
                session_objs,
                seed=1337
                + (0 if style == "numbered" else 100)
                + (0 if channel == "v135" else 10 if channel == "v134" else 20),
            )

            summary["results"][style][channel] = result

    concat_blocks = [
        summary["results"].get(style, {}).get("concat", {})
        for style in ("numbered", "requirements")
    ]

    strong = True
    gate_reasons = []

    for style, b in zip(("numbered", "requirements"), concat_blocks):
        try:
            ident = b["identity_top1"]["mean"]
            p = b["exact_within_system_label_permutation_p"]
            order = b["forward_reverse_order_cosine"]["mean"]

            ok = (
                ident is not None
                and ident >= 0.45
                and p <= 0.05
                and order is not None
                and order >= 0.70
            )

            gate_reasons.append({
                "style": style,
                "identity_top1": ident,
                "permutation_p": p,
                "order_cosine": order,
                "passes": bool(ok),
            })

            strong = strong and ok

        except Exception:
            strong = False
            gate_reasons.append({
                "style": style,
                "passes": False,
                "reason": "insufficient data",
            })

    summary["preregistered_gate"]["details"] = gate_reasons
    summary["preregistered_gate"]["passes"] = bool(strong)

    out_json = root / "analysis_summary.json"
    out_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path = root / "analysis_per_system.csv"
    if all_session_records:
        fields = list(all_session_records[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_session_records)

    lines = []
    lines.append("=" * 78)
    lines.append("HADAMARD SEMANTIC DEMULTIPLEXING V0")
    lines.append("=" * 78)
    lines.append(f"Design systems      : {design['n_sessions']}")
    lines.append(f"New MAS queries     : {design['total_new_queries']}")
    lines.append(f"Mux scored rows     : {len(mux_scored)}")
    lines.append(f"Matched mux rows    : {matched_mux_rows}")
    lines.append("")

    for style in ("numbered", "requirements"):
        lines.append(f"STYLE: {style}")
        for channel in channels:
            b = summary["results"][style][channel]

            if not b or b.get("n_sessions", 0) == 0:
                lines.append(f"  {channel}: NO COMPLETE SYSTEMS")
                continue

            ident = b["identity_top1"]
            diag = b["centered_diag_cosine"]
            margin = b["centered_true_vs_best_wrong_margin"]
            order = b["forward_reverse_order_cosine"]
            padd = b["pair_additive_cosine"]
            aall = b["all4_additive_cosine"]

            lines.append(
                f"  {channel}: n={b['n_sessions']} "
                f"top1={ident['mean']:.3f} "
                f"[95% {ident['ci95'][0]:.3f},{ident['ci95'][1]:.3f}] "
                f"(chance=.250)"
            )
            lines.append(
                f"      centered true cosine={diag['mean']:.3f} "
                f"margin={margin['mean']:+.3f} "
                f"perm-p={b['exact_within_system_label_permutation_p']:.5f}"
            )
            lines.append(
                f"      order cosine={order['mean']:.3f} "
                f"pair-additive={padd['mean']:.3f} "
                f"all4-additive={aall['mean']:.3f}"
            )
        lines.append("")

    lines.append("PRE-REGISTERED HEURISTIC GATE")
    for g in gate_reasons:
        lines.append(
            f"  {g['style']}: "
            f"top1={g.get('identity_top1')} "
            f"p={g.get('permutation_p')} "
            f"order={g.get('order_cosine')} "
            f"PASS={g['passes']}"
        )

    lines.append("")
    lines.append(
        "OVERALL: "
        + (
            "PASS — signal survives both composition wrappers."
            if strong
            else
            "NO PASS — inspect channel/style-specific results before extending the idea."
        )
    )
    lines.append("=" * 78)

    report = "\n".join(lines)
    (root / "analysis_report.txt").write_text(
        report + "\n",
        encoding="utf-8",
    )

    print(report, flush=True)
    print(f"\nWrote {out_json}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {root / 'analysis_report.txt'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        required=True,
        choices=("build", "analyze"),
    )
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    root = Path(args.root)

    if args.stage == "build":
        build(root)
    else:
        analyze(root)


if __name__ == "__main__":
    main()
