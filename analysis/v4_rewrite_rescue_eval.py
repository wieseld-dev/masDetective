#!/usr/bin/env python3

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("analysis").resolve()))

import observation_model_v1 as om
import visible_adaptive_bridge_v1 as vb

from targets.single_router_zoo_v11 import validate_v11_design
validate_v11_design()

from targets.single_router_fanout_mas import (
    SingleRouterFanoutMAS,
    SingleRouterSystemSpec,
)
from v11.role_universe import TRAIN_ROLES
from v3_common import (
    k_probs,
    load_matrix_npz,
    normalize_weights,
    role_probs,
)

ROOT = Path("artifacts/v3_coverage_elimination_pilot")
RESCUE = Path("artifacts/v4_rewrite_rescue")
SOURCES = RESCUE / "sources.jsonl"
REWRITES = RESCUE / "rewrites.jsonl"

OUTDIR = RESCUE / "evaluation"
OUTDIR.mkdir(parents=True, exist_ok=True)

EXEC_OUT = OUTDIR / "executions.jsonl"
METRICS_OUT = OUTDIR / "candidate_metrics.jsonl"
SUMMARY_OUT = OUTDIR / "summary.json"

OBS = Path("artifacts/observation_model_v1")

EPS = 1e-9


def read_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def append_jsonl(p, x):
    with Path(p).open("a", encoding="utf-8") as f:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")


def membership_score(p, truth_mask):
    p = np.clip(
        np.asarray(p, dtype=float),
        EPS,
        1.0 - EPS,
    )
    t = np.asarray(truth_mask, dtype=bool)

    pos = np.mean(np.log(p[t]))
    neg = np.mean(np.log(1.0 - p[~t]))

    return float(0.5 * pos + 0.5 * neg)


def margin(p, truth_mask):
    p = np.asarray(p, dtype=float)
    t = np.asarray(truth_mask, dtype=bool)

    return float(
        np.min(p[t]) - np.max(p[~t])
    )


def max_false(p, truth_mask):
    p = np.asarray(p, dtype=float)
    t = np.asarray(truth_mask, dtype=bool)

    return float(np.max(p[~t]))


def exec_key(x):
    return (
        int(x["rescue_id"]),
        str(x["kind"]),
        int(x.get("rewrite_index", 0)),
    )


def build_execution_plan():
    sources = {
        int(x["rescue_id"]): x
        for x in read_jsonl(SOURCES)
    }

    rewrite_rows = {
        int(x["rescue_id"]): x
        for x in read_jsonl(REWRITES)
        if "error" not in x
    }

    assert len(sources) == 20, len(sources)

    plan = []

    for rid, src in sorted(sources.items()):

        # META control: rerun exact original query.
        plan.append({
            "rescue_id": rid,
            "round": int(src["round"]),
            "session_id": str(src["session_id"]),
            "kind": "original",
            "rewrite_index": 0,
            "query": str(src["query"]),
            "historical_info_reward":
                float(src["info_reward"]),
        })

        rr = rewrite_rows.get(rid)

        if rr is None:
            continue

        # Only Frozen-Judge NON_META rewrites.
        for z in rr["rewrites"]:
            if z["label"] != "NON_META":
                continue

            plan.append({
                "rescue_id": rid,
                "round": int(src["round"]),
                "session_id": str(src["session_id"]),
                "kind": "rewrite",
                "rewrite_index":
                    int(z["rewrite_index"]),
                "query": str(z["query"]),
                "historical_info_reward":
                    float(src["info_reward"]),
            })

    n_orig = sum(x["kind"] == "original" for x in plan)
    n_rw = sum(x["kind"] == "rewrite" for x in plan)

    print("execution plan:", len(plan))
    print("original controls:", n_orig)
    print("clean rewrites:", n_rw)

    assert n_orig == 20
    assert n_rw == 44
    assert len(plan) == 64

    return plan


def execute_one(row, manifest_by_id, model, retries):
    sid = row["session_id"]
    m = manifest_by_id[sid]

    spec = SingleRouterSystemSpec(
        system_id=sid,
        split=f"adaptive_v11_{m['split']}",
        execution_mode=str(m["mode"]),
        style=str(m["style"]),
        worker_roles=tuple(map(str, m["roles"])),
    )

    last = None

    for attempt in range(retries + 1):
        try:
            target = SingleRouterFanoutMAS(
                spec,
                model=model,
            )

            output = str(
                target.query(row["query"])
            )

            tr = getattr(
                target,
                "_last_trace",
                None,
            )

            selected = (
                list(
                    map(
                        str,
                        getattr(
                            tr,
                            "selected_roles",
                            (),
                        ),
                    )
                )
                if tr is not None
                else []
            )

            return {
                **row,
                "final_output": output,
                "execution_status": "ok",
                "execution_error": "",
                "retries_used": attempt,
                "selected_roles_eval_only":
                    selected,
            }

        except Exception as e:
            last = e

            if attempt >= retries:
                break

            time.sleep(
                min(20.0, 2.0 ** attempt)
            )

    return {
        **row,
        "final_output": "",
        "execution_status": "error",
        "execution_error": repr(last),
        "retries_used": retries,
        "selected_roles_eval_only": [],
    }


def execute(args, plan):
    manifest = json.loads(
        (ROOT / "manifest_privileged.json")
        .read_text(encoding="utf-8")
    )

    manifest_by_id = {
        str(x["session_id"]): x
        for x in manifest["systems"]
        if x["split"] == "train"
    }

    existing = {
        exec_key(x): x
        for x in read_jsonl(EXEC_OUT)
    }

    pending = [
        x for x in plan
        if exec_key(x) not in existing
    ]

    print()
    print("already executed:", len(existing))
    print("pending:", len(pending))

    if not pending:
        return

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as ex:

        futs = {
            ex.submit(
                execute_one,
                x,
                manifest_by_id,
                args.model,
                args.retries,
            ): x
            for x in pending
        }

        done = 0

        for fut in as_completed(futs):
            rec = fut.result()
            append_jsonl(EXEC_OUT, rec)

            done += 1

            if (
                done % 5 == 0
                or done == len(pending)
            ):
                print(
                    f"executed "
                    f"{done}/{len(pending)}",
                    flush=True,
                )


def evaluate(args):
    rows = [
        x for x in read_jsonl(EXEC_OUT)
        if x.get("execution_status") == "ok"
    ]

    print()
    print("successful executions:", len(rows))

    if len(rows) != 64:
        bad = [
            x for x in read_jsonl(EXEC_OUT)
            if x.get("execution_status") != "ok"
        ]

        print("failed:", len(bad))

        for x in bad[:10]:
            print(
                x["rescue_id"],
                x["kind"],
                x.get("rewrite_index"),
                x.get("execution_error"),
            )

        raise SystemExit(
            "Not all 64 executions succeeded; "
            "rerun script to resume."
        )

    roles = list(TRAIN_ROLES)

    Z = np.load(
        ROOT / "hypotheses.npy"
    ).astype(np.float32)

    Klabels = np.load(
        ROOT / "klabels.npy"
    )

    asset = joblib.load(
        ROOT / "visible_bridge.joblib"
    )

    models = asset["models"]
    cals = asset["cals"]

    thresholds = np.asarray(
        asset["thresholds"],
        dtype=np.float64,
    )

    tpr = np.asarray(
        asset["tpr"],
        dtype=np.float64,
    )

    fpr = np.asarray(
        asset["fpr"],
        dtype=np.float64,
    )

    log_posw = np.asarray(
        asset["log_posweight"],
        dtype=np.float64,
    )

    Tcal = float(asset["temperature"])
    beta = float(asset["beta"])

    ckpt = torch.load(
        OBS / "model.pt",
        map_location="cpu",
    )

    fwd = om.ObservationNet(
        qdim=ckpt["query_embedding_dim"],
        n_roles=len(roles),
        hidden=ckpt["hidden"],
        dropout=ckpt["dropout"],
    )

    fwd.load_state_dict(
        ckpt["state_dict"]
    )

    fwd.eval()

    print("loading MiniLM...", flush=True)

    embedder = SentenceTransformer(
        args.embedding_model
    )

    print("embedding 64 q/output pairs...", flush=True)

    Q = np.asarray(
        embedder.encode(
            [x["query"] for x in rows],
            convert_to_numpy=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )

    O = np.asarray(
        embedder.encode(
            [x["final_output"] for x in rows],
            convert_to_numpy=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )

    pvisible = vb.visible_probs(
        models,
        cals,
        np.concatenate([Q, O], axis=1),
    )

    manifest = json.loads(
        (ROOT / "manifest_privileged.json")
        .read_text(encoding="utf-8")
    )

    truth = {
        str(x["session_id"]):
            tuple(map(str, x["roles"]))
        for x in manifest["systems"]
        if x["split"] == "train"
    }

    # Cache pre-round posterior matrices.
    post_cache = {}

    def get_posterior(rnd):
        if rnd not in post_cache:
            ids, P = load_matrix_npz(
                ROOT
                / f"posterior_train_round_{rnd-1:02d}.npz",
                "weights",
            )

            post_cache[rnd] = (
                {sid: i for i, sid in enumerate(ids)},
                P,
            )

        return post_cache[rnd]

    metrics = []

    for n, row in enumerate(rows):
        rnd = int(row["round"])
        sid = row["session_id"]

        row_of, P = get_posterior(rnd)

        W = normalize_weights(
            P[row_of[sid]]
        )

        rp0_dict = role_probs(
            W,
            Z,
            roles,
        )

        rp0 = np.asarray(
            [rp0_dict[r] for r in roles],
            dtype=np.float64,
        )

        kp0 = k_probs(
            W,
            Klabels,
        )

        truth_roles = set(
            truth[sid]
        )

        tm = np.asarray(
            [r in truth_roles for r in roles],
            dtype=bool,
        )

        belief = rp0.astype(
            np.float32
        )

        kvec = np.asarray(
            [
                kp0[k]
                for k in (3, 4, 5, 6)
            ],
            dtype=np.float32,
        )

        logits = vb.particle_logits(
            fwd,
            Q[n:n+1],
            belief,
            kvec,
            float(min(rnd, 8) / 8.0),
            Z,
            "cpu",
            4096,
        )

        pY = np.squeeze(
            np.asarray(
                vb.calibrated_hidden_probs(
                    logits,
                    log_posw,
                    Tcal,
                ),
                dtype=np.float64,
            )
        )

        pE = np.squeeze(
            np.asarray(
                vb.detector_channel_probs(
                    pY,
                    tpr,
                    fpr,
                ),
                dtype=np.float64,
            )
        )

        evidence = np.asarray(
            pvisible[n] >= thresholds,
            dtype=np.float64,
        ).reshape(-1)

        Wnew = normalize_weights(
            vb.update_weights(
                pE,
                evidence,
                W,
                beta,
            )
        )

        rp1_dict = role_probs(
            Wnew,
            Z,
            roles,
        )

        rp1 = np.asarray(
            [rp1_dict[r] for r in roles],
            dtype=np.float64,
        )

        sz0 = membership_score(
            rp0,
            tm,
        )

        sz1 = membership_score(
            rp1,
            tm,
        )

        m0 = margin(
            rp0,
            tm,
        )

        m1 = margin(
            rp1,
            tm,
        )

        mf0 = max_false(
            rp0,
            tm,
        )

        mf1 = max_false(
            rp1,
            tm,
        )

        dsz = float(sz1 - sz0)
        dm = float(m1 - m0)
        maxf_drop = float(mf0 - mf1)

        metrics.append({
            **row,
            "delta_sz": dsz,
            "delta_margin": dm,
            "maxfalse_drop": maxf_drop,
            "info_reward": dsz + dm,
            "visible_evidence_cardinality":
                int(evidence.sum()),
        })

        if (n + 1) % 10 == 0:
            print(
                f"evaluated {n+1}/{len(rows)}",
                flush=True,
            )

    with METRICS_OUT.open(
        "w",
        encoding="utf-8",
    ) as f:
        for x in metrics:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )

    summarize(metrics)


def summarize(rows):
    groups = defaultdict(list)

    for x in rows:
        groups[int(x["rescue_id"])].append(x)

    source_rows = []

    for rid, xs in sorted(groups.items()):
        orig = [
            x for x in xs
            if x["kind"] == "original"
        ]

        rw = [
            x for x in xs
            if x["kind"] == "rewrite"
        ]

        assert len(orig) == 1

        o = orig[0]

        if rw:
            best = max(
                rw,
                key=lambda z:
                    z["info_reward"],
            )

            mean_rw = float(
                np.mean(
                    [
                        x["info_reward"]
                        for x in rw
                    ]
                )
            )

            mean_dm = float(
                np.mean(
                    [
                        x["delta_margin"]
                        for x in rw
                    ]
                )
            )

            mean_mf = float(
                np.mean(
                    [
                        x["maxfalse_drop"]
                        for x in rw
                    ]
                )
            )
        else:
            best = None
            mean_rw = None
            mean_dm = None
            mean_mf = None

        source_rows.append({
            "rescue_id": rid,
            "n_rewrites": len(rw),
            "historical_original":
                o["historical_info_reward"],
            "replayed_original":
                o["info_reward"],
            "original_delta_sz":
                o["delta_sz"],
            "original_delta_margin":
                o["delta_margin"],
            "original_maxfalse_drop":
                o["maxfalse_drop"],
            "mean_rewrite":
                mean_rw,
            "mean_rewrite_delta_margin":
                mean_dm,
            "mean_rewrite_maxfalse_drop":
                mean_mf,
            "best_rewrite":
                (
                    best["info_reward"]
                    if best else None
                ),
            "best_rewrite_delta_sz":
                (
                    best["delta_sz"]
                    if best else None
                ),
            "best_rewrite_delta_margin":
                (
                    best["delta_margin"]
                    if best else None
                ),
            "best_rewrite_maxfalse_drop":
                (
                    best["maxfalse_drop"]
                    if best else None
                ),
            "best_rewrite_query":
                (
                    best["query"]
                    if best else None
                ),
        })

    rescued = [
        x for x in source_rows
        if x["n_rewrites"] > 0
    ]

    all_rw = [
        x for x in rows
        if x["kind"] == "rewrite"
    ]

    orig_same = np.asarray(
        [
            x["replayed_original"]
            for x in rescued
        ],
        dtype=float,
    )

    mean_rw = np.asarray(
        [
            x["mean_rewrite"]
            for x in rescued
        ],
        dtype=float,
    )

    best_rw = np.asarray(
        [
            x["best_rewrite"]
            for x in rescued
        ],
        dtype=float,
    )

    hist = np.asarray(
        [
            x["historical_original"]
            for x in source_rows
        ],
        dtype=float,
    )

    replay = np.asarray(
        [
            x["replayed_original"]
            for x in source_rows
        ],
        dtype=float,
    )

    denom = float(
        np.mean(orig_same)
    )

    mean_retention = (
        float(np.mean(mean_rw)) / denom
        if abs(denom) > 1e-12
        else float("nan")
    )

    best_retention = (
        float(np.mean(best_rw)) / denom
        if abs(denom) > 1e-12
        else float("nan")
    )

    summary = {
        "sources_total":
            len(source_rows),
        "sources_with_clean_rewrite":
            len(rescued),
        "clean_rewrites":
            len(all_rw),

        "historical_original_mean":
            float(np.mean(hist)),
        "replayed_original_mean":
            float(np.mean(replay)),

        "replay_vs_historical_corr":
            (
                float(
                    np.corrcoef(
                        hist,
                        replay,
                    )[0, 1]
                )
                if len(hist) > 1
                else None
            ),

        "same_sources_original_mean":
            float(np.mean(orig_same)),

        "all_clean_rewrite_mean":
            float(
                np.mean(
                    [
                        x["info_reward"]
                        for x in all_rw
                    ]
                )
            ),

        "per_source_mean_rewrite":
            float(np.mean(mean_rw)),

        "per_source_best_rewrite":
            float(np.mean(best_rw)),

        "mean_rewrite_retention":
            mean_retention,

        "best_rewrite_retention":
            best_retention,

        "rewrite_positive_fraction":
            float(
                np.mean(
                    [
                        x["info_reward"] > 0
                        for x in all_rw
                    ]
                )
            ),

        "rewrite_margin_positive_fraction":
            float(
                np.mean(
                    [
                        x["delta_margin"] > 0
                        for x in all_rw
                    ]
                )
            ),

        "rewrite_maxfalse_down_fraction":
            float(
                np.mean(
                    [
                        x["maxfalse_drop"] > 0
                        for x in all_rw
                    ]
                )
            ),

        "sources_any_positive_rewrite":
            float(
                np.mean(
                    [
                        x["best_rewrite"] > 0
                        for x in rescued
                    ]
                )
            ),

        "sources_any_positive_margin":
            float(
                np.mean(
                    [
                        x[
                            "best_rewrite_delta_margin"
                        ] > 0
                        for x in rescued
                    ]
                )
            ),
    }

    with SUMMARY_OUT.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "summary": summary,
                "per_source": source_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=== REWRITE RESCUE RESULTS ===")
    print(
        f"sources with rewrites    : "
        f"{len(rescued)}/20"
    )
    print(
        f"clean rewrites evaluated : "
        f"{len(all_rw)}"
    )

    print()
    print("=== STOCHASTIC CONTROL ===")
    print(
        f"historical META mean     : "
        f"{summary['historical_original_mean']:+.5f}"
    )
    print(
        f"replayed META mean       : "
        f"{summary['replayed_original_mean']:+.5f}"
    )
    print(
        f"historical/replay corr   : "
        f"{summary['replay_vs_historical_corr']:+.3f}"
    )

    print()
    print("=== CLEAN REWRITE SIGNAL ===")
    print(
        f"original mean (same 19)  : "
        f"{summary['same_sources_original_mean']:+.5f}"
    )
    print(
        f"all rewrite mean         : "
        f"{summary['all_clean_rewrite_mean']:+.5f}"
    )
    print(
        f"per-source mean rewrite  : "
        f"{summary['per_source_mean_rewrite']:+.5f}"
    )
    print(
        f"per-source BEST rewrite  : "
        f"{summary['per_source_best_rewrite']:+.5f}"
    )
    print(
        f"mean retention           : "
        f"{summary['mean_rewrite_retention']:.3f}"
    )
    print(
        f"best retention           : "
        f"{summary['best_rewrite_retention']:.3f}"
    )

    print()
    print("=== ELIMINATION ===")
    print(
        f"rewrite reward >0        : "
        f"{summary['rewrite_positive_fraction']:.3f}"
    )
    print(
        f"rewrite dMargin >0       : "
        f"{summary['rewrite_margin_positive_fraction']:.3f}"
    )
    print(
        f"rewrite maxFalse drop >0 : "
        f"{summary['rewrite_maxfalse_down_fraction']:.3f}"
    )
    print(
        f"sources any +reward      : "
        f"{summary['sources_any_positive_rewrite']:.3f}"
    )
    print(
        f"sources any +margin      : "
        f"{summary['sources_any_positive_margin']:.3f}"
    )

    print()
    print("wrote:", SUMMARY_OUT)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        default="gpt-4o-mini",
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=6,
    )

    ap.add_argument(
        "--retries",
        type=int,
        default=2,
    )

    ap.add_argument(
        "--embedding-model",
        default=
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set"
        )

    plan = build_execution_plan()

    execute(
        args,
        plan,
    )

    evaluate(args)


if __name__ == "__main__":
    main()
