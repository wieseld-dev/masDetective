#!/usr/bin/env python3
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("analysis").resolve()))

import observation_model_v1 as om
import visible_adaptive_bridge_v1 as vb

from v11.role_universe import TRAIN_ROLES
from v3_common import (
    k_probs,
    load_matrix_npz,
    normalize_weights,
    role_probs,
)

ROOT = Path("artifacts/v3_coverage_elimination_pilot")
MANIFEST = ROOT / "manifest_privileged.json"
OBS = Path("artifacts/observation_model_v1")
ROUNDS = range(1, 7)
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0]
EPS = 1e-9


def read_jsonl(p):
    with Path(p).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def membership_score(p, truth_mask):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    t = np.asarray(truth_mask, dtype=bool)

    pos = np.mean(np.log(p[t]))
    neg = np.mean(np.log(1.0 - p[~t]))

    # Balanced presence/absence objective:
    # false roles do not dominate just because there are more of them.
    return float(0.5 * pos + 0.5 * neg)


def margin(p, truth_mask):
    p = np.asarray(p, dtype=float)
    t = np.asarray(truth_mask, dtype=bool)
    return float(np.min(p[t]) - np.max(p[~t]))


def max_false(p, truth_mask):
    p = np.asarray(p, dtype=float)
    t = np.asarray(truth_mask, dtype=bool)
    return float(np.max(p[~t]))


def mean_true(p, truth_mask):
    p = np.asarray(p, dtype=float)
    return float(np.mean(p[np.asarray(truth_mask, dtype=bool)]))


def mean_false(p, truth_mask):
    p = np.asarray(p, dtype=float)
    return float(np.mean(p[~np.asarray(truth_mask, dtype=bool)]))


def gap_prediction(p, roles):
    ranked = sorted(
        zip(roles, p),
        key=lambda x: (-float(x[1]), x[0])
    )
    vals = [float(x[1]) for x in ranked]

    khat = max(
        (3, 4, 5, 6),
        key=lambda k: (vals[k - 1] - vals[k], -k)
    )

    return set(x[0] for x in ranked[:khat])


def set_f1(pred, truth):
    pred, truth = set(pred), set(truth)
    tp = len(pred & truth)

    if not pred or not truth:
        return 0.0

    pr = tp / len(pred)
    rc = tp / len(truth)

    return (
        2.0 * pr * rc / (pr + rc)
        if pr + rc > 0 else 0.0
    )


def choose_best(xs, score_fn):
    # deterministic candidate-index tie break
    return max(
        xs,
        key=lambda x: (
            float(score_fn(x)),
            -int(x["candidate_index"])
        )
    )


roles = list(TRAIN_ROLES)
Z = np.load(ROOT / "hypotheses.npy").astype(np.float32)

manifest = json.loads(MANIFEST.read_text())
truth = {
    x["session_id"]: tuple(map(str, x["roles"]))
    for x in manifest["systems"]
    if x["split"] == "train"
}

asset = joblib.load(ROOT / "visible_bridge.joblib")

models = asset["models"]
cals = asset["cals"]
thresholds = np.asarray(asset["thresholds"], dtype=np.float64)
tpr = np.asarray(asset["tpr"], dtype=np.float64)
fpr = np.asarray(asset["fpr"], dtype=np.float64)
log_posw = np.asarray(asset["log_posweight"], dtype=np.float64)
Tcal = float(asset["temperature"])
beta = float(asset["beta"])

ckpt = torch.load(OBS / "model.pt", map_location="cpu")

fwd = om.ObservationNet(
    qdim=ckpt["query_embedding_dim"],
    n_roles=len(roles),
    hidden=ckpt["hidden"],
    dropout=ckpt["dropout"],
)
fwd.load_state_dict(ckpt["state_dict"])
fwd.eval()

embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

all_rows = []

for rnd in ROUNDS:
    print(f"[replay] round {rnd}", flush=True)

    ids, P = load_matrix_npz(
        ROOT / f"posterior_train_round_{rnd-1:02d}.npz",
        "weights",
    )
    row_of = {sid: i for i, sid in enumerate(ids)}

    candidates = read_jsonl(
        ROOT / f"candidates_train_round_{rnd:02d}.jsonl"
    )
    responses = read_jsonl(
        ROOT / f"responses_train_round_{rnd:02d}.jsonl"
    )
    old_rows = read_jsonl(
        ROOT / f"train_rows_round_{rnd:02d}.jsonl"
    )

    response_by = {
        (x["session_id"], int(x["candidate_index"])): x
        for x in responses
    }

    old_by = {
        (x["session_id"], int(x["candidate_index"])): x
        for x in old_rows
    }

    valid = []

    for c in candidates:
        key = (c["session_id"], int(c["candidate_index"]))
        rr = response_by.get(key)

        if (
            c.get("valid")
            and rr is not None
            and rr.get("execution_status") == "ok"
            and key in old_by
        ):
            valid.append((c, rr))

    Q = np.asarray(
        embedder.encode(
            [c["query"] for c, _ in valid],
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )

    O = np.asarray(
        embedder.encode(
            [rr.get("final_output", "") for _, rr in valid],
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )

    pvisible = vb.visible_probs(
        models,
        cals,
        np.concatenate([Q, O], axis=1),
    )

    for n, (c, rr) in enumerate(valid):
        sid = c["session_id"]
        idx = int(c["candidate_index"])

        W = normalize_weights(P[row_of[sid]])

        kp0 = k_probs(
            W,
            np.load(ROOT / "klabels.npy"),
        )
        rp0_dict = role_probs(W, Z, roles)
        rp0 = np.asarray(
            [rp0_dict[r] for r in roles],
            dtype=np.float64,
        )

        t_roles = set(truth[sid])
        tm = np.asarray(
            [r in t_roles for r in roles],
            dtype=bool,
        )

        belief = rp0.astype(np.float32)
        kvec = np.asarray(
            [kp0[k] for k in (3, 4, 5, 6)],
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

        pY = np.squeeze(np.asarray(
            vb.calibrated_hidden_probs(
                logits,
                log_posw,
                Tcal,
            ),
            dtype=np.float64,
        ))

        if pY.shape != Z.shape:
            raise RuntimeError(
                f"pY shape {pY.shape}, expected {Z.shape}"
            )

        pE = np.squeeze(np.asarray(
            vb.detector_channel_probs(
                pY,
                tpr,
                fpr,
            ),
            dtype=np.float64,
        ))

        evidence = np.asarray(
            pvisible[n] >= thresholds,
            dtype=np.float64,
        ).reshape(-1)

        W1 = normalize_weights(
            vb.update_weights(
                pE,
                evidence,
                W,
                beta,
            )
        )

        rp1_dict = role_probs(W1, Z, roles)
        rp1 = np.asarray(
            [rp1_dict[r] for r in roles],
            dtype=np.float64,
        )

        sz0 = membership_score(rp0, tm)
        sz1 = membership_score(rp1, tm)

        m0 = margin(rp0, tm)
        m1 = margin(rp1, tm)

        mf0 = max_false(rp0, tm)
        mf1 = max_false(rp1, tm)

        mt0 = mean_true(rp0, tm)
        mt1 = mean_true(rp1, tm)

        favg0 = mean_false(rp0, tm)
        favg1 = mean_false(rp1, tm)

        f10 = set_f1(
            gap_prediction(rp0, roles),
            t_roles,
        )
        f11 = set_f1(
            gap_prediction(rp1, roles),
            t_roles,
        )

        old = old_by[(sid, idx)]

        all_rows.append({
            "round": rnd,
            "session_id": sid,
            "candidate_index": idx,
            "group_id": c["group_id"],
            "chosen_for_continuation":
                bool(c.get("chosen_for_continuation")),

            "old_reward":
                float(old["reward"]),

            "sz_before": sz0,
            "sz_after": sz1,
            "delta_sz": sz1 - sz0,

            "margin_before": m0,
            "margin_after": m1,
            "delta_margin": m1 - m0,

            # Positive means we successfully pushed
            # the strongest false role DOWN.
            "maxfalse_drop": mf0 - mf1,

            "mean_true_gain": mt1 - mt0,

            # Positive means average false probability fell.
            "meanfalse_drop": favg0 - favg1,

            "gap_f1_before": f10,
            "gap_f1_after": f11,
            "delta_gap_f1": f11 - f10,
        })


out = ROOT / "membership_relabel_candidates.jsonl"
with out.open("w", encoding="utf-8") as f:
    for x in all_rows:
        f.write(json.dumps(x) + "\n")


groups = defaultdict(list)
for x in all_rows:
    groups[(x["round"], x["session_id"])].append(x)


def summarize(label, selector):
    chosen = []

    for key, xs in groups.items():
        if selector == "actual":
            cc = [
                x for x in xs
                if x["chosen_for_continuation"]
            ]
            if not cc:
                continue
            z = cc[0]

        elif selector == "old":
            z = choose_best(
                xs,
                lambda x: x["old_reward"],
            )

        elif selector == "margin":
            z = choose_best(
                xs,
                lambda x: x["delta_margin"],
            )

        else:
            lam = float(selector)
            z = choose_best(
                xs,
                lambda x:
                    x["delta_sz"]
                    + lam * x["delta_margin"],
            )

        best_m = max(x["delta_margin"] for x in xs)

        y = dict(z)
        y["best_margin_hit"] = (
            abs(z["delta_margin"] - best_m) < 1e-12
        )
        chosen.append(y)

    return {
        "label": label,
        "n": len(chosen),
        "mean_delta_sz":
            float(np.mean([x["delta_sz"] for x in chosen])),
        "mean_delta_margin":
            float(np.mean([x["delta_margin"] for x in chosen])),
        "positive_margin_fraction":
            float(np.mean([x["delta_margin"] > 0 for x in chosen])),
        "mean_maxfalse_drop":
            float(np.mean([x["maxfalse_drop"] for x in chosen])),
        "maxfalse_down_fraction":
            float(np.mean([x["maxfalse_drop"] > 0 for x in chosen])),
        "mean_true_gain":
            float(np.mean([x["mean_true_gain"] for x in chosen])),
        "meanfalse_down_fraction":
            float(np.mean([x["meanfalse_drop"] > 0 for x in chosen])),
        "mean_delta_gap_f1":
            float(np.mean([x["delta_gap_f1"] for x in chosen])),
        "gap_f1_up_fraction":
            float(np.mean([x["delta_gap_f1"] > 1e-12 for x in chosen])),
        "best_margin_hit_fraction":
            float(np.mean([x["best_margin_hit"] for x in chosen])),
    }


summaries = [
    summarize("Actual/random continuation", "actual"),
    summarize("Best old V3 reward", "old"),
]

for lam in LAMBDAS:
    summaries.append(
        summarize(
            f"dSZ + {lam:g}*dMargin",
            str(lam),
        )
    )

summaries.append(
    summarize("Oracle best dMargin", "margin")
)


print()
print("=== MEMBERSHIP RELABEL: ALL 96 STATES ===")
print(
    "Method                 | dSZ     | dM      | M+%  | "
    "dMaxF  | MaxFdown% | dF1    | F1up% | BestMhit%"
)
print(
    "-----------------------+---------+---------+------+"
    "--------+-----------+--------+-------+----------"
)

for s in summaries:
    print(
        f"{s['label'][:22]:22s} | "
        f"{s['mean_delta_sz']:+.5f} | "
        f"{s['mean_delta_margin']:+.5f} | "
        f"{100*s['positive_margin_fraction']:4.1f} | "
        f"{s['mean_maxfalse_drop']:+.5f} | "
        f"{100*s['maxfalse_down_fraction']:8.1f}% | "
        f"{s['mean_delta_gap_f1']:+.4f} | "
        f"{100*s['gap_f1_up_fraction']:5.1f}% | "
        f"{100*s['best_margin_hit_fraction']:8.1f}%"
    )


# Round-wise: actual vs pure membership vs lambda=1
print()
print("=== PER ROUND ===")
print("R | Method | dSZ     | dM      | M+%  | MaxFdown%")
print("--+--------+---------+---------+------+----------")

for rnd in ROUNDS:
    rg = {
        k: v for k, v in groups.items()
        if k[0] == rnd
    }

    for name, mode in [
        ("ACT", "actual"),
        ("SZ ", "0.0"),
        ("SZM", "1.0"),
    ]:
        vals = []

        for _, xs in rg.items():
            if mode == "actual":
                cc = [
                    x for x in xs
                    if x["chosen_for_continuation"]
                ]
                if not cc:
                    continue
                z = cc[0]
            else:
                lam = float(mode)
                z = choose_best(
                    xs,
                    lambda x:
                        x["delta_sz"]
                        + lam * x["delta_margin"],
                )
            vals.append(z)

        print(
            f"{rnd} | {name}    | "
            f"{np.mean([x['delta_sz'] for x in vals]):+.5f} | "
            f"{np.mean([x['delta_margin'] for x in vals]):+.5f} | "
            f"{100*np.mean([x['delta_margin'] > 0 for x in vals]):4.1f} | "
            f"{100*np.mean([x['maxfalse_drop'] > 0 for x in vals]):8.1f}%"
        )


summary_path = ROOT / "membership_relabel_summary.json"
summary_path.write_text(
    json.dumps(
        {
            "candidates": len(all_rows),
            "groups": len(groups),
            "lambdas": LAMBDAS,
            "summaries": summaries,
            "note": (
                "Training/evaluation-only relabel. Truth is used "
                "only to construct offline labels."
            ),
        },
        indent=2,
    )
)

print()
print("wrote:", out)
print("wrote:", summary_path)
