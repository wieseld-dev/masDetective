from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np

from adaptive_v11_active.belief import FrozenBeliefBundle, k_features
from adaptive_v11_active.io import read_jsonl
from adaptive_v11_active.schema import PublicBeliefState
from v11.role_universe import TRAIN_ROLES

EPS = 1e-9
KS = (3, 4, 5, 6)


def role_only_probs(bundle: FrozenBeliefBundle, history):
    if not history:
        _, rp = bundle.initial()
        return dict(rp)
    raw = bundle._raw_role_vector_with(bundle.role_model, history)
    pcal = bundle._apply_platt(bundle.role_platt, raw)
    return {r: float(pcal[i]) for i, r in enumerate(bundle.roles)}


def aligned_one(artifact, x):
    model = artifact["model"]
    raw = model.predict_proba(np.asarray(x).reshape(1, -1))[0]
    classes = [int(v) for v in model.named_steps["clf"].classes_]
    p = np.full(len(KS), EPS, dtype=float)
    where = {k: i for i, k in enumerate(KS)}
    for j, k in enumerate(classes):
        if k in where:
            p[where[k]] = max(EPS, float(raw[j]))
    p /= p.sum()
    t = max(0.05, float(artifact["temperature"]))
    z = np.log(np.clip(p, EPS, 1.0)) / t
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()


def row_metrics(p, true_k):
    yi = KS.index(int(true_k))
    y = np.zeros(len(KS)); y[yi] = 1.0
    khat = int(KS[int(np.argmax(p))])
    return {
        "nll": float(-math.log(max(float(p[yi]), EPS))),
        "brier": float(np.sum((p-y)**2)),
        "acc": float(khat == true_k),
        "mae": float(abs(khat-true_k)),
        "khat": khat,
    }


def boot_ci(values, seed=917999, n=20000):
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = np.empty(n, dtype=float)
    for i in range(n):
        means[i] = a[rng.integers(0, len(a), size=len(a))].mean()
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def main():
    ap = argparse.ArgumentParser(description="Evaluate frozen K head on untouched DEV histories")
    ap.add_argument("--root", required=True)
    ap.add_argument("--belief", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--k-head", required=True)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=918001)
    args = ap.parse_args()

    root = Path(args.root)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    truth = {
        r["session_id"]: int(r["k"])
        for r in manifest["systems"] if r["split"] == "dev"
    }
    if len(truth) != 32:
        print(f"[warning] expected 32 DEV systems, found {len(truth)}", flush=True)

    bundle = FrozenBeliefBundle.load(args.belief)
    art = joblib.load(args.k_head)
    if tuple(map(int, art["ks"])) != KS:
        raise RuntimeError(f"unexpected K classes in head: {art['ks']}")
    if art.get("train_manifest_sha256") != manifest.get("sha256"):
        # Same manifest contains both disjoint TRAIN and DEV. This check confirms the
        # head was fit from the TRAIN side of this exact locked experiment manifest.
        raise RuntimeError("K head was not fit from this locked manifest")

    all_rounds = {}
    per_system_final = []
    for rnd in range(1, args.rounds + 1):
        states = {
            r["session_id"]: PublicBeliefState.from_public_dict(r)
            for r in read_jsonl(root / f"states_dev_round_{rnd:02d}.jsonl")
        }
        rows = []
        for sid, true_k in sorted(truth.items()):
            st = states[sid]
            h = list(st.history)
            rp = role_only_probs(bundle, h)
            x = k_features(h, rp, TRAIN_ROLES)
            p = aligned_one(art, x)
            m = row_metrics(p, true_k)
            rec = {
                "session_id": sid,
                "round": rnd,
                "true_k": true_k,
                "history_len": len(h),
                "probs": {str(k): float(p[i]) for i, k in enumerate(KS)},
                **m,
            }
            rows.append(rec)
            if rnd == args.rounds:
                per_system_final.append(rec)

        all_rounds[str(rnd)] = {
            "systems": len(rows),
            "mean_history_len": float(np.mean([r["history_len"] for r in rows])),
            "k_nll": float(np.mean([r["nll"] for r in rows])),
            "k_brier": float(np.mean([r["brier"] for r in rows])),
            "k_accuracy": float(np.mean([r["acc"] for r in rows])),
            "k_mae": float(np.mean([r["mae"] for r in rows])),
        }

    final = all_rounds[str(args.rounds)]
    uniform_nll = math.log(4.0)
    uniform_brier = 0.75
    best_constant_mae = 1.0  # balanced K=3,4,5,6; constant 4 or 5

    dnll = [r["nll"] - uniform_nll for r in per_system_final]
    dbrier = [r["brier"] - uniform_brier for r in per_system_final]
    nll_ci = boot_ci(dnll, args.seed, args.bootstrap)
    brier_ci = boot_ci(dbrier, args.seed + 1, args.bootstrap)

    point_go = bool(final["k_nll"] < uniform_nll and final["k_brier"] < uniform_brier)
    strong_go = bool(nll_ci[1] < 0.0 and brier_ci[1] < 0.0)

    report = {
        "version": "av11-k-bootstrap-eval-v1",
        "protocol": {
            "train": "96 fresh systems, role-adaptive acquisition with K fixed uniform, K head fit on TRAIN only",
            "dev": "32 fresh disjoint systems, same frozen acquisition policy; DEV untouched until K head frozen",
            "queries_per_system": args.rounds,
            "primary_endpoint": f"round-{args.rounds} K NLL and multiclass Brier",
        },
        "rounds": all_rounds,
        "final_round": {
            **final,
            "uniform_k_nll": uniform_nll,
            "uniform_k_brier": uniform_brier,
            "chance_accuracy": 0.25,
            "best_constant_mae_balanced_k": best_constant_mae,
            "delta_nll_vs_uniform": float(final["k_nll"] - uniform_nll),
            "delta_brier_vs_uniform": float(final["k_brier"] - uniform_brier),
            "delta_nll_bootstrap_95ci": nll_ci,
            "delta_brier_bootstrap_95ci": brier_ci,
            "point_go": point_go,
            "strong_go": strong_go,
        },
        "decision": (
            "STRONG_GO_TO_FRESH_TEST_FACTORIAL" if strong_go
            else "POINT_GO_BUT_CI_CROSSES_ZERO" if point_go
            else "NO_GO"
        ),
        "note": "Primary go/no-go uses proper scores. Accuracy/MAE are secondary because uniform posterior has no unique MAP class.",
    }

    (root / "k_head_dev_predictions.json").write_text(
        json.dumps(per_system_final, indent=2, sort_keys=True), encoding="utf-8"
    )
    (root / "k_head_dev_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
