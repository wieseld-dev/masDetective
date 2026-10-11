from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


def model_for(C: float):
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            C=float(C), class_weight="balanced", solver="lbfgs",
            max_iter=5000,
        )),
    ])


def aligned_probs(model, X):
    raw = model.predict_proba(X)
    classes = [int(x) for x in model.named_steps["clf"].classes_]
    out = np.full((len(X), len(KS)), EPS, dtype=np.float64)
    pos = {k: i for i, k in enumerate(KS)}
    for j, k in enumerate(classes):
        if k in pos:
            out[:, pos[k]] = np.maximum(EPS, raw[:, j])
    out /= out.sum(axis=1, keepdims=True)
    return out


def temp_scale(p, t):
    z = np.log(np.clip(p, EPS, 1.0)) / max(0.05, float(t))
    z -= z.max(axis=1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(axis=1, keepdims=True)


def nll(probs, y):
    idx = np.asarray([KS.index(int(v)) for v in y], dtype=int)
    return float(np.mean(-np.log(np.clip(probs[np.arange(len(y)), idx], EPS, 1.0))))


def balanced_system_folds(system_k: dict[str, int], folds: int, seed: int):
    rng = np.random.default_rng(seed)
    byk = defaultdict(list)
    for sid, k in system_k.items():
        byk[int(k)].append(sid)
    buckets = [[] for _ in range(folds)]
    for k in KS:
        ids = list(byk[k])
        rng.shuffle(ids)
        for i, sid in enumerate(ids):
            buckets[i % folds].append(sid)
    return [set(x) for x in buckets]


def main():
    ap = argparse.ArgumentParser(description="Fit a new K head on adaptive TRAIN histories only")
    ap.add_argument("--root", required=True)
    ap.add_argument("--belief", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=917777)
    ap.add_argument("--folds", type=int, default=6)
    args = ap.parse_args()

    root = Path(args.root)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    truth = {
        r["session_id"]: int(r["k"])
        for r in manifest["systems"] if r["split"] == "train"
    }
    final_states = {
        r["session_id"]: PublicBeliefState.from_public_dict(r)
        for r in read_jsonl(root / f"states_train_round_{args.rounds:02d}.jsonl")
    }
    if set(final_states) != set(truth):
        raise RuntimeError(f"train state/manifest mismatch states={len(final_states)} truth={len(truth)}")

    bundle = FrozenBeliefBundle.load(args.belief)
    rows = []
    per_system = Counter()
    for sid, st in sorted(final_states.items()):
        hist = list(st.history)
        if not hist:
            continue
        for plen in range(1, min(args.rounds, len(hist)) + 1):
            h = hist[:plen]
            rp = role_only_probs(bundle, h)
            x = k_features(h, rp, TRAIN_ROLES)
            rows.append({"sid": sid, "k": truth[sid], "prefix": plen, "x": x})
            per_system[sid] += 1

    if len(per_system) < 80:
        raise RuntimeError(f"too few training systems with evidence: {len(per_system)}")
    if set(truth.values()) != set(KS):
        raise RuntimeError(f"training truth lacks K classes: {sorted(set(truth.values()))}")

    X = np.vstack([r["x"] for r in rows])
    y = np.asarray([r["k"] for r in rows], dtype=int)
    groups = np.asarray([r["sid"] for r in rows], dtype=object)
    w = np.asarray([1.0 / per_system[r["sid"]] for r in rows], dtype=float)
    w *= len(per_system)

    folds = min(args.folds, min(Counter(truth.values()).values()))
    val_sets = balanced_system_folds(truth, folds, args.seed)
    C_grid = (0.01, 0.03, 0.10, 0.30, 1.0, 3.0)
    T_grid = (0.50, 0.70, 0.85, 1.00, 1.20, 1.50, 2.00)

    candidates = []
    for C in C_grid:
        oof = np.zeros((len(rows), len(KS)), dtype=float)
        covered = np.zeros(len(rows), dtype=bool)
        for val_sids in val_sets:
            va = np.asarray([sid in val_sids for sid in groups], dtype=bool)
            tr = ~va
            m = model_for(C)
            m.fit(X[tr], y[tr], clf__sample_weight=w[tr])
            oof[va] = aligned_probs(m, X[va])
            covered[va] = True
        if not covered.all():
            raise RuntimeError("OOF coverage failure")
        for T in T_grid:
            q = temp_scale(oof, T)
            candidates.append((nll(q, y), C, T))
    candidates.sort()
    best_nll, best_C, best_T = candidates[0]

    final_model = model_for(best_C)
    final_model.fit(X, y, clf__sample_weight=w)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "version": "av11-k-bootstrap-head-v1",
        "ks": KS,
        "roles": tuple(TRAIN_ROLES),
        "model": final_model,
        "temperature": float(best_T),
        "C": float(best_C),
        "rounds": int(args.rounds),
        "train_manifest_sha256": manifest.get("sha256"),
        "train_systems_with_evidence": len(per_system),
        "train_prefix_rows": len(rows),
        "oof_nll": float(best_nll),
        "uniform_nll": float(math.log(4.0)),
        "feature_source": "adaptive_v11_active.belief.k_features + role-only calibrated probabilities",
    }
    joblib.dump(artifact, out)

    summary = {k: v for k, v in artifact.items() if k != "model"}
    summary["oof_beats_uniform_nll"] = bool(best_nll < math.log(4.0))
    (out.parent / "k_head_train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"[done] frozen K head -> {out}", flush=True)


if __name__ == "__main__":
    main()
