#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("analysis").resolve()))

import observation_model_v1 as om
import visible_adaptive_bridge_v1 as vb

from adaptive_v11_active.schema import PublicBeliefState
from targets.single_router_zoo_v11 import MODES, KS, _make_train_specs, get_v11_split, validate_v11_design
from v11.role_universe import TRAIN_ROLES

from v3_common import exact_hypotheses, k_probs, role_probs, save_matrix_npz

HIST_ROOT = Path("artifacts/adaptive_v11_curriculum_v2")
OBS = Path("artifacts/observation_model_v1")
VIS = Path("artifacts/visible_output_fingerprint_v1")
CAL = Path("artifacts/calibrated_bayes_diag_v1")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def role_key(roles):
    return tuple(sorted(map(str, roles)))


def collect_role_sets_from_json(obj, out: set[tuple[str, ...]]):
    if isinstance(obj, dict):
        for key in ("roles", "worker_roles", "truth_roles", "truth_roles_training_only"):
            v = obj.get(key)
            if isinstance(v, list) and 3 <= len(v) <= 6:
                s = role_key(v)
                if all(r in TRAIN_ROLES for r in s):
                    out.add(s)
        for v in obj.values():
            collect_role_sets_from_json(v, out)
    elif isinstance(obj, list):
        for v in obj:
            collect_role_sets_from_json(v, out)


def forbidden_role_sets(own_manifest: Path) -> set[tuple[str, ...]]:
    out = {role_key(s.worker_roles) for s in get_v11_split("train")}
    patterns = [
        "artifacts/**/manifest_privileged.json",
        "artifacts/**/*manifest*.json",
    ]
    seen = set()
    for pat in patterns:
        for p in Path(".").glob(pat):
            try:
                p = p.resolve()
            except Exception:
                continue
            if p == own_manifest.resolve() or p in seen:
                continue
            seen.add(p)
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            collect_role_sets_from_json(obj, out)
    return out


def select_balanced(specs, seed: int):
    rng = random.Random(seed)
    used = set()
    role_counts = Counter({r: 0 for r in TRAIN_ROLES})
    train = []

    # 16 train: one fresh system per mode x K cell.
    for mode in MODES:
        for kval in (3,4,5,6):
            pool = [
                s for s in specs
                if s.execution_mode == mode
                and len(s.worker_roles) == kval
                and role_key(s.worker_roles) not in used
            ]
            if not pool:
                raise RuntimeError(f"no fresh candidate for train mode={mode} K={kval}")
            def cost(s):
                return sum((role_counts[r] + 1) ** 2 - role_counts[r] ** 2 for r in s.worker_roles) + rng.random()*1e-6
            best = min(pool, key=cost)
            train.append(best)
            used.add(role_key(best.worker_roles))
            role_counts.update(best.worker_roles)

    # 8 untouched dev: two systems per mode, balanced K pattern.
    dev_pattern = {
        MODES[0]: (3,4),
        MODES[1]: (5,6),
        MODES[2]: (3,5),
        MODES[3]: (4,6),
    }
    dev_counts = Counter({r: 0 for r in TRAIN_ROLES})
    dev = []
    for mode in MODES:
        for kval in dev_pattern[mode]:
            pool = [
                s for s in specs
                if s.execution_mode == mode
                and len(s.worker_roles) == kval
                and role_key(s.worker_roles) not in used
            ]
            if not pool:
                raise RuntimeError(f"no fresh candidate for dev mode={mode} K={kval}")
            def cost(s):
                return sum((dev_counts[r] + 1) ** 2 - dev_counts[r] ** 2 for r in s.worker_roles) + rng.random()*1e-6
            best = min(pool, key=cost)
            dev.append(best)
            used.add(role_key(best.worker_roles))
            dev_counts.update(best.worker_roles)

    return train, dev


def manifest_row(spec, split, idx):
    raw = "|".join(map(str, sorted(spec.worker_roles))) + f"|{spec.execution_mode}|{getattr(spec,'style','s1')}|{split}|{idx}"
    sid = "v3_" + hashlib.sha256(raw.encode()).hexdigest()[:16]
    return {
        "session_id": sid,
        "split": split,
        "k": len(spec.worker_roles),
        "roles": list(map(str, spec.worker_roles)),
        "mode": str(spec.execution_mode),
        "style": str(getattr(spec, "style", "s1")),
    }


def recursive_find(obj, target_keys):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in target_keys:
                return v
        for v in obj.values():
            z = recursive_find(v, target_keys)
            if z is not None:
                return z
    elif isinstance(obj, list):
        for v in obj:
            z = recursive_find(v, target_keys)
            if z is not None:
                return z
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/v3_coverage_elimination_pilot")
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest_privileged.json"

    required = [
        Path("artifacts/robust_visible_generator_v1_epoch2/policy/adapter_model.safetensors"),
        Path("artifacts/v11_policy/adapter_model.safetensors"),
        OBS / "model.pt",
        OBS / "split_systems.json",
        OBS / "query_embeddings.npy",
        VIS / "output_embeddings.npy",
        CAL / "summary.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("missing required artifacts:\n" + "\n".join(missing))

    needed_api = [
        "labels", "fit_visible_detector", "visible_probs", "particle_logits",
        "calibrated_hidden_probs", "detector_channel_probs", "update_weights",
    ]
    absent = [x for x in needed_api if not hasattr(vb, x)]
    if absent:
        raise SystemExit(f"visible_adaptive_bridge_v1 API mismatch, missing: {absent}")

    print("[api]")
    for name in needed_api:
        print(name, inspect.signature(getattr(vb, name)))

    validate_v11_design()

    if manifest_path.exists() and not args.force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print("[manifest] reusing existing", manifest_path)
    else:
        forbidden = forbidden_role_sets(manifest_path)
        candidates = _make_train_specs(per_mode=512, seed=args.seed)
        candidates = [
            s for s in candidates
            if role_key(s.worker_roles) not in forbidden
        ]
        train, dev = select_balanced(candidates, args.seed)
        systems = [
            *(manifest_row(s, "train", i) for i, s in enumerate(train)),
            *(manifest_row(s, "dev", i) for i, s in enumerate(dev)),
        ]
        manifest = {
            "protocol": "v3-coverage-elimination-visible-bayes-pilot",
            "seed": args.seed,
            "systems": systems,
            "train_systems": len(train),
            "dev_systems": len(dev),
            "forbidden_role_sets_seen": len(forbidden),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("[manifest] wrote", manifest_path)

    # Exact joint hypothesis space.
    Z, Klabels, W0 = exact_hypotheses(len(TRAIN_ROLES))
    if len(Z) != 60249:
        raise RuntimeError(f"expected 60249 hypotheses, got {len(Z)}")
    np.save(root / "hypotheses.npy", Z)
    np.save(root / "klabels.npy", Klabels)
    np.save(root / "prior_weights.npy", W0)

    # Public initial state + exact posterior + coverage for both splits.
    kp0 = k_probs(W0, Klabels)
    rp0 = role_probs(W0, Z, TRAIN_ROLES)
    for split_name in ("train", "dev"):
        sysrows = [x for x in manifest["systems"] if x["split"] == split_name]
        ids = [x["session_id"] for x in sysrows]
        states = []
        for sid in ids:
            st = PublicBeliefState(
                session_id=sid,
                split=split_name,
                round_index=0,
                k_probs=dict(kp0),
                role_probs=dict(rp0),
                history=[],
            )
            st.validate(TRAIN_ROLES, (3,4,5,6))
            states.append(st.to_public_dict())
        write_jsonl(root / f"states_{split_name}_round_00.jsonl", states)
        save_matrix_npz(
            root / f"posterior_{split_name}_round_00.npz",
            ids,
            np.stack([W0 for _ in ids]),
            "weights",
        )
        save_matrix_npz(
            root / f"coverage_{split_name}_round_00.npz",
            ids,
            np.zeros((len(ids), len(TRAIN_ROLES)), dtype=np.float32),
            "coverage",
        )

    # Fit deployment-visible fingerprint on historical TRAIN; calibrate on historical VAL.
    print("[visible] loading historical examples")
    examples, obs_roles, _, _ = om.collect_examples(HIST_ROOT)
    if list(obs_roles) != list(TRAIN_ROLES):
        raise RuntimeError("observation-model role order != TRAIN_ROLES")

    split = json.loads((OBS / "split_systems.json").read_text())
    trset, vaset = set(map(str, split["train"])), set(map(str, split["val"]))
    tri = np.asarray([i for i,x in enumerate(examples) if str(x["system_id"]) in trset], dtype=int)
    vai = np.asarray([i for i,x in enumerate(examples) if str(x["system_id"]) in vaset], dtype=int)

    Q = np.load(OBS / "query_embeddings.npy").astype(np.float32)
    O = np.load(VIS / "output_embeddings.npy").astype(np.float32)
    if len(Q) != len(examples) or len(O) != len(examples):
        raise RuntimeError(f"embedding length mismatch Q={len(Q)} O={len(O)} examples={len(examples)}")
    X = np.concatenate([Q, O], axis=1)
    Y = vb.labels(examples, TRAIN_ROLES, "selected_roles")
    models, cals, thr, tpr, fpr, metrics = vb.fit_visible_detector(X, Y, tri, vai)

    # Observation-model positive-class correction.
    ri = {r:i for i,r in enumerate(TRAIN_ROLES)}
    pos = np.zeros(len(TRAIN_ROLES), dtype=np.float64)
    for i in tri:
        for r in examples[i]["selected_roles"]:
            pos[ri[r]] += 1
    posw = np.clip((len(tri) - pos) / np.maximum(pos, 1.0), 1.0, 20.0)
    log_posw = np.log(posw)

    cal = json.loads((CAL / "summary.json").read_text())
    T = recursive_find(cal, {"best_T", "best_temperature"})
    beta = recursive_find(cal, {"selected_beta_by_validation_log_gain", "selected_beta", "best_beta"})
    if T is None:
        T = 1.25
    if beta is None:
        beta = 0.10

    asset = {
        "roles": list(TRAIN_ROLES),
        "models": models,
        "cals": cals,
        "thresholds": np.asarray(thr, dtype=np.float64),
        "tpr": np.asarray(tpr, dtype=np.float64),
        "fpr": np.asarray(fpr, dtype=np.float64),
        "visible_metrics": metrics,
        "log_posweight": np.asarray(log_posw, dtype=np.float64),
        "temperature": float(T),
        "beta": float(beta),
    }
    joblib.dump(asset, root / "visible_bridge.joblib")

    summary = {
        "root": str(root),
        "exact_hypotheses": len(Z),
        "train_systems": len([x for x in manifest["systems"] if x["split"] == "train"]),
        "dev_systems": len([x for x in manifest["systems"] if x["split"] == "dev"]),
        "visible_beta": float(beta),
        "temperature": float(T),
        "visible_metrics": metrics,
        "max_train_calls_for_6_rounds_g4": 16 * 6 * 4,
        "truth_boundary": "truth used only in reward/metrics; posterior update is query + final_output only",
    }
    def _json_default(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        raise TypeError(f"Object of type {type(x).__name__} is not JSON serializable")

    (root / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
