from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

# Importing this module registers EXTRA_ROLE_PROMPTS in every fresh process.
from targets.single_router_zoo_v11 import MODES, KS, _make_train_specs, get_v11_split, validate_v11_design
from v11.role_universe import TRAIN_ROLES
from adaptive_v11_active.io import sha256_json, write_json


def key(roles):
    return tuple(sorted(map(str, roles)))


def load_forbidden_from_json(path: Path) -> set[tuple[str, ...]]:
    out: set[tuple[str, ...]] = set()
    if not path.exists():
        return out
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return out
    for field in ("systems", "known_specs", "dev_specs"):
        for row in obj.get(field, []) if isinstance(obj, dict) else []:
            roles = row.get("roles", row.get("worker_roles", ()))
            if roles:
                out.add(key(roles))
    return out


def main():
    ap = argparse.ArgumentParser(description="Create fresh disjoint Adaptive-V11 RL systems")
    ap.add_argument("--output", default="artifacts/adaptive_v11_active_v1/manifest_privileged.json")
    ap.add_argument("--seed", type=int, default=864209)
    ap.add_argument("--baseline-manifest", default="artifacts/adaptive_v1/manifest.json")
    ap.add_argument("--strong-manifest", default="artifacts/v11_2_strong_eval/benchmark_manifest.json")
    args = ap.parse_args()

    validate_v11_design()
    forbidden = {key(s.worker_roles) for s in get_v11_split("train")}
    # Also reconstruct strong-eval role sets to avoid ordering dependence.
    strong_candidates = _make_train_specs(per_mode=8, seed=24681357)
    v11_train = {key(s.worker_roles) for s in get_v11_split("train")}
    strong_fresh = [s for s in strong_candidates if key(s.worker_roles) not in v11_train]
    for mode in MODES:
        for s in [x for x in strong_fresh if x.execution_mode == mode][:6]:
            forbidden.add(key(s.worker_roles))
    forbidden |= load_forbidden_from_json(Path(args.strong_manifest))

    # The previous fixed-probe adaptive bank is a baseline, so the RL systems must be disjoint from it.
    baseline_paths = []
    bp = Path(args.baseline_manifest)
    if bp.exists():
        baseline_paths.append(bp)
    baseline_paths.extend(p for p in sorted(Path("artifacts/adaptive_v1").glob("*manifest*.json")) if p not in baseline_paths)
    for p in baseline_paths:
        forbidden |= load_forbidden_from_json(p)

    candidates = _make_train_specs(per_mode=96, seed=args.seed)
    candidates = [s for s in candidates if key(s.worker_roles) not in forbidden]
    rng = random.Random(args.seed)
    counts = Counter({r: 0 for r in TRAIN_ROLES})
    chosen = []
    used = set()
    for mode in MODES:
        for kval in KS:
            pool = [s for s in candidates if s.execution_mode == mode and len(s.worker_roles) == kval and key(s.worker_roles) not in used]
            for _ in range(2):
                if not pool:
                    raise RuntimeError(f"not enough fresh systems for mode={mode} K={kval}")
                def cost(s):
                    role_cost = sum((counts[r] + 1) ** 2 - counts[r] ** 2 for r in s.worker_roles)
                    return role_cost + rng.random() * 1e-4
                best = min(pool, key=cost)
                pool.remove(best)
                used.add(key(best.worker_roles))
                counts.update(best.worker_roles)
                chosen.append(best)

    # Same balanced 24 train / 8 dev pattern: two dev systems per mode and two per K.
    dev_k_by_mode = {
        MODES[0]: (3, 4),
        MODES[1]: (5, 6),
        MODES[2]: (3, 5),
        MODES[3]: (4, 6),
    }
    train, dev = [], []
    dev_counts = Counter({r: 0 for r in TRAIN_ROLES})
    for mode in MODES:
        designated = set(dev_k_by_mode[mode])
        for kval in KS:
            cell = [s for s in chosen if s.execution_mode == mode and len(s.worker_roles) == kval]
            if kval in designated:
                d = min(cell, key=lambda s: sum((dev_counts[r] + 1) ** 2 - dev_counts[r] ** 2 for r in s.worker_roles))
                dev.append(d)
                dev_counts.update(d.worker_roles)
                train.extend([s for s in cell if s is not d])
            else:
                train.extend(cell)
    assert len(train) == 24 and len(dev) == 8

    rows = []
    for split, systems in (("train", train), ("dev", dev)):
        for s in systems:
            opaque = hashlib.sha256(
                f"adaptive-v11|{args.seed}|{split}|{s.execution_mode}|{','.join(sorted(s.worker_roles))}".encode()
            ).hexdigest()[:16]
            rows.append({
                "session_id": f"av11_{opaque}",
                "split": split,
                "mode": s.execution_mode,
                "style": s.style,
                "roles": list(s.worker_roles),
                "k": len(s.worker_roles),
            })

    public_canon = [{k: v for k, v in r.items() if k not in {"roles", "k", "mode", "style"}} for r in rows]
    obj = {
        "version": "adaptive_v11_active_manifest_v1",
        "seed": args.seed,
        "systems": rows,
        "sha256": sha256_json(rows),
        "notes": {
            "privileged": True,
            "policy_must_never_read_this_file": True,
            "baseline_manifests_excluded": [str(p) for p in baseline_paths],
        },
    }
    write_json(args.output, obj)
    print(f"[manifest] systems={len(rows)} train={len(train)} dev={len(dev)} sha256={obj['sha256']}")
    print("[mode]", dict(Counter(r["mode"] for r in rows)))
    print("[K]", dict(Counter(r["k"] for r in rows)))
    print("[role]", json.dumps(dict(Counter(x for r in rows for x in r["roles"])), sort_keys=True))
    print(f"[done] privileged manifest={args.output}")


if __name__ == "__main__":
    main()
