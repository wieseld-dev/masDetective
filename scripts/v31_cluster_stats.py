from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# Import the already-installed V3.1 implementation from repo/scripts.
THIS = Path(__file__).resolve()
for p in [Path.cwd() / "scripts", THIS.parent]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import v3_1_mas_auc_semantic as v31


def load_model(outdir: Path, emb_dim: int, mode: str, seed: int):
    ckpt = torch.load(outdir / f"selector_{mode}_seed{seed}.pt", map_location="cpu")
    model = v31.AUCSelector(emb_dim, mode)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_rows(model, items):
    rows = []
    for x in items:
        scores = model.score(x.turns, x.belief, x.queries)
        pick = int(scores.argmax())
        u = x.utility.tolist()

        old_delta = None
        if x.group.old_chosen is not None:
            for j, c in enumerate(x.group.candidates):
                if c.idx == x.group.old_chosen:
                    old_delta = float(u[j])
                    break

        rows.append({
            "sid": x.group.sid,
            "round": int(x.group.round),
            "learned": float(u[pick]),
            "random": float(sum(u) / len(u)),
            "old_v14": old_delta,
            "oracle": float(max(u)),
            "before_auc": float(x.group.before_auc),
        })
    return rows


def by_key(rows):
    return {(r["sid"], r["round"]): r for r in rows}


def session_means(values_by_key):
    tmp = defaultdict(list)
    for (sid, _rd), v in values_by_key.items():
        if v is not None and math.isfinite(float(v)):
            tmp[sid].append(float(v))
    return {sid: float(np.mean(vs)) for sid, vs in tmp.items() if vs}


def exact_signflip(effects):
    arr = np.asarray(effects, dtype=float)
    n = len(arr)
    obs = float(arr.mean())
    stats = []
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        stats.append(float(np.mean(arr * np.asarray(signs))))
    stats = np.asarray(stats)
    eps = 1e-12
    p_one = float(np.mean(stats >= obs - eps))
    p_two = float(np.mean(np.abs(stats) >= abs(obs) - eps))
    return obs, p_one, p_two


def bootstrap_ci(effects, seed=20261006, B=100000):
    arr = np.asarray(effects, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(B, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def compare(name, a_by_session, b_by_session):
    sids = sorted(set(a_by_session) & set(b_by_session))
    effects = [a_by_session[s] - b_by_session[s] for s in sids]
    obs, p_one, p_two = exact_signflip(effects)
    lo, hi = bootstrap_ci(effects)
    sd = float(np.std(effects, ddof=1)) if len(effects) > 1 else float("nan")
    dz = float(obs / sd) if sd > 1e-12 else float("inf")
    return {
        "comparison": name,
        "n_sessions": len(sids),
        "mean_paired_delta_auc_difference": obs,
        "median_paired_difference": float(np.median(effects)),
        "cluster_bootstrap_95ci": [lo, hi],
        "exact_signflip_p_one_sided_A_gt_B": p_one,
        "exact_signflip_p_two_sided": p_two,
        "paired_cohens_dz": dz,
        "sessions_A_gt_B": int(sum(e > 0 for e in effects)),
        "sessions_equal": int(sum(abs(e) <= 1e-12 for e in effects)),
        "sessions_A_lt_B": int(sum(e < 0 for e in effects)),
        "session_effects": {sid: float(e) for sid, e in zip(sids, effects)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/v14_adaptive_role_v1")
    ap.add_argument("--manifest", default="artifacts/v14_adaptive_role_v1/manifest_privileged.json")
    ap.add_argument("--belief", default="artifacts/v14_adaptive_role_v1/frozen_belief.joblib")
    ap.add_argument("--outdir", default="artifacts/eqs_v3_1_mas_auc")
    ap.add_argument("--rounds", default="1,2,3,4,5,6")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    rounds = [int(x) for x in args.rounds.split(",")]
    seeds = [7, 17, 29]
    modes = ["query_only", "belief_query", "history_belief_query"]

    print("=" * 92)
    print("V3.1 CLUSTERED STATISTICAL AUDIT — DEV SESSION IS THE UNIT")
    print("=" * 92)
    print("Primary outcome: one-step delta role ROC-AUC")
    print("Inference unit: held-out MAS session (rounds averaged within session)")
    print("Exact paired sign-flip test: 2^8 possibilities when n=8")

    groups = v31.load_groups(root, Path(args.manifest), Path(args.belief), "dev", rounds)
    cache_obj = torch.load(outdir / "semantic_embeddings.pt", map_location="cpu")
    cache = cache_obj["embeddings"]
    emb_dim = int(cache_obj["dim"])
    items = v31.build_selector_items(groups, cache, emb_dim)

    print(f"DEV groups={len(items)} sessions={len(set(x.group.sid for x in items))} emb_dim={emb_dim}")

    # Per-seed predictions.
    seed_rows = {}
    for seed in seeds:
        seed_rows[seed] = {}
        for mode in modes:
            model = load_model(outdir, emb_dim, mode, seed)
            rows = evaluate_rows(model, items)
            seed_rows[seed][mode] = by_key(rows)
            mean_delta = float(np.mean([r["learned"] for r in rows]))
            print(f"loaded seed={seed:2d} mode={mode:22s} mean_delta_auc={mean_delta:.6f}")

    keys = sorted(seed_rows[seeds[0]]["history_belief_query"])

    # Architecture-level estimate: average achieved delta AUC over the three training seeds
    # for each logged decision point, then cluster by held-out MAS session.
    arch = {}
    for mode in modes:
        vals = {}
        for key in keys:
            vals[key] = float(np.mean([
                seed_rows[s][mode][key]["learned"] for s in seeds
            ]))
        arch[mode] = session_means(vals)

    # Deterministic baselines; identical across selector seeds.
    ref = seed_rows[seeds[0]]["history_belief_query"]
    random_session = session_means({k: r["random"] for k, r in ref.items()})
    old_session = session_means({k: r["old_v14"] for k, r in ref.items()})
    oracle_session = session_means({k: r["oracle"] for k, r in ref.items()})

    comparisons = [
        compare("History+Belief+Query > Query-only", arch["history_belief_query"], arch["query_only"]),
        compare("History+Belief+Query > Belief+Query", arch["history_belief_query"], arch["belief_query"]),
        compare("History+Belief+Query > Random", arch["history_belief_query"], random_session),
        compare("History+Belief+Query > Old V14", arch["history_belief_query"], old_session),
        compare("Query-only > Random", arch["query_only"], random_session),
    ]

    # Seed-level reproducibility of the key adaptivity contrast.
    seed_contrasts = {}
    for seed in seeds:
        h = session_means({k: r["learned"] for k, r in seed_rows[seed]["history_belief_query"].items()})
        q = session_means({k: r["learned"] for k, r in seed_rows[seed]["query_only"].items()})
        sids = sorted(set(h) & set(q))
        effects = [h[s] - q[s] for s in sids]
        seed_contrasts[str(seed)] = {
            "mean_history_minus_query_only": float(np.mean(effects)),
            "sessions_history_gt_query_only": int(sum(e > 0 for e in effects)),
            "session_effects": {sid: float(e) for sid, e in zip(sids, effects)},
        }

    summary = {
        "unit_of_inference": "held-out MAS session",
        "n_sessions": len(arch["history_belief_query"]),
        "rounds_per_session_expected": len(rounds),
        "training_seeds_averaged_for_primary_architecture_test": seeds,
        "session_mean_delta_auc": {
            "history_belief_query": float(np.mean(list(arch["history_belief_query"].values()))),
            "query_only": float(np.mean(list(arch["query_only"].values()))),
            "belief_query": float(np.mean(list(arch["belief_query"].values()))),
            "random": float(np.mean(list(random_session.values()))),
            "old_v14": float(np.mean(list(old_session.values()))),
            "oracle": float(np.mean(list(oracle_session.values()))),
        },
        "comparisons": comparisons,
        "seed_level_history_vs_query_only": seed_contrasts,
        "interpretation_rule": (
            "Primary evidence for adaptivity is History+Belief+Query > Query-only at the session level. "
            "Because n=8 is small, report effect, exact permutation p-value, and cluster-bootstrap CI together."
        ),
    }

    out_json = outdir / "clustered_stats.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 92)
    print("CLUSTERED RESULTS")
    print("=" * 92)
    for c in comparisons:
        lo, hi = c["cluster_bootstrap_95ci"]
        print(f"\n{c['comparison']}")
        print(f"  mean paired effect : {c['mean_paired_delta_auc_difference']:+.6f} AUC")
        print(f"  cluster 95% CI     : [{lo:+.6f}, {hi:+.6f}]")
        print(f"  exact p one-sided  : {c['exact_signflip_p_one_sided_A_gt_B']:.6f}")
        print(f"  exact p two-sided  : {c['exact_signflip_p_two_sided']:.6f}")
        print(f"  paired Cohen dz    : {c['paired_cohens_dz']:+.3f}")
        print(
            f"  session wins/ties/losses: "
            f"{c['sessions_A_gt_B']}/{c['sessions_equal']}/{c['sessions_A_lt_B']}"
        )

    print("\nSEED REPRODUCIBILITY — History minus Query-only")
    for seed, x in seed_contrasts.items():
        print(
            f"  seed={seed}: effect={x['mean_history_minus_query_only']:+.6f} "
            f"session_wins={x['sessions_history_gt_query_only']}/8"
        )

    print(f"\nSAVED: {out_json}")


if __name__ == "__main__":
    main()
