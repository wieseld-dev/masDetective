from __future__ import annotations

import importlib.util
import itertools
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


# ------------------------------------------------------------
# Reuse the already-tested HAA-V0 implementation.
# ------------------------------------------------------------

SPEC = importlib.util.spec_from_file_location(
    "haa0",
    "scripts/haa_role_acquisition_v0.py",
)
haa0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(haa0)


TRAIN_ROOT = Path(
    "artifacts/adaptive_v11_role_only_v1"
)

TRAIN_MANIFEST = Path(
    "artifacts/adaptive_v11_active_v1/manifest_privileged.json"
)

BELIEF = Path(
    "artifacts/adaptive_v11_active_v1/frozen_belief.joblib"
)

EVAL_ROOT = Path(
    "artifacts/haa_prospective_v1"
)

OUT = EVAL_ROOT / "analysis"

POOL_SIZES = [4, 8, 12]

METHODS = [
    "policy_logp",
    "uncertainty",
    "q_only",
    "direct_belief",
    "haa_full",
    "haa_focus",
]

SEED = 84317


def mirror_eval_as_train():
    """
    HAA-V0 loader expects train_* names.
    Create a private analysis-only mirror of DEV round 2.
    """
    mirror = EVAL_ROOT / "_analysis_mirror"

    if mirror.exists():
        shutil.rmtree(mirror)

    mirror.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = json.loads(
        (
            EVAL_ROOT
            / "manifest_privileged.json"
        ).read_text(encoding="utf-8")
    )

    for s in manifest["systems"]:
        s["split"] = "train"

    mp = mirror / "manifest_privileged.json"

    mp.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    mapping = {
        "states_dev_round_01.jsonl":
            "states_train_round_01.jsonl",

        "candidates_dev_round_02.jsonl":
            "candidates_train_round_02.jsonl",

        "responses_dev_round_02.jsonl":
            "responses_train_round_02.jsonl",

        "scored_dev_round_02.jsonl":
            "scored_train_round_02.jsonl",
    }

    for src, dst in mapping.items():
        p = EVAL_ROOT / src

        if not p.exists():
            raise FileNotFoundError(p)

        shutil.copy2(
            p,
            mirror / dst,
        )

    return mirror, mp


def rank_info(chosen, utility):
    y = np.asarray(utility, float)

    order = np.argsort(y)[::-1]

    rank = int(
        np.where(order == chosen)[0][0]
    ) + 1

    return rank


def system_signflip(values):
    vals = np.asarray(
        list(values.values()),
        float,
    )

    if len(vals) == 0:
        return None

    obs = float(np.mean(vals))

    if len(vals) <= 16:
        null = []

        for signs in itertools.product(
            [-1.0, 1.0],
            repeat=len(vals),
        ):
            null.append(
                np.mean(
                    vals * np.asarray(signs)
                )
            )

        null = np.asarray(null)

    else:
        rng = np.random.default_rng(SEED)

        signs = rng.choice(
            [-1.0, 1.0],
            size=(50000, len(vals)),
        )

        null = np.mean(
            signs * vals[None, :],
            axis=1,
        )

    return float(
        (
            1
            + np.sum(null >= obs - 1e-12)
        )
        / (
            1 + len(null)
        )
    )


def boot_ci(values):
    vals = np.asarray(
        list(values.values()),
        float,
    )

    if len(vals) == 0:
        return [None, None]

    rng = np.random.default_rng(
        SEED + 2
    )

    out = []

    for _ in range(20000):
        out.append(
            np.mean(
                rng.choice(
                    vals,
                    len(vals),
                    replace=True,
                )
            )
        )

    return [
        float(np.quantile(out, .025)),
        float(np.quantile(out, .975)),
    ]


def summarize(records, method):
    rr = [
        r for r in records
        if f"{method}_utility" in r
    ]

    by_gain = defaultdict(list)

    for r in rr:
        by_gain[r["sid"]].append(
            r[f"{method}_gain_random"]
        )

    sys_gain = {
        sid: float(np.mean(v))
        for sid, v in by_gain.items()
    }

    return {
        "groups": len(rr),

        "utility":
            float(np.mean([
                r[f"{method}_utility"]
                for r in rr
            ])),

        "gain_random":
            float(np.mean([
                r[f"{method}_gain_random"]
                for r in rr
            ])),

        "gain_random_ci":
            boot_ci(sys_gain),

        "gain_random_p":
            system_signflip(sys_gain),

        "top1":
            float(np.mean([
                r[f"{method}_top1"]
                for r in rr
            ])),

        "top25pct":
            float(np.mean([
                r[f"{method}_top25"]
                for r in rr
            ])),

        "regret":
            float(np.mean([
                r[f"{method}_regret"]
                for r in rr
            ])),
    }


def paired(records, a, b):
    by = defaultdict(list)

    for r in records:
        by[r["sid"]].append(
            r[f"{a}_utility"]
            -
            r[f"{b}_utility"]
        )

    sysd = {
        sid: float(np.mean(v))
        for sid, v in by.items()
    }

    return {
        "delta":
            float(np.mean(
                list(sysd.values())
            )),

        "ci":
            boot_ci(sysd),

        "p":
            system_signflip(sysd),
    }


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("[1/6] load original HAA training data")

    train_rows = haa0.load_observed_rows(
        TRAIN_ROOT,
        TRAIN_MANIFEST,
        BELIEF,
    )

    print(
        f"  train rows={len(train_rows)} "
        f"systems={len(set(x['sid'] for x in train_rows))}"
    )

    print("[2/6] construct held-out analysis mirror")

    mirror, mirror_manifest = (
        mirror_eval_as_train()
    )

    eval_rows = haa0.load_observed_rows(
        mirror,
        mirror_manifest,
        BELIEF,
    )

    print(
        f"  eval rows={len(eval_rows)} "
        f"systems={len(set(x['sid'] for x in eval_rows))}"
    )

    # Hard anti-leak check.
    train_sids = set(
        x["sid"] for x in train_rows
    )

    eval_sids = set(
        x["sid"] for x in eval_rows
    )

    overlap = train_sids & eval_sids

    if overlap:
        raise RuntimeError(
            f"TRAIN/EVAL SYSTEM LEAK: {sorted(overlap)}"
        )

    print("[3/6] frozen MiniLM embeddings")

    combined = train_rows + eval_rows
    haa0.embed_queries(combined)

    print("[4/6] fit HAA ONLY on original 24 systems")

    fold = haa0.fit_fold(
        train_rows,
        list(range(len(train_rows))),
    )

    print("[5/6] score held-out candidates")

    groups = defaultdict(list)

    for r in eval_rows:
        groups[r["sid"]].append(r)

    all_results = {}

    for pool in POOL_SIZES:
        records = []

        for sid, candidates_all in sorted(
            groups.items()
        ):
            candidates = [
                r for r in candidates_all
                if int(r["candidate_index"]) < pool
            ]

            if len(candidates) < 2:
                continue

            candidates = sorted(
                candidates,
                key=lambda x: x["candidate_index"],
            )

            base = candidates[0]

            H, w, inclusion = (
                haa0.particles_from_belief(
                    base["role_probs"],
                    base["k_probs"],
                    top_per_k=16,
                )
            )

            utilities = np.asarray([
                r["utility"]
                for r in candidates
            ])

            random_u = float(
                np.mean(utilities)
            )

            oracle_u = float(
                np.max(utilities)
            )

            worst_u = float(
                np.min(utilities)
            )

            gap = (
                oracle_u
                - random_u
            )

            rng_range = (
                oracle_u
                - worst_u
            )

            scores = {
                m: []
                for m in METHODS
            }

            for r in candidates:
                s = haa0.score_candidate(
                    r,
                    fold,
                    H,
                    w,
                    inclusion,
                )

                for m in METHODS:
                    scores[m].append(
                        float(s[m])
                    )

            rec = {
                "sid": sid,
                "pool": pool,
                "n_valid": len(candidates),
                "random_utility": random_u,
                "oracle_utility": oracle_u,
                "oracle_gap": gap,
            }

            for m in METHODS:
                sc = np.asarray(
                    scores[m],
                    float,
                )

                choice = int(
                    np.argmax(sc)
                )

                u = float(
                    utilities[choice]
                )

                rank = rank_info(
                    choice,
                    utilities,
                )

                regret = (
                    (oracle_u-u) / rng_range
                    if rng_range > 1e-12
                    else 0.0
                )

                rec[f"{m}_utility"] = u

                rec[f"{m}_gain_random"] = (
                    u - random_u
                )

                rec[f"{m}_top1"] = int(
                    rank == 1
                )

                rec[f"{m}_top25"] = int(
                    rank
                    <= max(
                        1,
                        int(np.ceil(
                            .25*len(candidates)
                        )),
                    )
                )

                rec[f"{m}_regret"] = float(
                    regret
                )

            records.append(rec)

        if not records:
            raise RuntimeError(
                f"No evaluable groups for pool={pool}"
            )

        methods = {
            m: summarize(
                records,
                m,
            )
            for m in METHODS
        }

        oracle_gap = float(
            np.mean([
                x["oracle_gap"]
                for x in records
            ])
        )

        haa_gain = (
            methods["haa_focus"][
                "gain_random"
            ]
        )

        capture = (
            haa_gain / oracle_gap
            if oracle_gap > 1e-12
            else 0.0
        )

        all_results[str(pool)] = {
            "groups": len(records),

            "mean_valid_candidates":
                float(np.mean([
                    x["n_valid"]
                    for x in records
                ])),

            "random_top1_chance":
                float(np.mean([
                    1.0/x["n_valid"]
                    for x in records
                ])),

            "oracle_gap":
                oracle_gap,

            "haa_oracle_opportunity_capture":
                capture,

            "methods":
                methods,

            "haa_vs_q_only":
                paired(
                    records,
                    "haa_focus",
                    "q_only",
                ),

            "haa_vs_direct":
                paired(
                    records,
                    "haa_focus",
                    "direct_belief",
                ),

            "haa_vs_policy":
                paired(
                    records,
                    "haa_focus",
                    "policy_logp",
                ),
        }

        with (
            OUT
            / f"groups_pool{pool}.jsonl"
        ).open(
            "w",
            encoding="utf-8",
        ) as f:
            for r in records:
                f.write(
                    json.dumps(r)
                    + "\n"
                )

    print("[6/6] decision")

    primary = all_results["12"]

    h = primary["methods"]["haa_focus"]
    d = primary["haa_vs_direct"]
    q = primary["haa_vs_q_only"]

    # Pre-declared continuation gate:
    #
    # 1. HAA really beats random on unseen systems.
    # 2. It captures >=25% of the available oracle opportunity.
    # 3. It is directionally better than BOTH strong learned baselines.
    # 4. It has a better top-1 rate than direct q+belief.
    #
    strong = (
        h["gain_random"] > 0
        and h["gain_random_p"] <= .05
        and primary[
            "haa_oracle_opportunity_capture"
        ] >= .25
        and d["delta"] > 0
        and q["delta"] > 0
        and h["top1"]
            > primary["methods"][
                "direct_belief"
            ]["top1"]
    )

    weak = (
        h["gain_random"] > 0
        and h["gain_random_p"] <= .05
        and primary[
            "haa_oracle_opportunity_capture"
        ] >= .20
    )

    if strong:
        verdict = "GO"
    elif weak:
        verdict = "WEAK_GO"
    else:
        verdict = "NO_GO"

    summary = {
        "experiment":
            "haa_prospective_heldout_v1",

        "train_systems":
            len(train_sids),

        "heldout_systems":
            len(eval_sids),

        "system_overlap":
            len(overlap),

        "results":
            all_results,

        "decision": {
            "verdict":
                verdict,

            "primary_pool":
                12,

            "criteria":
                "Held-out HAA must significantly beat random, "
                "capture >=25% oracle opportunity, and directionally "
                "beat both q-only and direct q+belief for GO.",
        },
    }

    (
        OUT / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    L = []

    L.append("="*104)
    L.append(
        "HAA PROSPECTIVE HELD-OUT V1"
    )
    L.append("="*104)

    L.append(
        f"training systems={len(train_sids)} | "
        f"held-out systems={len(eval_sids)} | "
        f"overlap={len(overlap)}"
    )

    L.append(
        "One bootstrap probe per held-out MAS. "
        "HAA never sees candidate outputs before ranking."
    )

    for pool in POOL_SIZES:
        x = all_results[str(pool)]

        L.append("")
        L.append("-"*104)

        L.append(
            f"POOL PREFIX N={pool} | "
            f"groups={x['groups']} | "
            f"mean-valid={x['mean_valid_candidates']:.2f} | "
            f"random-top1≈{x['random_top1_chance']:.3f}"
        )

        L.append(
            f"  oracle opportunity = {x['oracle_gap']:+.6f}"
        )

        for m in METHODS:
            z = x["methods"][m]
            ci = z["gain_random_ci"]

            L.append(
                f"  {m:16s} "
                f"U={z['utility']:+.6f} "
                f"gain/random={z['gain_random']:+.6f} "
                f"CI=[{ci[0]:+.6f},{ci[1]:+.6f}] "
                f"p={z['gain_random_p']:.5f} "
                f"top1={z['top1']:.3f} "
                f"top25%={z['top25pct']:.3f} "
                f"regret={z['regret']:.3f}"
            )

        L.append(
            f"  HAA oracle-opportunity capture = "
            f"{x['haa_oracle_opportunity_capture']:.3f}"
        )

        for label, key in (
            ("q-only", "haa_vs_q_only"),
            ("direct", "haa_vs_direct"),
            ("policy", "haa_vs_policy"),
        ):
            z = x[key]
            ci = z["ci"]

            L.append(
                f"  HAA vs {label:7s}: "
                f"delta-U={z['delta']:+.6f} "
                f"CI=[{ci[0]:+.6f},{ci[1]:+.6f}] "
                f"p={z['p']:.5f}"
            )

    L.append("")
    L.append("="*104)
    L.append(
        f"VERDICT: {verdict}"
    )

    if verdict == "GO":
        L.append(
            "Direction survives prospective held-out evaluation. "
            "Build the sequential HAA controller."
        )
    elif verdict == "WEAK_GO":
        L.append(
            "Probe ranking contains real held-out signal, but explicit "
            "hypothesis modeling is not yet convincingly superior to "
            "the strongest learned baselines."
        )
    else:
        L.append(
            "HAA does not survive the prospective held-out test strongly "
            "enough. Do not build the sequential controller from this version."
        )

    L.append("="*104)

    report = "\n".join(L)

    (
        OUT / "REPORT.txt"
    ).write_text(
        report + "\n",
        encoding="utf-8",
    )

    print()
    print(report)


if __name__ == "__main__":
    main()
