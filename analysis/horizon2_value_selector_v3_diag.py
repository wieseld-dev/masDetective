#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import analysis.observation_model_v1 as om
import analysis.realized_posterior_gain_diag_v1 as rg
import analysis.build_visible_realized_gain_v3 as vg
import analysis.train_value_selector_v3 as tv


def read_jsonl(p):
    with Path(p).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def seed32(*x):
    s = "|".join(map(str, x)).encode()
    return int.from_bytes(
        hashlib.sha256(s).digest()[:8], "big"
    ) % (2**32)


def load_value(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)

    m = tv.ValueNetV3(
        int(ck["feature_dim"]),
        int(ck["hidden"]),
        float(ck["dropout"]),
    )
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


def kprior_from_weights(W, Klabels):
    out = {}
    for k in sorted(set(map(int, Klabels))):
        out[int(k)] = float(W[Klabels == k].sum())

    s = sum(out.values())
    return {k: v / s for k, v in out.items()}


def marginals(W, Z, Klabels, k_keys):
    W = np.asarray(W, np.float64)
    W = W / W.sum()

    p = np.einsum("m,mr->r", W, Z)

    kp = np.asarray([
        W[Klabels == k].sum()
        for k in k_keys
    ], dtype=np.float32)

    return p.astype(np.float32), kp


def feature(
    p, kp, qemb,
    mi_k, mi_id, mi_z, exposure,
    ck,
):
    p = np.asarray(p, np.float32)
    kp = np.asarray(kp, np.float32)

    extra = np.asarray([
        float(tv.hbin(p).sum()),
        float(np.mean(p * (1-p))),
        tv.entropy_cat(kp),
        float(mi_k),
        float(mi_id),
        float(mi_z),
        float(exposure),
    ], dtype=np.float32)

    x = np.concatenate([
        p,
        p * (1-p),
        kp,
        qemb.astype(np.float32),
        extra,
    ]).astype(np.float32)

    mu = np.asarray(ck["x_mean"], np.float32)
    sd = np.asarray(ck["x_std"], np.float32)

    return ((x - mu) / sd).astype(np.float32)


@torch.no_grad()
def role_value(model, ck, X):
    x = torch.from_numpy(
        np.stack(X).astype(np.float32)
    )
    r, _ = model(x)

    return (
        r.numpy() * float(ck["role_target_std"])
        + float(ck["role_target_mean"])
    )


def make_sensor_pools(train_rows, roles):
    pools = {
        (r, y): []
        for r in roles
        for y in (0, 1)
    }

    for x in train_rows:
        sel = set(x["selected_roles_not_used_for_update"])
        ss = x["sensor_scores_visible_only"]

        for j, r in enumerate(roles):
            pools[(r, int(r in sel))].append(
                float(ss[j])
            )

    for k, v in pools.items():
        if len(v) < 5:
            raise RuntimeError(
                f"too few sensor samples {k}: {len(v)}"
            )

        pools[k] = np.asarray(v, np.float64)

    return pools


def sample_visible_scores(
    rng,
    p_activation_particle,
    roles,
    pools,
):
    y = (
        rng.random(len(roles))
        < np.asarray(p_activation_particle)
    )

    s = np.empty(len(roles), np.float64)

    for j, r in enumerate(roles):
        pool = pools[(r, int(y[j]))]
        s[j] = pool[rng.integers(len(pool))]

    return s


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--r1-train",
        default="artifacts/visible_realized_gain_v3/train.jsonl",
    )
    ap.add_argument(
        "--r1-val",
        default="artifacts/visible_realized_gain_v3/val.jsonl",
    )
    ap.add_argument(
        "--r1-value",
        default="artifacts/value_selector_v3/value_selector_v3.pt",
    )
    ap.add_argument(
        "--r2-value",
        default="artifacts/value_selector_v3_r2_retry/value_selector_v3.pt",
    )
    ap.add_argument(
        "--calibration",
        default="artifacts/visible_realized_gain_v3/channel_calibration.json",
    )
    ap.add_argument(
        "--obs-dir",
        default="artifacts/observation_model_v1",
    )
    ap.add_argument(
        "--root",
        default="artifacts/adaptive_v11_curriculum_v2",
    )
    ap.add_argument(
        "--out",
        default="artifacts/horizon2_value_selector_v3_diag",
    )

    ap.add_argument("--branches", type=int, default=16)
    ap.add_argument("--particles-per-k", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=1391)

    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    train = read_jsonl(a.r1_train)
    val = read_jsonl(a.r1_val)

    obsdir = Path(a.obs_dir)

    obsck = torch.load(
        obsdir / "model.pt",
        map_location="cpu",
    )

    roles = list(obsck["role_order"])
    role_i = {r:i for i,r in enumerate(roles)}

    qemb_all = np.load(
        obsdir / "query_embeddings.npy"
    ).astype(np.float32)

    fwd = om.ObservationNet(
        qdim=obsck["query_embedding_dim"],
        n_roles=len(roles),
        hidden=obsck["hidden"],
        dropout=obsck["dropout"],
    )
    fwd.load_state_dict(obsck["state_dict"])
    fwd.eval()

    r1_model, r1ck = load_value(a.r1_value)
    r2_model, r2ck = load_value(a.r2_value)

    if r1ck["roles"] != roles or r2ck["roles"] != roles:
        raise RuntimeError("role order mismatch")

    k_keys = list(map(int, r1ck["k_keys"]))

    cal = json.loads(
        Path(a.calibration).read_text()
    )

    pools = make_sensor_pools(train, roles)

    examples, roles2, _, _ = om.collect_examples(
        Path(a.root)
    )
    if list(roles2) != roles:
        raise RuntimeError("example role mismatch")

    max_round = max(int(x["round"]) for x in examples)

    groups = defaultdict(list)
    for x in val:
        groups[x["state_id"]].append(x)

    stats = defaultdict(list)
    details = []

    for gi, (sid, rows) in enumerate(
        sorted(groups.items()), 1
    ):
        rows = sorted(
            rows,
            key=lambda z: int(z["candidate_index"]),
        )

        x0 = rows[0]

        p0 = np.asarray([
            x0["role_probs_before"][r]
            for r in roles
        ], np.float32)

        kp0_dict = x0["k_probs_before"]

        rng = np.random.default_rng(
            seed32(sid, a.seed)
        )

        Z, W, Klabels, Kprior = rg.stratified_particles(
            kp0_dict,
            x0["role_probs_before"],
            roles,
            a.particles_per_k,
            rng,
        )

        W = np.asarray(W, np.float64)
        W /= W.sum()

        kp0 = np.asarray([
            kp0_dict.get(str(k), 0.0)
            for k in k_keys
        ], np.float32)

        qemb = np.stack([
            qemb_all[int(x["example_index"])]
            for x in rows
        ]).astype(np.float32)

        r1norm = 1.0 / max_round

        P1 = rg.particle_predictions(
            fwd,
            qemb,
            p0,
            kp0,
            r1norm,
            Z,
            "cpu",
            a.batch_size,
        )

        mi_k1, mi_id1, mi_z1, exp1 = (
            rg.predicted_mi(
                P1, W, Klabels, Kprior
            )
        )

        root_X = [
            feature(
                p0, kp0, qemb[j],
                mi_k1[j], mi_id1[j],
                mi_z1[j], exp1[j],
                r1ck,
            )
            for j in range(len(rows))
        ]

        q1_value = role_value(
            r1_model, r1ck, root_X
        )

        h2_scores = []

        for j1 in range(len(rows)):
            leaf_best = []

            for b in range(a.branches):
                brng = np.random.default_rng(
                    seed32(
                        sid,
                        rows[j1]["candidate_index"],
                        b,
                        a.seed,
                    )
                )

                m = brng.choice(
                    len(W),
                    p=W / W.sum(),
                )

                scores = sample_visible_scores(
                    brng,
                    P1[j1, m],
                    roles,
                    pools,
                )

                Wp = vg.visible_posterior_update(
                    P1[j1],
                    scores,
                    roles,
                    cal,
                    W,
                    a.beta,
                )

                p2, kp2 = marginals(
                    Wp,
                    Z,
                    Klabels,
                    k_keys,
                )

                Kprior2 = kprior_from_weights(
                    Wp,
                    Klabels,
                )

                P2 = rg.particle_predictions(
                    fwd,
                    qemb,
                    p2,
                    kp2,
                    2.0 / max_round,
                    Z,
                    "cpu",
                    a.batch_size,
                )

                mi_k2, mi_id2, mi_z2, exp2 = (
                    rg.predicted_mi(
                        P2,
                        Wp,
                        Klabels,
                        Kprior2,
                    )
                )

                leaf_X = [
                    feature(
                        p2, kp2, qemb[j],
                        mi_k2[j],
                        mi_id2[j],
                        mi_z2[j],
                        exp2[j],
                        r2ck,
                    )
                    for j in range(len(rows))
                    if j != j1
                ]

                if not leaf_X:
                    leaf_best.append(0.0)
                    continue

                lv = role_value(
                    r2_model,
                    r2ck,
                    leaf_X,
                )

                leaf_best.append(float(np.max(lv)))

            h2_scores.append(
                float(q1_value[j1])
                + a.gamma * float(np.mean(leaf_best))
            )

        h2_scores = np.asarray(h2_scores)

        realized = np.asarray([
            x["role_brier_gain_training_only"]
            for x in rows
        ], dtype=float)

        mi_baseline = np.asarray([
            x["mi_z_baseline"]
            for x in rows
        ], dtype=float)

        ig = int(np.argmax(q1_value))
        ih = int(np.argmax(h2_scores))
        im = int(np.argmax(mi_baseline))
        io = int(np.argmax(realized))

        stats["greedy"].append(float(realized[ig]))
        stats["h2"].append(float(realized[ih]))
        stats["mi_z"].append(float(realized[im]))
        stats["random"].append(float(realized.mean()))
        stats["oracle"].append(float(realized[io]))
        stats["changed"].append(int(ig != ih))

        details.append({
            "state_id": sid,

            "greedy_candidate":
                int(rows[ig]["candidate_index"]),
            "h2_candidate":
                int(rows[ih]["candidate_index"]),
            "mi_z_candidate":
                int(rows[im]["candidate_index"]),
            "oracle_candidate":
                int(rows[io]["candidate_index"]),

            "greedy_realized_gain":
                float(realized[ig]),
            "h2_realized_gain":
                float(realized[ih]),
            "mi_z_realized_gain":
                float(realized[im]),
            "random_mean_gain":
                float(realized.mean()),
            "oracle_gain":
                float(realized[io]),

            "greedy_predicted_q1":
                float(q1_value[ig]),
            "h2_score":
                float(h2_scores[ih]),
        })

        if gi % 10 == 0 or gi == len(groups):
            print(
                f"{gi}/{len(groups)} states",
                flush=True,
            )

    def mean(k):
        return float(np.mean(stats[k]))

    random_gain = mean("random")
    oracle_gain = mean("oracle")

    def capture(v):
        den = oracle_gain - random_gain
        return (
            (v - random_gain) / den
            if abs(den) > 1e-12
            else None
        )

    summary = {
        "protocol":
            "horizon2_value_selector_v3_fixed_bank_diag",

        "states": len(groups),
        "branches": a.branches,
        "particles_per_k": a.particles_per_k,
        "gamma": a.gamma,

        "test_used": False,
        "truth_used_for_selection": False,
        "real_belief_mutated_during_planning": False,

        "visible_branch_model":
            "TRAIN-only empirical V13.8b score bootstrap "
            "+ continuous LR posterior update",

        "q1_difference_rate":
            mean("changed"),

        "mean_realized_role_gain": {
            "greedy_q1_value": mean("greedy"),
            "h2_value": mean("h2"),
            "mi_z": mean("mi_z"),
            "random": random_gain,
            "oracle": oracle_gain,
        },

        "headroom_capture_fraction": {
            "greedy_q1_value":
                capture(mean("greedy")),
            "h2_value":
                capture(mean("h2")),
            "mi_z":
                capture(mean("mi_z")),
        },

        "important_limitation":
            "Only q1 has external realized evaluation. "
            "q2 is fixed-bank simulated lookahead; "
            "this is not yet a real two-step MAS rollout.",
    }

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    with (out / "per_state.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for x in details:
            f.write(
                json.dumps(x, ensure_ascii=False)
                + "\n"
            )

    print("\n===== H2 VALUE DIAGNOSTIC =====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
