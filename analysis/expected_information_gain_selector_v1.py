#!/usr/bin/env python3
"""
expected_information_gain_selector_v1.py

Proper next-stage selector diagnostic.

Key correction versus posterior_observation_selector_v1:
DO NOT average P(Y|Z,q) over Z first and then score the mean.
That destroys hypothesis-disagreement information.

Instead retain P(Y|Z,q) for every posterior particle and score each probe by
approximate mutual information:

  MI_K(q) ~= sum_i [ h(E[Y_i|q]) - sum_k P(k) h(E[Y_i|K=k,q]) ]

  MI_ID(q) ~= sum_k P(k) sum_i [
                  h(E[Y_i|K=k,q])
                  - E_{Z|K=k} h(P(Y_i=1|Z,q))
              ]

  MI_Z(q) = MI_K(q) + MI_ID(q)

This is a mean-field / per-output-bit approximation to:
  I(K;Y | q,D_t)
  I(Z;Y | K,q,D_t)
  I(Z;Y | q,D_t)

No ground-truth Z is used in candidate scoring.
Hidden selected_roles are used ONLY for retrospective evaluation.

Uses the trained observation_model_v1 checkpoint and its exact held-out split.
"""

from __future__ import annotations
import argparse, json, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_obs_module():
    sys.path.insert(0, str(Path("analysis").resolve()))
    import observation_model_v1 as om
    return om


def suffix_esym(weights, kmax):
    n = len(weights)
    E = np.zeros((n + 1, kmax + 1), dtype=np.float64)
    E[n, 0] = 1.0
    for i in range(n - 1, -1, -1):
        E[i, 0] = 1.0
        for k in range(1, kmax + 1):
            E[i, k] = E[i + 1, k] + weights[i] * E[i + 1, k - 1]
    return E


def sample_fixed_k(weights, k, rng, E):
    n = len(weights)
    z = np.zeros(n, dtype=np.float32)
    rem = int(k)
    for i in range(n):
        if rem == 0:
            break
        if n - i == rem:
            z[i:] = 1.0
            break
        denom = E[i, rem]
        if denom <= 0:
            ids = rng.choice(np.arange(i, n), size=rem, replace=False)
            z[ids] = 1.0
            break
        pin = float(np.clip(weights[i] * E[i + 1, rem - 1] / denom, 0, 1))
        if rng.random() < pin:
            z[i] = 1.0
            rem -= 1
    return z


def stratified_particles(k_probs, role_probs, role_order, per_k, rng):
    """
    Draw the same number of particles within each supported K, but assign
    weights p(K)/per_k. This preserves the supplied K posterior exactly.
    """
    p = np.array([float(role_probs[r]) for r in role_order], dtype=np.float64)
    p = np.clip(p, 1e-5, 1 - 1e-5)
    odds = p / (1 - p)

    kp = [(int(k), float(v)) for k, v in k_probs.items() if float(v) > 0]
    zsum = sum(v for _, v in kp)
    kp = [(k, v / zsum) for k, v in kp]

    E = suffix_esym(odds, max(k for k, _ in kp))

    Z, W, K = [], [], []
    for k, pk in kp:
        for _ in range(per_k):
            Z.append(sample_fixed_k(odds, k, rng, E))
            W.append(pk / per_k)
            K.append(k)
    return (
        np.stack(Z).astype(np.float32),
        np.asarray(W, dtype=np.float64),
        np.asarray(K, dtype=np.int64),
        dict(kp),
    )


def hbin(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1 - 1e-8)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def safe_spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    if spearmanr is not None:
        v = spearmanr(a, b).correlation
        return None if np.isnan(v) else float(v)
    ra = np.empty(len(a)); rb = np.empty(len(b))
    ra[np.argsort(a)] = np.arange(len(a))
    rb[np.argsort(b)] = np.arange(len(b))
    v = np.corrcoef(ra, rb)[0, 1]
    return None if np.isnan(v) else float(v)


@torch.no_grad()
def particle_predictions(model, qemb_group, belief, kvec, round_norm,
                         particles, device, batch_size):
    C, M = qemb_group.shape[0], particles.shape[0]
    N = particles.shape[1]

    q = np.repeat(qemb_group[:, None, :], M, axis=1).reshape(C * M, -1)
    z = np.repeat(particles[None, :, :], C, axis=0).reshape(C * M, N)
    b = np.repeat(belief[None, :], C * M, axis=0)
    kv = np.repeat(kvec[None, :], C * M, axis=0)
    rr = np.full((C * M, 1), round_norm, dtype=np.float32)

    out = []
    for s in range(0, C * M, batch_size):
        e = min(C * M, s + batch_size)
        logits = model(
            torch.from_numpy(q[s:e]).to(device),
            torch.from_numpy(z[s:e]).to(device),
            torch.from_numpy(b[s:e]).to(device),
            torch.from_numpy(kv[s:e]).to(device),
            torch.from_numpy(rr[s:e]).to(device),
        )
        out.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(out, axis=0).reshape(C, M, N).astype(np.float64)


def mi_scores(P, weights, klabels, kprior):
    """
    P: [C, M, R], per candidate/particle/role Bernoulli exposure probabilities.

    Returns arrays per candidate:
      mi_k       approximate information about count K
      mi_id      approximate information about identity within fixed K
      mi_z       total approximate information about hidden map
      pred_exp   posterior expected number of exposed roles
    """
    C, M, R = P.shape
    w = weights / weights.sum()

    rbar = np.einsum("m,cmr->cr", w, P)
    H_marg = hbin(rbar).sum(axis=1)

    # E_Z H(Y | Z,q), factorized across output roles
    H_cond_z = np.einsum("m,cmr->c", w, hbin(P))

    # H(Y | K,q), using the Bernoulli marginal within each K.
    H_cond_k = np.zeros(C, dtype=np.float64)
    H_within_k_z = np.zeros(C, dtype=np.float64)

    for k, pk in kprior.items():
        mask = (klabels == int(k))
        if not np.any(mask):
            continue
        wk = weights[mask]
        wk = wk / wk.sum()
        Pk = P[:, mask, :]
        rbar_k = np.einsum("m,cmr->cr", wk, Pk)

        H_k = hbin(rbar_k).sum(axis=1)
        H_z_given_k = np.einsum("m,cmr->c", wk, hbin(Pk))

        H_cond_k += pk * H_k
        H_within_k_z += pk * H_z_given_k

    mi_k = H_marg - H_cond_k
    mi_id = H_cond_k - H_cond_z
    mi_z = H_marg - H_cond_z

    # numerical cleanup
    mi_k = np.maximum(mi_k, 0)
    mi_id = np.maximum(mi_id, 0)
    mi_z = np.maximum(mi_z, 0)

    pred_exp = rbar.sum(axis=1)
    return mi_k, mi_id, mi_z, pred_exp


def summarize(stats):
    def m(k):
        v = stats.get(k, [])
        return float(np.mean(v)) if v else None
    return {
        "groups": len(stats.get("oracle_reward", [])),
        "mean_oracle_reward": m("oracle_reward"),
        "mean_selected_reward": m("selected_reward"),
        "reward_optimal_hit_rate": m("reward_optimal"),
        "exact_oracle_candidate_match_rate": m("exact_match"),
        "mean_regret": m("regret"),
        "mean_within_group_spearman": m("spearman"),
        "spearman_groups": len(stats.get("spearman", [])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/adaptive_v11_curriculum_v2")
    ap.add_argument("--obs-dir", default="artifacts/observation_model_v1")
    ap.add_argument("--out", default="artifacts/eig_selector_v1")
    ap.add_argument("--particles-per-k", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--hybrid-alpha", type=float, default=0.7,
                    help="hybrid = alpha*MI_K + (1-alpha)*MI_ID")
    args = ap.parse_args()

    seed_all(args.seed)
    om = load_obs_module()

    obs_dir = Path(args.obs_dir)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(obs_dir / "model.pt", map_location="cpu")
    roles = ckpt["role_order"]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    model = om.ObservationNet(
        qdim=ckpt["query_embedding_dim"],
        n_roles=len(roles),
        hidden=ckpt["hidden"],
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    examples, roles2, _, _ = om.collect_examples(Path(args.root))
    if roles2 != roles:
        raise RuntimeError("role order mismatch")

    split = json.loads((obs_dir / "split_systems.json").read_text())
    test_systems = set(split["test"])
    test_idx = [i for i, x in enumerate(examples) if x["system_id"] in test_systems]

    qemb = np.load(obs_dir / "query_embeddings.npy")
    if len(qemb) != len(examples):
        raise RuntimeError("query embedding/example count mismatch")

    max_round = max(x["round"] for x in examples)

    traj = defaultdict(lambda: defaultdict(list))
    for i in test_idx:
        x = examples[i]
        traj[(x["parent"], x["system_id"])][x["round"]].append((i, x))

    methods = ["mi_k", "mi_identity", "mi_z", "hybrid", "exposure"]
    stats = {m: defaultdict(list) for m in methods}
    stats_r1 = {m: defaultdict(list) for m in methods}
    base = defaultdict(list); base_r1 = defaultdict(list)
    per_group = []

    total = sum(len(rm) for rm in traj.values())
    done = 0

    for (parent, sid), rm in traj.items():
        seen_actual = set()

        for rnd in sorted(rm):
            done += 1
            rows = sorted(rm[rnd], key=lambda x: x[1]["candidate_index"])
            if len(rows) < 2:
                actual = [x for _, x in rows if x["chosen"]]
                if len(actual) == 1:
                    seen_actual.update(actual[0]["selected_roles"])
                continue

            x0 = rows[0][1]
            belief = np.array([x0["role_probs"][r] for r in roles], dtype=np.float32)
            kvec = np.array([x0["k_probs"].get(str(k), 0.0) for k in [3,4,5,6]], dtype=np.float32)
            round_norm = float(rnd / max(max_round, 1))

            # deterministic state-specific particles
            state_seed = abs(hash((parent, sid, rnd, args.seed))) % (2**32)
            rng = np.random.default_rng(state_seed)
            Z, W, Klabels, Kprior = stratified_particles(
                x0["k_probs"], x0["role_probs"], roles, args.particles_per_k, rng
            )

            qg = np.stack([qemb[i] for i, _ in rows]).astype(np.float32)
            P = particle_predictions(
                model, qg, belief, kvec, round_norm, Z, device, args.batch_size
            )

            mi_k, mi_id, mi_z, exp = mi_scores(P, W, Klabels, Kprior)
            hybrid = args.hybrid_alpha * mi_k + (1 - args.hybrid_alpha) * mi_id

            cands = []
            for j, (idx, x) in enumerate(rows):
                reward = len(set(x["selected_roles"]) - seen_actual)
                cands.append({
                    "candidate_index": x["candidate_index"],
                    "query": x["query"],
                    "oracle_new_hidden_eval_only": reward,
                    "selected_roles_hidden_eval_only": list(x["selected_roles"]),
                    "chosen_actual": x["chosen"],
                    "existing_selector_score": x["selector_predicted_utility"],
                    "scores": {
                        "mi_k": float(mi_k[j]),
                        "mi_identity": float(mi_id[j]),
                        "mi_z": float(mi_z[j]),
                        "hybrid": float(hybrid[j]),
                        "exposure": float(exp[j]),
                    },
                })

            oracle = max(
                cands,
                key=lambda c: (
                    c["oracle_new_hidden_eval_only"],
                    len(c["selected_roles_hidden_eval_only"]),
                    -c["candidate_index"],
                ),
            )
            oreward = oracle["oracle_new_hidden_eval_only"]

            grec = {
                "parent": parent,
                "system_id": sid,
                "round": rnd,
                "oracle_reward_hidden_eval_only": oreward,
                "oracle_candidate_hidden_eval_only": oracle["candidate_index"],
                "seen_actual_hidden_eval_only": sorted(seen_actual),
                "candidates": cands,
            }

            for method in methods:
                ch = max(cands, key=lambda c: (c["scores"][method], -c["candidate_index"]))
                reward = ch["oracle_new_hidden_eval_only"]
                rho = safe_spearman(
                    [c["scores"][method] for c in cands],
                    [c["oracle_new_hidden_eval_only"] for c in cands],
                )

                for block in [stats[method]] + ([stats_r1[method]] if rnd == 1 else []):
                    block["oracle_reward"].append(oreward)
                    block["selected_reward"].append(reward)
                    block["reward_optimal"].append(int(reward == oreward))
                    block["exact_match"].append(int(ch["candidate_index"] == oracle["candidate_index"]))
                    block["regret"].append(oreward - reward)
                    if rho is not None:
                        block["spearman"].append(rho)

                grec[f"{method}_candidate"] = ch["candidate_index"]
                grec[f"{method}_reward"] = reward
                grec[f"{method}_spearman"] = rho

            rnd_exp = float(np.mean([c["oracle_new_hidden_eval_only"] for c in cands]))
            base["random_reward"].append(rnd_exp)
            if rnd == 1:
                base_r1["random_reward"].append(rnd_exp)

            old_rows = [c for c in cands if c["existing_selector_score"] is not None]
            if len(old_rows) == len(cands):
                old = max(old_rows, key=lambda c: (float(c["existing_selector_score"]), -c["candidate_index"]))
                base["existing_reward"].append(old["oracle_new_hidden_eval_only"])
                if rnd == 1:
                    base_r1["existing_reward"].append(old["oracle_new_hidden_eval_only"])

            actual = [c for c in cands if c["chosen_actual"]]
            if len(actual) == 1:
                base["actual_reward"].append(actual[0]["oracle_new_hidden_eval_only"])
                if rnd == 1:
                    base_r1["actual_reward"].append(actual[0]["oracle_new_hidden_eval_only"])
                seen_actual.update(actual[0]["selected_roles_hidden_eval_only"])

            per_group.append(grec)
            if done % 100 == 0:
                print(f"processed {done}/{total} groups")

    def base_summary(b):
        return {
            "random_mean_reward": float(np.mean(b["random_reward"])) if b["random_reward"] else None,
            "existing_selector_mean_reward": float(np.mean(b["existing_reward"])) if b["existing_reward"] else None,
            "actual_chosen_mean_reward": float(np.mean(b["actual_reward"])) if b["actual_reward"] else None,
        }

    summary = {
        "setting": {
            "true_Z_used_for_scoring": False,
            "selected_roles_used_only_for_evaluation": True,
            "particles_per_K": args.particles_per_k,
            "hybrid_alpha": args.hybrid_alpha,
            "test_systems": len(test_systems),
        },
        "round_1_clean": {
            "methods": {m: summarize(stats_r1[m]) for m in methods},
            "baselines": base_summary(base_r1),
        },
        "all_rounds_one_step_conditional_on_actual_history": {
            "methods": {m: summarize(stats[m]) for m in methods},
            "baselines": base_summary(base),
        },
        "interpretation": {
            "mi_k": "Approximate I(K;Y|q,D) using per-role Bernoulli marginals.",
            "mi_identity": "Approximate I(Z;Y|K,q,D), identity information after conditioning on count.",
            "mi_z": "Approximate total I(Z;Y|q,D) = MI_K + MI_identity under the factorized approximation.",
            "critical_difference": "Unlike the previous selector, this preserves per-particle disagreement instead of averaging P(Y|Z,q) before scoring.",
        },
    }

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out / "per_group.jsonl").open("w", encoding="utf-8") as f:
        for x in per_group:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print("\n" + "=" * 100)
    print("EXPECTED INFORMATION GAIN SELECTOR V1")
    print("=" * 100)
    for bn in ["round_1_clean", "all_rounds_one_step_conditional_on_actual_history"]:
        print("\n" + bn)
        print("BASELINES:", summary[bn]["baselines"])
        for m in methods:
            x = summary[bn]["methods"][m]
            print(
                f"{m:12s} reward={x['mean_selected_reward']} "
                f"opt={x['reward_optimal_hit_rate']} "
                f"regret={x['mean_regret']} "
                f"rho={x['mean_within_group_spearman']}"
            )
    print("\n[saved]", out / "summary.json")


if __name__ == "__main__":
    main()
