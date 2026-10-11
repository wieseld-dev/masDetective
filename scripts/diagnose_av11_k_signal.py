from __future__ import annotations

import json
import numpy as np

from adaptive_v11_active.belief import (
    FrozenBeliefBundle,
    systems_from_replay_rows,
    temperature_scale_distribution,
)
from adaptive_v11_active.io import read_jsonl
from adaptive_v11_active.reward import map_k
from v11.role_universe import TRAIN_ROLES


BANK = "artifacts/adaptive_v1/replay_bank.jsonl"
BELIEF = "artifacts/adaptive_v11_active_v1/frozen_belief.joblib"


rows = read_jsonl(BANK)
dev = systems_from_replay_rows(rows, TRAIN_ROLES, split="dev")
bundle = FrozenBeliefBundle.load(BELIEF)

print(f"belief_version={bundle.VERSION}")
print(f"dev_systems={len(dev)}")
print(f"fusion_alpha={bundle.fusion_alpha}")
print(f"roles_temperature={bundle.roles_temperature}")
print()

results = []

for ds in dev:
    history = ds.observations

    # Role probabilities BEFORE the K-consistency projection.
    raw = bundle._raw_role_vector_with(bundle.role_model, history)
    pcal = bundle._apply_platt(bundle.role_platt, raw)

    # K implied only by role probabilities.
    p_roles = temperature_scale_distribution(
        bundle._role_k_probs(pcal),
        bundle.roles_temperature,
    )
    kp_roles = {
        int(k): float(p_roles[i])
        for i, k in enumerate(bundle.ks)
    }

    # Final full bundle prediction.
    kp_final, rp_final = bundle.predict(history)

    ordered = sorted(
        zip(bundle.roles, pcal),
        key=lambda x: x[1],
        reverse=True,
    )
    s = np.sort(np.asarray(pcal))[::-1]

    def gap(k):
        if k <= 0 or k >= len(s):
            return float("nan")
        return float(s[k - 1] - s[k])

    rec = {
        "system_id": ds.system_id,
        "true_k": ds.k,
        "role_mass_pre_projection": float(np.sum(pcal)),
        "mean_role_p": float(np.mean(pcal)),
        "top1": float(s[0]),
        "top3_mass": float(np.sum(s[:3])),
        "top4_mass": float(np.sum(s[:4])),
        "top5_mass": float(np.sum(s[:5])),
        "top6_mass": float(np.sum(s[:6])),
        "gap_3_4": gap(3),
        "gap_4_5": gap(4),
        "gap_5_6": gap(5),
        "role_only_pred_k": int(map_k(kp_roles)),
        "final_pred_k": int(map_k(kp_final)),
        "pK_role_only": kp_roles,
        "pK_final": {int(k): float(v) for k, v in kp_final.items()},
        "top_roles": [
            [str(r), round(float(p), 4)]
            for r, p in ordered[:8]
        ],
    }
    results.append(rec)

    print("=" * 90)
    print(f"SYSTEM: {ds.system_id}")
    print(f"TRUE K: {ds.k}")
    print(f"ROLE MASS pre-projection: {rec['role_mass_pre_projection']:.4f}")
    print(
        "BOUNDARY GAPS: "
        f"3|4={rec['gap_3_4']:.4f} "
        f"4|5={rec['gap_4_5']:.4f} "
        f"5|6={rec['gap_5_6']:.4f}"
    )
    print(f"ROLE-ONLY K: {rec['role_only_pred_k']}  probs={kp_roles}")
    print(f"FINAL K:     {rec['final_pred_k']}  probs={kp_final}")
    print("TOP ROLES:")
    for role, p in ordered[:8]:
        mark = "*" if role in set(ds.truth_roles) else " "
        print(f"  {mark} {role:28s} {p:.4f}")

print()
print("=" * 90)
print("SUMMARY BY TRUE K")

for k in bundle.ks:
    rr = [r for r in results if r["true_k"] == k]
    masses = [r["role_mass_pre_projection"] for r in rr]
    print(
        f"K={k}: n={len(rr)} "
        f"role_mass_mean={np.mean(masses):.4f} "
        f"values={[round(x,4) for x in masses]}"
    )

true_k = np.asarray([r["true_k"] for r in results], dtype=float)
mass = np.asarray(
    [r["role_mass_pre_projection"] for r in results],
    dtype=float,
)

if len(results) > 1 and np.std(mass) > 0:
    corr = float(np.corrcoef(true_k, mass)[0, 1])
else:
    corr = float("nan")

print()
print(f"Pearson corr(true_K, role_mass) = {corr:.4f}")

with open(
    "artifacts/adaptive_v11_active_v1/k_signal_diagnostic_v2.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(results, f, indent=2)

print()
print(
    "[done] artifacts/adaptive_v11_active_v1/"
    "k_signal_diagnostic_v2.json"
)
print("ZERO API CALLS.")
