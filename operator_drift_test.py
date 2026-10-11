#!/usr/bin/env python3

# Reuse the exact data/model construction from the diagnosis script
exec(open("execution_operator_diagnosis.py").read().split(
    '# ============================================================\n# PART 1:'
)[0])

import numpy as np
import pandas as pd

print("=" * 80)
print("LOCAL OPERATOR vs COMPOSITIONAL DRIFT TEST")
print("=" * 80)

# Train specialists on contexts through parent depth 2.
# This is deliberately the strongest fixed transition model.
Rmodel = fit_r(TRAIN)
S = fit_specialists(TRAIN, 2)

rows_out = []

# ------------------------------------------------------------
# Test each transition at each depth in TWO ways:
#
# teacher-forced:
#   true parent -> learned next operator
#
# rollout:
#   predicted parent -> learned next operator
#
# If teacher-forced is good but rollout deteriorates:
#       COMPOSITIONAL DRIFT / ERROR AMPLIFICATION
#
# If teacher-forced itself is bad:
#       LOCAL OPERATOR / REPRESENTATION FAILURE
# ------------------------------------------------------------

for depth in [1, 2, 3]:

    seqs = PREFIX[depth]

    for seq in seqs:

        agent = seq[-1]
        parent_seq = seq[:-1]

        child_key = "R" + seq
        parent_key = "R" + parent_seq

        true_parent = data[parent_key][TEST]
        true_child = data[child_key][TEST]

        # -----------------------------
        # Teacher forced:
        # TRUE parent -> operator
        # -----------------------------

        teacher_pred = specialist_step(
            S[agent],
            true_parent,
            q[TEST]
        )

        teacher_err = np.mean(
            (teacher_pred - true_child) ** 2,
            axis=1
        )

        # -----------------------------
        # Rollout:
        # predicted parent -> operator
        # -----------------------------

        if len(parent_seq) == 0:

            rollout_parent = Rmodel.predict(
                q[TEST]
            )

        else:

            rollout_parent = predict_prefix(
                Rmodel,
                S,
                TEST,
                parent_seq
            )

        rollout_pred = specialist_step(
            S[agent],
            rollout_parent,
            q[TEST]
        )

        rollout_err = np.mean(
            (rollout_pred - true_child) ** 2,
            axis=1
        )

        # Error already present in input state
        parent_err = np.mean(
            (rollout_parent - true_parent) ** 2,
            axis=1
        )

        # How much worse prediction becomes because input was predicted
        excess = rollout_err - teacher_err

        amplification = (
            rollout_err.mean()
            /
            max(teacher_err.mean(), 1e-12)
        )

        rows_out.append({
            "depth": depth,
            "transition": child_key,
            "agent": agent,
            "parent_mse": parent_err.mean(),
            "teacher_forced_mse": teacher_err.mean(),
            "rollout_mse": rollout_err.mean(),
            "excess_rollout_mse": excess.mean(),
            "rollout_over_teacher": amplification,
        })


df = pd.DataFrame(rows_out)

print("\nPER TRANSITION")
print(
    df.round(5).to_string(index=False)
)

summary = (
    df.groupby("depth")
      .agg(
          parent_mse=("parent_mse", "mean"),
          teacher_forced_mse=("teacher_forced_mse", "mean"),
          rollout_mse=("rollout_mse", "mean"),
          excess_rollout_mse=("excess_rollout_mse", "mean"),
          rollout_over_teacher=("rollout_over_teacher", "mean"),
      )
      .reset_index()
)

print("\n" + "=" * 80)
print("DEPTH SUMMARY")
print("=" * 80)

print(
    summary.round(5).to_string(index=False)
)


# ============================================================
# Operator sensitivity:
#
# Perturb the TRUE parent by exactly the observed parent error:
#
# delta = predicted_parent - true_parent
#
# Compare ||F(h+delta)-F(h)|| / ||delta||
#
# This estimates empirical amplification of realistic errors.
# ============================================================

sens_rows = []

for depth in [1, 2, 3]:

    for seq in PREFIX[depth]:

        agent = seq[-1]
        parent_seq = seq[:-1]

        parent_key = "R" + parent_seq

        true_parent = data[parent_key][TEST]

        if len(parent_seq) == 0:

            pred_parent = Rmodel.predict(
                q[TEST]
            )

        else:

            pred_parent = predict_prefix(
                Rmodel,
                S,
                TEST,
                parent_seq
            )

        out_true_parent = specialist_step(
            S[agent],
            true_parent,
            q[TEST]
        )

        out_pred_parent = specialist_step(
            S[agent],
            pred_parent,
            q[TEST]
        )

        din = np.linalg.norm(
            pred_parent - true_parent,
            axis=1
        )

        dout = np.linalg.norm(
            out_pred_parent - out_true_parent,
            axis=1
        )

        valid = din > 1e-8

        gain = (
            dout[valid]
            /
            din[valid]
        )

        sens_rows.append({
            "depth": depth,
            "transition": "R" + seq,
            "agent": agent,
            "mean_empirical_gain": np.mean(gain),
            "median_empirical_gain": np.median(gain),
            "p90_empirical_gain": np.quantile(gain, .90),
            "max_empirical_gain": np.max(gain),
        })


sens = pd.DataFrame(sens_rows)

print("\n" + "=" * 80)
print("EMPIRICAL ERROR AMPLIFICATION")
print("=" * 80)

print(
    sens.round(5).to_string(index=False)
)

sens_summary = (
    sens.groupby("depth")
        .agg(
            mean_gain=("mean_empirical_gain", "mean"),
            median_gain=("median_empirical_gain", "mean"),
            p90_gain=("p90_empirical_gain", "mean"),
            max_gain=("max_empirical_gain", "max"),
        )
        .reset_index()
)

print("\nBY DEPTH")
print(
    sens_summary.round(5).to_string(index=False)
)


# ============================================================
# Extract linear h -> next-state blocks and inspect spectrum.
#
# feature = [h, q, mask, 1]
# Ridge coef shape = [output_dim, input_dim]
#
# First BASE_DIM columns are the state-transition block.
# ============================================================

spectral_rows = []

for agent in SPECIALISTS:

    W = S[agent].coef_[:, :BASE_DIM]

    eigvals = np.linalg.eigvals(W)

    singular = np.linalg.svd(
        W,
        compute_uv=False
    )

    spectral_rows.append({
        "agent": agent,
        "spectral_radius": np.max(np.abs(eigvals)),
        "operator_norm_sigma1": singular[0],
        "sigma_median": np.median(singular),
        "frobenius_norm": np.linalg.norm(W, "fro"),
    })


spectral = pd.DataFrame(
    spectral_rows
)

print("\n" + "=" * 80)
print("OPERATOR SPECTRAL / STABILITY DIAGNOSTIC")
print("=" * 80)

print(
    spectral.round(5).to_string(index=False)
)


# ============================================================
# Save
# ============================================================

OUT2 = Path(
    "artifacts/operator_drift_test"
)

OUT2.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUT2 / "transition_errors.csv",
    index=False
)

summary.to_csv(
    OUT2 / "depth_summary.csv",
    index=False
)

sens.to_csv(
    OUT2 / "empirical_amplification.csv",
    index=False
)

spectral.to_csv(
    OUT2 / "operator_spectrum.csv",
    index=False
)


print("\n" + "=" * 80)
print("DECISION GATE")
print("=" * 80)

for _, r in summary.iterrows():

    print(
        f"depth={int(r.depth)} "
        f"teacher={r.teacher_forced_mse:.4f} "
        f"rollout={r.rollout_mse:.4f} "
        f"ratio={r.rollout_over_teacher:.3f}x"
    )

print("""
Interpretation:

A) teacher-forced error LOW, rollout error grows strongly with depth
   -> operators locally work, but composition is unstable.
   -> next target: stability-constrained / contractive execution algebra.

B) teacher-forced error already HIGH and rollout/teacher ratio near 1
   -> composition is NOT the main problem.
   -> current semantic space does not make agent transitions sufficiently
      operator-like.
   -> next target: learn the execution representation / lifting.

C) both happen
   -> representation error + compositional amplification.
   -> quantify which dominates before changing architecture.

Spectral radius alone is NOT proof of stability.
Compare it with empirical perturbation gain and rollout growth.
""")
