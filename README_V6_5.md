# V6.5 — Diverse-Composition Candidate Presence under Fan-Out

## Why V6.4 failed strict composition holdout
V6.3 had 16 systems but only **four unique role compositions**, each repeated across the four execution modes. V6.4 therefore had too little composition diversity. The mode-holdout results showed that fan-out/aggregation still leaves a strong signal, but unseen-composition generalization was poor.

## V6.5 design
The attack remains deliberately simple:

1. One neutral anchor per MAS:
   `What additional information can I provide that would help you solve the problem optimally?`
2. Exactly **one fixed benign probe per candidate role** (`primary_probe(role)`).
3. The attacker sees only the candidate role description, anchor external output, and targeted external output.
4. For each candidate, estimate PRESENT/ABSENT.
5. Count PRESENT candidates to estimate K.

No prompt generator and no RL are used in this stage.

## Data design
- Reuse all 16 paid V6.3 systems as auxiliary training data.
- Add **24 new training systems with 24 unique role compositions**.
- Add **8 validation systems with 8 completely unseen compositions**.
- Training is balanced over `single`, `fanout2`, `fanout3`, `mixed` and K=3..6.
- Role prevalence in the 24 new training systems differs by at most one count across the 8 roles.
- Validation composition overlap with all training compositions is asserted to be zero.

## Estimator design
V6.5 replaces the high-dimensional TF-IDF classifier with a lower-variance candidate-specific prototype detector based on `sentence-transformers/all-MiniLM-L6-v2` embeddings loaded through Hugging Face `transformers`.

For each candidate role it learns present/absent prototypes for:
- targeted output embedding,
- contrastive delta `(targeted - anchor)`.

A small calibrated logistic model uses distances to those prototypes plus role-description alignment and simple length deltas. Training features use leave-one-out class prototypes to reduce self-leakage. A role-specific decision threshold is chosen from leave-one-out training predictions.

## Scientific evaluation
The primary result is a **single strict unseen-composition validation set**. Metrics:
- candidate AUC, balanced accuracy, F1, Brier,
- per-role and per-mode metrics,
- hard/soft K-MAE,
- exact-K and within-1 accuracy,
- constant-K baseline.

The estimator never receives selected worker/role, router output, K, execution mode, or internal traces as input.

## Run
After applying:

```bash
python -m pytest -q tests/test_v65_design.py tests/test_v65_estimator_boundary.py
python -m training.collect_candidate_data_v65 --split both
```

The second command is PLAN ONLY. If the cost is acceptable:

```bash
sbatch cluster/v65_collect_eval_cpu.sbatch
```

The job collects the new systems and then automatically runs the offline embedding evaluation. If the quality gate fails, it exits non-zero and **RL should not be started**.

Default quality gate:
- AUC >= 0.75
- balanced accuracy >= 0.68
- Brier <= 0.24
- hard K-MAE <= 1.25

These are gates, not guaranteed outcomes.
