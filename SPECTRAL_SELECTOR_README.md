# Spectral Probe Selector POC

## What this tests

This POC tests a more careful version of the idea:

1. Current posterior over hidden agents -> covariance `Sigma`.
2. `Sigma` -> top spectral ambiguity modes `(lambda_k, v_k)`.
3. Learn a model that predicts which agents a candidate query is likely to diagnose.
4. Score a candidate by how strongly its predicted diagnostic sensitivity aligns with the unresolved spectral subspace:

   score(q) = sum_k lambda_k * (v_k^T a_hat(q))^2

5. Penalize candidates for which the learned sensitivity predictor is uncertain.

The sentence embedding is *not* used as the selector geometry.
It is only an input to a supervised model trained from historical probe outcomes.

## Why this version

A raw eigenvector over semantic embeddings is too weak: semantic similarity is not the same thing as MAS diagnostic value.

This version learns:

    (belief, query) -> expected per-agent diagnostic effect

and then applies the spectral criterion only in agent-belief space.

## Minimum JSONL requirements

For every state:
- `belief`: vector of N agent probabilities
- `candidates`: list

For every candidate:
- `query`
- either:
  - `belief_after`, or
  - `posterior_after`, or
  - `agent_sensitivity`

For oracle evaluation, add either:
- candidate `oracle_gain`, or
- state `true_z` together with candidate `belief_after`

## Smoke test

```bash
python spectral_probe_selector.py \
  --synthetic-smoke \
  --out artifacts/spectral_selector_smoke
```

## Real run

```bash
python spectral_probe_selector.py \
  --train artifacts/train.jsonl \
  --eval artifacts/eval.jsonl \
  --out artifacts/spectral_selector_v1 \
  --top-r 4 \
  --device cpu
```

## Outputs

- `selector.pt`
- `eval_detailed.jsonl`
- `summary.json`

Important metrics:
- `mean_state_spearman`
- `mean_chosen_oracle_gain`
- `mean_oracle_best_gain`
- `mean_regret`
- `top1_oracle_hit_rate`
- `mean_random_oracle_gain`
- `mean_current_selector_oracle_gain`

## Interpretation

The first question is not whether this beats the current full system.

The first question is:

> Given an already-good candidate pool, does the spectral score rank candidates in the same direction as oracle next-step gain?

If Spearman is positive and regret falls materially versus the current selector, then the idea is worth integrating into the generator/RL loop.
