# V6.3 — Single Router + Dynamic Fan-Out

This version deliberately keeps **one routing layer only** and trains across four execution modes:

- `single`: router -> exactly 1 worker
- `fanout2`: router -> exactly 2 workers -> hidden aggregation
- `fanout3`: router -> exactly 3 workers -> hidden aggregation
- `mixed`: router -> dynamically 1..3 workers -> hidden aggregation

There are 16 training MASs: every execution mode is paired with K=3,4,5,6. Validation/test use held-out role compositions.

## Why V6.3 changes the learning pipeline

The earlier fixed-role PPO was cleaner than free-text probing, but the reward was globally averaged over 8 roles and therefore weak for the selected action. V6.3 adds an action-specific Brier improvement while retaining global reconstruction and a small K term:

`reward = 1.0*focus_Brier_improvement + 1.5*global_Brier_improvement + 0.15*K_improvement - 0.01`

The role estimator is also rebuilt for fan-out outputs. It conditions on the chosen fixed probe identity plus **external output only** (not query wording), uses TF-IDF + regularized logistic classifiers, and runs system-level GroupKFold CV before RL. The supplied pipeline refuses to start the GPU RL job if this estimator quality gate fails.

The PPO rollout is more stable than V6.2:

- 48 balanced episodes: every 16-episode cycle visits all 16 training MASs once.
- budget=4 role probes after the anchor.
- 8 episodes / PPO update for less noisy advantages.
- 2 PPO epochs, clip=0.20, entropy bonus=0.01.
- memory-safe per-transition backward accumulation so the larger rollout does not retain many Qwen graphs on the GTX 1080 Ti.

## Apply

```bash
unzip -o ~/mas_rl_detective_v6_3_single_router_fanout_patch.zip -d /tmp/v63
bash /tmp/v63/v6_3_single_router_fanout_patch/apply_v6_3.sh ~/mas_rl_detective
conda activate masdetective
python -m pytest -q
```

## Inspect plan (no API calls)

```bash
python -m training.bootstrap_role_estimator_v63
python -m training.train_role_ppo_v63
```

## Submit unattended pipeline

```bash
bash submit_v6_3_pipeline.sh
```

This submits a CPU bootstrap/CV job and a dependent GPU PPO job. The PPO job starts only if bootstrap + CV succeeds.

## Held-out evaluation after training

Plan only:

```bash
python -m evaluation.eval_v63 --policy qwen --adapter outputs/detective_v6_3_fanout48 --split val
```

Actual evaluation requires a GPU and `--execute`.

No run is guaranteed to produce a scientific improvement; V6.3 is designed to remove the main avoidable failure modes observed in V6/V6.2 and to make failures diagnosable.
