# V5 — Trainable Live MAS Detective

V5 upgrades the V4 live white-box smoke environment into a first **real trainable RL pipeline**.

## Threat-model boundary

The Detective policy can observe only:

- its own previous benign queries,
- external MAS outputs,
- remaining query budget,
- the K-belief produced by a black-box estimator from that public history.

Training-only white-box truth (`K_true`, worker IDs/roles, path nodes/edges) is consumed by the reward grader and debug logs only. It is never inserted into the policy prompt.

## What is included

- 24-system topology zoo: 12 train / 6 validation / 6 test.
- Flat, 2-level and 3-level live MAS targets.
- K counts **task/worker agents only**; hidden routers are excluded.
- Black-box K estimator trained from public query/output histories.
- Rich white-box scalar reward:
  - improvement in log P(K_true),
  - new-worker coverage,
  - new-edge coverage,
  - novel path reward / repeated-path penalty,
  - query and token costs.
- Hard benign-query gate.
- Qwen2.5-1.5B Detective with LoRA / optional NF4 QLoRA.
- Self-contained sequence-level clipped PPO update. It does not depend on TRL's PPO API.
- Slurm batch scripts so training continues after the laptop disconnects.

## Important: cost controls

Both expensive entry points are plan-only by default. They make API calls only with `--execute`.

### 1. Install V5 Python additions

Keep the PyTorch build that already passed your GTX 1080 Ti CUDA test, then:

```bash
pip install -r requirements_v5.txt
```

### 2. Verify GPU libraries

```bash
python -m scripts.check_gpu_stack
python -m scripts.check_4bit_load
```

The second command downloads/loads the 1.5B model and performs one local GPU generation. It makes no OpenAI API call.

If 4-bit fails on the GTX 1080 Ti, do not force it. V5 supports FP16 LoRA with `--no-4bit`.

### 3. Run tests

```bash
python -m pytest -q tests/test_whitebox_reward.py tests/test_v5_benign_gate.py tests/test_v5_k_estimator.py tests/test_v5_reward.py tests/test_v5_topology_zoo.py
```

### 4. Bootstrap the black-box K estimator

First inspect cost only:

```bash
python -m training.bootstrap_estimator --limit-systems 6 --max-seeds 5
```

Then execute only after the plan looks reasonable:

```bash
python -m training.bootstrap_estimator --limit-systems 6 --max-seeds 5 --execute
```

For a fuller first bootstrap, remove `--limit-systems 6` and use all 12 train systems.

### 5. PPO plan (no API calls)

```bash
python -m training.train_ppo --episodes 12 --budget 4
```

### 6. Real PPO training

Interactive smoke:

```bash
python -m training.train_ppo --episodes 2 --budget 2 --rollout-batch 2 --execute
```

For unattended Slurm training:

```bash
mkdir -p logs artifacts outputs
conda deactivate
sbatch cluster/train_detective.sbatch
```

Monitor:

```bash
squeue --me
tail -f logs/mas-detective-<JOB_ID>.out
```

You may close VS Code / disconnect your laptop after `sbatch` returns the job ID.

## Research-status warning

This is the first runnable RL training pipeline, not the final paper experiment. The 6-system estimator bootstrap is intentionally small and cheap. Before reporting scientific results, train the estimator on the full train split, freeze it, and evaluate the Detective on the held-out validation/test MASs without any white-box information at inference time.
