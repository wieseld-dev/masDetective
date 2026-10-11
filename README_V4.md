# V4 — Live White-Box RL Environment

This is the first version that matches the intended research loop:

1. Start from benign seed probes.
2. A trainable Detective policy will eventually generate/modify the next benign probe.
3. The probe is sent to a **live MAS**.
4. The Detective sees only the external MAS output.
5. A white-box grader privately sees which internal worker was selected.
6. The policy receives only a scalar reward.

## Current reward

For the first live experiment, reward measures **new internal-worker coverage**:

```text
reward =
    normalized increase in unique-worker coverage
    - query cost
    + one-time completion bonus
```

The worker identity itself is not present in the policy observation.

This is deliberately simpler than using a K-estimator on day one. Once live coverage
reward behaves sensibly, add the black-box K-belief improvement term.

## Important cost fact

The current target MAS uses `gpt-4o-mini` by default and makes **two API calls per probe**:
one router call and one worker call.

So:

```text
5 probes = 10 model calls
100 RL episodes × 6 probes = 1,200 model calls
```

Do NOT start large RL training against this target yet.

## Local smoke test

```bash
source .venv/bin/activate
export OPENAI_API_KEY="..."
python -u -m scripts.smoke_live_whitebox --system-id C03_s1 --max-probes 5
```

## Cluster

For the API-only smoke test a GPU is unnecessary. The cluster becomes useful once the
Detective itself is a local trainable model.

Copy the project:

```bash
scp -r ~/Documents/Fujitsu/mas_rl_detective <BGU_USER>@slurm.bgu.ac.il:~/
```

SSH to the cluster, create a cluster-local environment, install requirements, export
`OPENAI_API_KEY`, and submit:

```bash
cd ~/mas_rl_detective
mkdir -p logs
sbatch cluster/smoke_live_whitebox.sbatch
```

View:

```bash
squeue --me
tail -f logs/mas-wb-smoke-<JOB_ID>.out
```

## Next stage

After the reward smoke test passes, attach a **trainable local Detective model** on the
cluster (LoRA + policy optimization). The target MAS can remain API-based; only the
Detective needs trainable weights for true RL.
