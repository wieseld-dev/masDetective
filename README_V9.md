# V9 — Seeded Counterfactual Probe Discovery

V9 deliberately fixes the core failure revealed by V8:

- V8 mostly learned to *route to the candidate role*; target activation dominated reward.
- V9 makes activation an eligibility constraint only.
- V9 optimizes robust, target-specific **external output separation across multiple matched counterfactual systems**.
- The human-designed `primary_probe(role)` is used as ONE anchor/example. The policy must invent a different standalone probe; it does not choose from a bank.
- Every candidate is scored across a bundle of matched PRESENT / ABSENT / SHAM systems. This reduces single-system reward noise.
- SHAM subtraction penalizes generic perturbations that are not specific to removing the target capability.
- Optimization is GRPO plus a small elite-imitation term, so successful discoveries are actually absorbed by the small Qwen policy.
- A held-out base-system evaluation is included in the same batch job; no separate smoke/eval cycle is required.

## Apply

```bash
cd ~/mas_rl_detective
rm -rf /tmp/v9
mkdir -p /tmp/v9
unzip -o ~/mas_rl_detective_v9_seeded_probe_discovery_patch.zip -d /tmp/v9
bash /tmp/v9/v9_seeded_counterfactual_probe_discovery_patch/apply_v9.sh ~/mas_rl_detective
conda activate masdetective
python -m pytest -q tests/test_v9_reward.py tests/test_v9_policy_prompt.py tests/test_v8_reward.py tests/test_v8_query.py
```

## Plan only

```bash
python -m training.train_seeded_probe_discovery_v9 --rounds-per-role 2 --group-size 5 --bundle-size 2
```

## One real run

```bash
sbatch cluster/v9_one_shot_gpu.sbatch
```

Monitor:

```bash
squeue --me
sacct -u wieseld --starttime today --format=JobID,JobName,Partition,State,ExitCode,Elapsed | tail -15
```

After completion:

```bash
tail -n 240 "$(ls -t logs/v9-discovery-*.out | head -1)"
PYTHONPATH="$PWD" python scripts/diagnose_v9.py --log artifacts/v9_probe_discovery.jsonl
```

Interpretation: the key metric is **specific_separation**, not activation. A good discovered probe should have `ACT >= 0.75` and `SPEC > 0` on matched bundles. The end-of-job HOLDOUT block tells you whether freshly generated probes retain positive target-specific separation on base systems that were not used for RL.
