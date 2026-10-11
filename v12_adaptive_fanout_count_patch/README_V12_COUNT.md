# V12 adaptive FANOUT worker-count development test

This freezes `artifacts/v11_policy` and adds an end-to-end black-box presence/count layer.

Important distinction: the V11 RL weights are **not updated on the test systems**. Test outputs only update an external sequential belief state. Otherwise ground-truth test roles/K would contaminate the model. The controller spends more benign probes on high-uncertainty candidate roles until confidence is high or the query budget is exhausted.

Detector training is offline from `phase=train` records in `artifacts/v11_probe_discovery.jsonl`. It uses only public `(candidate description, probe, final output)` as features. PRESENT/ABSENT/SHAM branch identity is used only as an offline label. The fresh FANOUT session never sees true roles, K, execution mode, router choices, traces, or branch identity.

Default test grid: 8 fresh composition-disjoint systems = `fanout2` and `fanout3` x K in {3,4,5,6}. Candidate universe is the 20 V11 training roles. Primary `K` is the number of specialist workers. In this synthetic runtime total hidden nodes are `K+2` because there is one router and one aggregator.

Run plan only:

```bash
cd ~/mas_rl_detective
PYTHONPATH="$PWD" ~/.conda/envs/masdetective/bin/python analysis/adaptive_fanout_count_v12.py
```

Launch:

```bash
sbatch cluster/v12_adaptive_fanout_count_gpu.sbatch
```

Monitor:

```bash
squeue --me
tail -f "$(ls -t logs/v12-fanout-count-*.out | head -1)"
```

Outputs:

- `artifacts/v12_fanout_count/summary.json`
- `artifacts/v12_fanout_count/sessions.jsonl`
- `artifacts/v12_fanout_count/detector_oof.json`
- `artifacts/v12_fanout_count/target_public_cache.jsonl`
- `artifacts/v12_fanout_count/test_manifest_eval_only.json`

The public response cache makes API work reusable after interruption. Probe generation is seeded deterministically per system/role/probe index, and completed sessions are skipped on resume when the run configuration hash matches.

This is a **development** end-to-end count test, not the final untouched paper test. After the detector/controller are frozen, build a new untouched role/composition set for the final claim.
