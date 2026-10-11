# V9-FULL

This patch scales the V9 objective that already produced positive holdout-specific separation, without changing the core reward.

Key changes:
- 6 rounds per role × 6 candidates = 288 candidate probes.
- 20-ish V6.5 training systems for RL, a small detector-reserved subset, and the composition-disjoint V6.5 validation split for policy holdout.
- Three held-out generations per role (reported individually and by per-role mean/best SPEC).
- Local syntax retries before the semantic gate. These retries cost GPU generation only, not target API calls, and prevent malformed generations from wasting an entire candidate slot.
- Stronger user-perspective prompt constraints to avoid reversed-speaker probes.
- Elite imitation weight 0.10 and LR 2e-5; the reward itself is unchanged.
- Adapter + optimizer checkpoints after every GRPO update and a resume sbatch script.
- Existing response caches are persistent.

Plan-only:
`python -m training.train_seeded_probe_discovery_v9full`

Run:
`sbatch cluster/v9_full_gpu.sbatch`

Resume after an interrupted job:
`sbatch cluster/v9_full_resume_gpu.sbatch`

Live progress:
`tail -f "$(ls -t logs/v9-full-*.out | head -1)"`

Final diagnostic:
`PYTHONPATH="$PWD" python scripts/diagnose_v9full.py --log artifacts/v9_full_probe_discovery.jsonl`

The planned upper bound is 288 × 2 matched pairs × 3 branches = 1,728 candidate MAS interactions, plus anchor/holdout work. Blocked probes never reach the MAS, and cached identical interactions are reused.
