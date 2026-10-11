ROLE-DIRECT WITHIN-MAS V4 (2026-10-09)

OFFLINE continuation of EXISTING models, never from scratch:
- Qwen2.5-3B Sensor: continue prior final LoRA with BCE + pairwise-within-MAS ranking on V13.7 TRAIN, evaluate on V13.7 DEV. Old adapter stays unchanged.
- Qwen2.5-1.5B Generator: continue existing V14/V11 LoRA by 1 conservative SFT epoch from PRESENT examples with confirmed activation. NO new GPT API calls, and DOES NOT constitute a validated PRESENT/ABSENT improvement. Compare old vs new samples, fall back on surface regressions. 20 roles x 8 strategies.
- Selector: train lightweight query-only ridge ranker using observed PRESENT routing and old-sensor paired win proxy from 100 x 3 matched comparisons; group-disjoint pair holdout, auto-fallback if not stronger than random. NOT validated EIG.
- NEW Qwen3-8B sensor and Qwen3.5-9B generator are deferred until model weights/environment and reliable discriminative rewards are verified; their adapters cannot be directly transferred from Qwen2.5.

STRICT: no API key, no .env, no model download, no GPT calls, original adapters only read. All outputs under artifacts/role_direct_within_v4/runs/<run>.

Run:
  cd /home/wieseld/mas_rl_detective
  bash cluster/role_direct_within_v4/submit.sh

Stages: synchronous prepare; concurrent CPU selector, RTX PRO 6000 GPU sensor, RTX PRO 6000 GPU generator; dependent GPU A/B probe audit; afterany CPU report. --time=20:00:00 all jobs.

Monitor:
  squeue --me
  ROOT=$(find artifacts/role_direct_within_v4/runs -maxdepth 1 -type d -name 'v4_*' | sort | tail -1)
  cat "$ROOT/job_ids.tsv"
  sacct -j "$(cut -f1 "$ROOT/job_ids.tsv" | paste -sd, -)" --format=JobID,JobName,State,Elapsed,ExitCode
  tail -n 40 "$(ls -t logs/v4s-*.out | head -1)"

Results:
  python -m json.tool "$ROOT/reports/final.json"
  python -m json.tool "$ROOT/reports/sensor_selected.json"
  python -m json.tool "$ROOT/reports/selector.json"
  python -m json.tool "$ROOT/reports/generator_probe_audit.json"

Estimated actual runtime: 2-12h + Slurm queues; no guarantee on target cluster. Sensor up to 3 epochs, generator one SFT epoch, 320 offline generator completions. Time cap 20h per job.

LIMITS: No new target MAS executions, no true discriminator-trained GRPO/DPO, no proof improved PRESENT vs ABSENT. Development performance, not sealed independent test. Future stages require verified matched discrimination and sham controls.
