MAS5 V4.7.4 crash/resume hotfix

Fix 1: catch invalid/padded token IDs on decode and retry only the current sample.
Fix 2: partial checkpoint is role-order prefix, not complete approved bank; merge 102 immutable approved with partial and previous candidate.
Fix 3: salvage check permits later approved role slots to be absent from prefix, but forbids conflicts.

Install from project root: unzip -o mas5_v474_resume_fix.zip
Run: export V46_ROOT="$PWD/artifacts/role_direct_within_v4/overnight_v46/mas5_night_20261009_224735_1879051"
     bash cluster/role_direct_within_v4/v473_submit.sh
Check: squeue --me ; tail -n 50 $(ls -t logs/v473-mas5-v473-regen-*.out | head -1)

GPU/Slurm job keeps 20-hour cap. No MAS API calls.
Do NOT delete approved_102.jsonl or v473_progress.partial.jsonl.
After successful generation, inspect output before downstream experiments.
