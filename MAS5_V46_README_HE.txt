MAS5 V4.6 — HMHY Generator SFT + evidence-based DPO + Sensor + Selector + MAS5 evaluation

What changed vs V4.5
1) Bootstrap: V4 trained Qwen2.5-1.5B Generator LoRA -> continue SFT using real FULL Help Me Help You questions from V4.3. Not a prompt-only wrapper: the Generator outputs the complete HMHY question.
2) For every role: 2x INTAKE + 2x AMBIGUITY + 2x VALIDATION + 2x direct task (8 total). Same conditioning within each HMHY family allows valid DPO comparison.
3) Fresh TRAIN identical-query PRESENT/ABSENT matched evaluation: 20 roles x 8 questions x 3 background pairs x 2 conditions = up to 960 actual MAS executions. 3 independently matched contrasts per candidate.
4) Select high-confidence CHOSEN and REJECTED for the SAME ROLE AND SAME FAMILY from observed target activation and frozen V4.2 public response discriminant. Min 12 pairs in >=8 roles. IMPORTANT: V4.2 proxy did NOT beat TF-IDF; the resulting reward remains provisional. If insufficient, STOP with no DPO false-positive claim.
5) Native length-normalized DPO from the SFT adapter, using frozen pre-DPO reference scores; heldout ROLES gate. No new base-model training.
6) Sensor continued QLoRA trained on matched contrasts, validation on untrained HMHY strategies and backgrounds. Selector learned with role-held-out CV, fallback if no validated gain.
7) afterok -> MAS5 5x20x2 = 200 actual MAS queries; report Within-MAS AUC, Exact Top-3, F1. MAS5 is a previously inspected DEVELOPMENT cohort, not sealed held-out.

No code was run on BGU from this environment. Only static checks, syntax checks, and local smoke tests were run. Training/evaluation gates can still fail and stop the DAG.

CANCEL ALL CURRENT SLURM JOBS (INCLUDING UNRELATED JOBS):
  scancel -u "$USER"
  squeue -u "$USER"
This cannot undo completed API charges or delete existing checkpoints.

INSTALL, from local computer:
  scp ~/Downloads/mas5_v46_generator_dpo_overnight.zip wieseld@slurm.bgu.ac.il:~/mas_rl_detective/
On BGU login node:
  cd ~/mas_rl_detective
  unzip -n mas5_v46_generator_dpo_overnight.zip
  V46_APPROVE_API_MAS_EXECUTIONS=1160 bash cluster/role_direct_within_v4/v46_submit.sh
Explicit approval covers up to 960 TRAIN + 200 MAS5 = 1160 MAS executions. API calls and monetary charges can be MULTIPLE per MAS execution; unknown exact total. Individual jobs --time=20:00:00, whole chain may exceed 20 hours.

MONITOR:
  squeue -u "$USER"
  RUN=$(ls -dt artifacts/role_direct_within_v4/overnight_v46/mas5_night_* | head -1)
  cat "$RUN/jobs.tsv"
  sacct -j "$(head -1 "$RUN/jobs.tsv" | cut -f1)" -X --format=JobID,JobName%24,State,ExitCode,Elapsed
  tail -n 35 "$(ls -t logs/v46-*.out | head -1)"

CHECK RESULTS AFTER COMPLETION:
  RUN=$(ls -dt artifacts/role_direct_within_v4/overnight_v46/mas5_night_* | head -1)
  cat "$RUN/train/generator_bootstrap_report.json"
  cat "$RUN/train/evidence_report.json"
  cat "$RUN/train/generator_train_report.json"
  cat "$RUN/train/sensor_train_report.json"
  cat "$RUN/train/selector/selector_report.json"
  cat "$RUN/train/COMPLETED_TRAIN.json"
  cat "$RUN/final_MAS5_report.json"

All original V4/V4.2/V4.3/V4.5 models, scripts and artifacts are preserved: new files all named v46_*, run under overnight_v46/.
