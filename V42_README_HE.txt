V4.2 role-directed diagnostic + fresh 8x20 probe candidate bank

Not a promise of improved AUC. These are real training-based CPU baselines and new
LoRA-generated *candidate* probes; neither includes PRESENT/ABSENT MAS executions.

Key correction: V137 TRAIN=910 samples over 226 MAS, DEV=760 over 38 MAS.
Train appears much sparser within-system; V4.2 audits actual per-MAS sizes.
No further blind Sensor Epochs or DPO on the six available preferences.

Pipeline:
  - CPU (6 cores/24GB) compares controls and role-aware/ranking methods with
    GroupKFold(system_id) on TRAIN. Select ONLY on TRAIN OOF, then evaluate DEV.
  - GPU (one RTX 4090, 4 cores/32GB) loads CURRENT trained V4 Generator LoRA,
    samples eight different operations for each of 20 roles, validates and
    de-duplicates, including room for observable specialist behavior.
  - Report combines the results and records uncertainty; no API calls.

All output under original V4 run/role_upgrade_v2/run_*.
Old Generator and Sensor adapters are read-only. No alteration to V4.

INSTALL: download ZIP to your local Downloads and scp to BGU:
  scp ~/Downloads/v42_role_upgrade.zip wieseld@slurm.bgu.ac.il:~/mas_rl_detective/
ON SSH:
  cd ~/mas_rl_detective
  unzip -n v42_role_upgrade.zip
  bash cluster/role_direct_within_v4/v42_submit.sh
CHECK:
  squeue --me
  sacct -S 2026-10-09 -X --format=JobID,JobName%20,State,ExitCode,Elapsed | grep v42
  find artifacts/role_direct_within_v4/runs/v4_20261009_185359_1475154/role_upgrade_v2 -name v42_final_summary.json -print
RUNTIME: CPU ~2-12 minutes; 160 probe GPU ~10-35 minutes; queue excluded.
Slurm hard limit: 20 hours per stage.

Next step AFTER seeing numbers: targeted controlled matched PRESENT/ABSENT runs
using the new bank (not automatic; they call an API and have cost). The optional
v42_matched.py tool reads existing executed candidate/response files and scores
matched pairs; it will refuse use of familiar DEV as fresh DPO training.

Limits: no blinded prospective proof of higher Within-MAS AUC / Exact Top-3.

After truly FRESH matched observations exist:
  python -m analysis.role_direct_within_v4.v42_matched \
    --candidates PATH_TO_NEW_CANDIDATES.jsonl \
    --responses PATH_TO_NEW_RESPONSES.jsonl \
    --scorer V42_RUN/sensor_candidate.joblib \
    --out V42_RUN/fresh_matched_report --fresh-train-data
  python -m analysis.role_direct_within_v4.v42_selector \
    --evidence V42_RUN/fresh_matched_report/query_evidence.jsonl \
    --out V42_RUN/fresh_selector
These commands cannot yet be run: new controlled matched execution data must first be collected,
and if the CPU sensor gate fails it must not be treated as an accurate teacher.
Selector heldout folds are by target ROLE, not by repeated system background.
