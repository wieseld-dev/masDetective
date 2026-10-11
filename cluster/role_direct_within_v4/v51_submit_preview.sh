#!/usr/bin/env bash
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -m py_compile analysis/role_direct_within_v4/v51_adaptive_r2.py
mkdir -p logs artifacts/role_direct_within_v4/adaptive_v51
PILOT="$(ls -dt "$PWD"/artifacts/role_direct_within_v4/pilot_v49/pilot_* 2>/dev/null | head -1)"
if [[ -z "$PILOT" || ! -f "$PILOT/responses_train_round_01.jsonl" ]]; then
  echo 'V5.1 requires completed V4.9 pilot (54 responses). No jobs submitted.' >&2;exit 2
fi
RUN="$PWD/artifacts/role_direct_within_v4/adaptive_v51/r2preview_$(date +%Y%m%d_%H%M%S)_$$"
"$PY" -u -m analysis.role_direct_within_v4.v51_adaptive_r2 prepare --root "$RUN" --pilot "$PILOT"
SCORE=$(sbatch --parsable --export="ALL,V51_ROOT=$RUN" cluster/role_direct_within_v4/v51_score_4090.sbatch)
SCORE="${SCORE%%;*}"
GEN=$(sbatch --parsable --dependency="afterok:$SCORE" --export="ALL,V51_ROOT=$RUN,V51_TRIES=${V51_TRIES:-5}" cluster/role_direct_within_v4/v51_generate_4090.sbatch)
GEN="${GEN%%;*}"
printf '%s\tscore\n%s\tgenerate\n' "$SCORE" "$GEN" > "$RUN/jobs.tsv"
echo "MAS5_V51_SUBMITTED R2 preview only, NO MAS/API calls"
echo "RUN=$RUN"
echo "SCORE_JOB=$SCORE GENERATE_JOB=$GEN"
echo "AUDIT WHEN FINISHED: $PY -m analysis.role_direct_within_v4.v51_adaptive_r2 audit --root $RUN"
