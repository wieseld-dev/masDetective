#!/bin/bash
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"; export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
PY="$HOME/.conda/envs/masdetective/bin/python"
EXP="${EIG_EXPERIMENT_ROOT:-artifacts/adaptive_v11_query_eig_v2}"
BELIEF="${EIG_BELIEF:-artifacts/adaptive_v11_active_v1/frozen_belief.joblib}"
UTIL="$EXP/query_utility.joblib"
[ -s artifacts/adaptive_v11_k_bootstrap_v1/states_train_round_06.jsonl ] || { echo 'missing KBOOT training trajectories'; exit 1; }
[ -s artifacts/adaptive_v11_controller_v1/static/states_dev_round_06.jsonl ] || { echo 'missing prior controller-v1 trajectories'; exit 1; }
[ -s "$BELIEF" ] || { echo "missing belief: $BELIEF"; exit 1; }
[ -d artifacts/v11_policy ] || { echo 'missing original V11 adapter'; exit 1; }
if [ -e "$EXP" ] && find "$EXP" -name 'responses_*' -print -quit | grep -q .; then echo "REFUSING: $EXP contains executed responses"; exit 1; fi
mkdir -p "$EXP"/{common,static,maxent,utility,report}
echo '===== FIT QUERY UTILITY (NO API) ====='
"$PY" -u scripts/av11_fit_query_utility.py --output "$UTIL"
echo '===== FRESH HELD-OUT MANIFEST ====='
"$PY" -u scripts/kboot_make_manifest.py --output "$EXP/manifest_fresh.json" --seed "${EIG_MANIFEST_SEED:-2026091727}" --train-per-cell 1 --dev-per-cell 2
for arm in common static maxent utility; do cp "$EXP/manifest_fresh.json" "$EXP/$arm/manifest_privileged.json"; "$PY" -u scripts/adaptive_v11_init_states.py --manifest "$EXP/$arm/manifest_privileged.json" --belief "$BELIEF" --output-dir "$EXP/$arm"; done
"$PY" - <<'PY'
import json, os
from pathlib import Path
exp=Path(os.environ.get('EIG_EXPERIMENT_ROOT','artifacts/adaptive_v11_query_eig_v2'))
o=json.loads((exp/'manifest_fresh.json').read_text()); d=[x for x in o['systems'] if x['split']=='dev']
assert len(d)==32
print('PASS')
print('fresh DEV systems=',len(d))
print('arms=common/static/maxent/utility initialized')
print('utility model trained only from prior trajectories; fresh DEV is unseen')
print('shared exact R1, then 3 arms R2-R6')
print('MAX TARGET QUERIES = 32 + 3*32*5 = 512')
PY
echo "EXP_ROOT=$EXP"
echo 'PREPARED ONLY. NOTHING EXPENSIVE WAS SUBMITTED.'
