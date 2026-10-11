#!/usr/bin/env bash
set -euo pipefail
cd /home/wieseld/mas_rl_detective
: "${V46_ROOT:?Export V46_ROOT as the existing MAS5 run}"
V46_ROOT=$(realpath -e "$V46_ROOT")
export V46_ROOT
[[ "$V46_ROOT" == "$PWD"/artifacts/role_direct_within_v4/overnight_v46/mas5_night_* ]] || { echo 'Invalid RUN'; exit 2; }
if squeue -h -u "$USER" -o '%j' | grep -q '^mas5-'; then
  echo 'Another MAS5 job is queued/running; refusing concurrent work.' >&2
  exit 2
fi
PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -m py_compile analysis/role_direct_within_v4/v477_generator.py analysis/role_direct_within_v4/v477_regenerate.py
"$PY" - "$V46_ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
d=root/'train/v47_quality/v472_reviewed_second_pass/v476_manual_review'
m=json.loads((d/'manifest.json').read_text())
a=[json.loads(x) for x in (d/'approved_118.jsonl').read_text().splitlines() if x.strip()]
assert m['approved_total']==118 and m['regenerate']==42 and len(a)==118
assert len({(r['target_role'],int(r['strategy_index'])) for r in a})==118
assert (root/'train/generator_bootstrap_adapter/adapter_config.json').is_file()
print('PREFLIGHT OK: 118 approved, 42 missing, trained adapter found; no MAS calls')
PY
mkdir -p logs
JOB=$(sbatch --parsable --export="ALL,V46_ROOT=$V46_ROOT,V477_ATTEMPTS=${V477_ATTEMPTS:-18}" cluster/role_direct_within_v4/v477_regenerate_4090.sbatch)
echo "MAS5_V477_JOB=$JOB"
echo "Watch: sacct -j $JOB -X --format=JobID,State,ExitCode,Elapsed"
echo "Log: logs/v477-mas5-v477-regen-$JOB.out"
echo 'Generation only. No target MAS/API calls, no DPO training.'
