#!/usr/bin/env bash
set -euo pipefail
cd /home/wieseld/mas_rl_detective
: "${V46_ROOT:?Export V46_ROOT pointing to your existing MAS5 run}"
export V46_ROOT
PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -m py_compile analysis/role_direct_within_v4/v477_generator.py analysis/role_direct_within_v4/v477_regenerate.py analysis/role_direct_within_v4/v478_reconcile.py
"$PY" - "$V46_ROOT" <<'PYCHECK'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]).expanduser().resolve(strict=True)
d=r/'train/v47_quality/v472_reviewed_second_pass/v476_manual_review'
a=[json.loads(s) for s in (d/'approved_118.jsonl').read_text().splitlines() if s.strip()]
assert len(a)==118 and len({(x['target_role'],int(x['strategy_index'])) for x in a})==118
assert (r/'train/generator_bootstrap_adapter/adapter_config.json').is_file()
print('V478 PREFLIGHT: 118 immutable approved questions; resume with GPU',flush=True)
PYCHECK
if squeue -h -u "$USER" -o '%j' | grep -q '^mas5-'; then echo 'Active MAS5 Slurm job: refusing concurrency' >&2;exit 2;fi
mkdir -p logs
JOB=$(sbatch --parsable --export="ALL,V46_ROOT=$V46_ROOT,V477_ATTEMPTS=${V477_ATTEMPTS:-18}" cluster/role_direct_within_v4/v478_regenerate_4090.sbatch)
echo "V478 JOB=$JOB"
echo "STATUS: sacct -j $JOB -X --format=JobID,State,ExitCode,Elapsed"
echo "LOG: logs/v478-mas5-v478-regen-$JOB.out"
