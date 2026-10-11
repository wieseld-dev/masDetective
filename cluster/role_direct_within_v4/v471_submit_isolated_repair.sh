#!/usr/bin/env bash
# Isolated GPU regeneration ONLY: no target MAS/API calls, no source bank changes.
set -euo pipefail
cd /home/wieseld/mas_rl_detective
: "${V46_ROOT:?Please export V46_ROOT as the original V4.6.1 MAS5 run directory}"
V46_ROOT=$(realpath -e "$V46_ROOT")
[[ "$V46_ROOT" == "$PWD"/artifacts/role_direct_within_v4/overnight_v46/mas5_night_* ]] || { echo 'Bad run path' >&2; exit 2; }
FOLDER="$V46_ROOT/train/v47_quality"
[[ -f "$FOLDER/curation.json" ]] || { echo 'Run v47_curate curate first (after uploading reviewed TSV)' >&2; exit 2; }
[[ -f "$FOLDER/kept.jsonl" ]] || { echo 'kept.jsonl missing' >&2; exit 2; }
[[ -f "$V46_ROOT/train/generator_bootstrap_adapter/adapter_config.json" ]] || { echo 'Bootstrap SFT adapter missing' >&2; exit 2; }
if [[ -d "$FOLDER/gpu_candidate" ]]; then
  echo 'gpu_candidate already exists. This command will not overwrite it.' >&2
  echo 'Review it first; retries require intentional separate actions.' >&2
  exit 2
fi
if squeue -h -u "$USER" -o '%j' | grep -q '^mas5-'; then
  echo 'A MAS5 job is already queued/running. Refusing concurrent repair.' >&2
  exit 2
fi
PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" - <<'PY'
import inspect
from pathlib import Path
from analysis.role_direct_within_v4 import v47_curate as c
assert 'allow_prepared' in inspect.signature(c.root_paths).parameters, 'Apply earlier V47 audit safety hotfix first'
root=Path(__import__('os').environ['V46_ROOT'])
_,source,folder=c.root_paths(root,allow_prepared=True)
import json
m=json.loads((folder/'curation.json').read_text())
assert c.sha(source)==m['source_sha256'], 'Bank changed since review'
assert c.sha(folder/'review.tsv')==m['review_sha256'], 'Review decisions changed after curation'
assert m['source_count']==160 and m['kept']+m['rejected']==160
print('CURATED SOURCE VERIFIED', 'KEEP',m['kept'],'REGENERATE',m['rejected'])
if (root/'train/candidates_train_round_01.jsonl').exists() or (root/'train/responses_train_round_01.jsonl').exists():
    print('NOTE: existing training artifacts detected: this job regenerates in isolated folder ONLY; do not finalize into this source run.')
PY
mkdir -p logs
"$PY" -m py_compile analysis/role_direct_within_v4/v47_repair.py
"$PY" -m py_compile analysis/role_direct_within_v4/v47_curate.py
JOB=$(sbatch --parsable --export="ALL,V46_ROOT=$V46_ROOT,V47_RETRY=0" \
 cluster/role_direct_within_v4/v47_repair_4090.sbatch)
echo "ISOLATED MAS5 V4.7.1 REPAIR JOB: $JOB"
echo 'Expected: 30 minutes to several hours on RTX 4090 + queue. Zero target MAS/API calls.'
echo "Check: sacct -j $JOB -X --format=JobID,State,ExitCode,Elapsed"
echo "Log: tail -n 60 logs/v47-mas5-v47-repair-$JOB.out"
