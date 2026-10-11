#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v42-%x-%j.out
#SBATCH --error=logs/v42-%x-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
"$HOME/.conda/envs/masdetective/bin/python" - "$V42_ROOT" <<'PY'
import json,sys,pathlib
p=pathlib.Path(sys.argv[1]);files={'sensor':p/'sensor_benchmark.json','generator':p/'probe_generation_round_01.json'}
result={}
for name,f in files.items():
    if not f.exists():result[name]={'status':'MISSING_OR_FAILED'};continue
    d=json.loads(f.read_text());result[name]=({'status':d['status'],'selected_model':d['selected_from_train_oof'],
    'within_mas_auc':d['dev_results'][d['selected_from_train_oof']]['metrics']['within_auc'],
    'exact_topk':d['dev_results'][d['selected_from_train_oof']]['metrics']['exact_topk'],
    'gate':d['promotion_gate']} if name=='sensor' else
    {'status':d['status'],'candidate_count':d['candidate_count'],'per_role':d['per_role']})
f=p/'v42_final_summary.json';f.write_text(json.dumps(result,indent=2)+'\n');print('V42_FINAL',json.dumps(result,indent=2))
PY
