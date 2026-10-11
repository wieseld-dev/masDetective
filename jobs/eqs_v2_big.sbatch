#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=1-00:00:00
#SBATCH --job-name=eqs-v2-big
#SBATCH --output=logs/eqs-v2-big-%J.out
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PYTHON="$HOME/.conda/envs/eqs/bin/python"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore:Failed to initialize NumPy"

echo "============================================================"
echo "EQS V2 BIG MULTI-WORLD EXPERIMENT"
echo "============================================================"
echo "JOB=$SLURM_JOB_ID"
echo "NODE=$(hostname)"
echo "START=$(date)"
echo "PYTHON=$PYTHON"
echo

test -x "$PYTHON"
test -s scripts/eqs_multiworld_big.py
mkdir -p artifacts/eqs_v2_big

"$PYTHON" - <<'PY'
import torch
print("torch:", torch.__version__)
x=torch.randn(32,32); y=x@x.T
assert torch.isfinite(y).all()
print("ENVIRONMENT TEST: PASS")
PY

for SEED in 7 17 27; do
  echo
  echo "################################################################"
  echo "MODEL SEED=$SEED START=$(date)"
  echo "################################################################"
  "$PYTHON" -u scripts/eqs_multiworld_big.py \
    --model-seed "$SEED" \
    --master-seed 1234 \
    --train-worlds 96 \
    --dev-worlds 24 \
    --test-worlds-id 32 \
    --test-worlds-ood 32 \
    --train-episodes 220 \
    --dev-episodes 70 \
    --test-ranking-episodes 80 \
    --rollout-episodes 60 \
    --epochs 12 \
    --max-history 8 \
    --batch-size 256 \
    --dev-states 6000 \
    --outdir "artifacts/eqs_v2_big/seed${SEED}"
  echo "MODEL SEED=$SEED DONE=$(date)"
done

"$PYTHON" - <<'PY'
import glob, json, statistics, os
paths=sorted(glob.glob("artifacts/eqs_v2_big/seed*/results.json"))
if len(paths) != 3:
    raise SystemExit(f"Expected 3 result files, found {len(paths)}: {paths}")
runs=[json.load(open(p)) for p in paths]

def get(r, path):
    x=r
    for k in path.split('.'):
        x=x[k]
    return float(x)

metrics={
 "ID ranking seenQ/oracle":"test_id.ranking_seen_queries.utility_ratio_vs_oracle",
 "ID ranking heldoutQ/oracle":"test_id.ranking_heldout_queries.utility_ratio_vs_oracle",
 "OOD ranking seenQ/oracle":"test_ood.ranking_seen_queries.utility_ratio_vs_oracle",
 "OOD ranking heldoutQ/oracle":"test_ood.ranking_heldout_queries.utility_ratio_vs_oracle",
 "ID seenQ random acc@8":"test_id.policy_seen_queries.random.8.accuracy",
 "ID seenQ static acc@8":"test_id.policy_seen_queries.static.8.accuracy",
 "ID seenQ learned acc@8":"test_id.policy_seen_queries.learned.8.accuracy",
 "ID seenQ oracle acc@8":"test_id.policy_seen_queries.oracle.8.accuracy",
 "OOD seenQ random acc@8":"test_ood.policy_seen_queries.random.8.accuracy",
 "OOD seenQ static acc@8":"test_ood.policy_seen_queries.static.8.accuracy",
 "OOD seenQ learned acc@8":"test_ood.policy_seen_queries.learned.8.accuracy",
 "OOD seenQ oracle acc@8":"test_ood.policy_seen_queries.oracle.8.accuracy",
 "ID allQ learned acc@8":"test_id.policy_all_queries.learned.8.accuracy",
 "ID allQ oracle acc@8":"test_id.policy_all_queries.oracle.8.accuracy",
 "OOD allQ learned acc@8":"test_ood.policy_all_queries.learned.8.accuracy",
 "OOD allQ oracle acc@8":"test_ood.policy_all_queries.oracle.8.accuracy",
 "ID seenQ learned entropy@8":"test_id.policy_seen_queries.learned.8.entropy",
 "ID seenQ learned NLL@8":"test_id.policy_seen_queries.learned.8.nll",
 "OOD seenQ learned entropy@8":"test_ood.policy_seen_queries.learned.8.entropy",
 "OOD seenQ learned NLL@8":"test_ood.policy_seen_queries.learned.8.nll",
}
summary={}
for name,path in metrics.items():
    xs=[get(r,path) for r in runs]
    summary[name]={"mean":statistics.fmean(xs),"std":statistics.pstdev(xs),"values":xs}

def gap(r, split):
    p=r[split]["policy_seen_queries"]
    a=float(p["random"]["8"]["accuracy"])
    b=float(p["learned"]["8"]["accuracy"])
    c=float(p["oracle"]["8"]["accuracy"])
    return (b-a)/(c-a) if abs(c-a)>1e-9 else None
for split,label in [("test_id","ID oracle-gap-closure@8"),("test_ood","OOD oracle-gap-closure@8")]:
    xs=[gap(r,split) for r in runs]
    xs=[x for x in xs if x is not None]
    summary[label]={"mean":statistics.fmean(xs),"std":statistics.pstdev(xs),"values":xs}

out={"result_files":paths,"summary":summary}
os.makedirs("artifacts/eqs_v2_big",exist_ok=True)
json.dump(out,open("artifacts/eqs_v2_big/aggregate.json","w"),indent=2)
print("\n"+"="*88)
print("AGGREGATE OVER 3 SEEDS")
print("="*88)
for name,v in summary.items():
    print(f"{name:38s} mean={v['mean']:.4f} std={v['std']:.4f}")
print("\nSaved: artifacts/eqs_v2_big/aggregate.json")
PY

echo
echo "============================================================"
echo "ALL DONE: $(date)"
echo "AGGREGATE: artifacts/eqs_v2_big/aggregate.json"
echo "============================================================"
