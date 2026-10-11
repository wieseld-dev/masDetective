#!/bin/bash
#SBATCH --job-name=dag-dims
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/dag-dims-%j.out
#SBATCH --error=logs/dag-dims-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

TMPBASE="${SLURM_SCRATCH_DIR:-/tmp}"
mkdir -p "$TMPBASE/tmp"
export TMPDIR="$TMPBASE/tmp"
export JOBLIB_TEMP_FOLDER="$TMPBASE/tmp"

PY="$HOME/.conda/envs/masdetective/bin/python"

for D in 8 16 32 64; do
    echo
    echo "################################################################"
    echo "DIM=$D"
    echo "################################################################"

    export DAG_DIM="$D"

    "$PY" -u scripts/fixed_dag_dim_sweep_v1.py

    mkdir -p "artifacts/fixed_dag_dim_sweep_v1/dim_${D}"

    cp artifacts/fixed_dag_rank_v1/summary.json \
       "artifacts/fixed_dag_dim_sweep_v1/dim_${D}/summary.json"

    cp artifacts/fixed_dag_rank_v1/REPORT.txt \
       "artifacts/fixed_dag_dim_sweep_v1/dim_${D}/REPORT.txt"
done

echo
echo "================ DIMENSION SUMMARY ================"

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/fixed_dag_dim_sweep_v1")

print(
    "DIM | OP_RHO | OP_LIFT | OP_GLOBAL_RHO | "
    "DIRECT_RHO | DIRECT_LIFT | DIRECT_GLOBAL_RHO"
)
print("-" * 90)

for d in [8,16,32,64]:
    x = json.loads(
        (root / f"dim_{d}" / "summary.json").read_text()
    )

    op = x["operator"]["summary"]
    di = x["direct"]["summary"]

    og = x["operator"]["multi_hypothesis_uniform"]
    dg = x["direct"]["multi_hypothesis_uniform"]

    print(
        f"{d:3d} | "
        f"{op['macro_spectral_rho']:+.4f} | "
        f"{op['macro_spectral_lift20']:.3f}x | "
        f"{og['rho']:+.4f} | "
        f"{di['macro_spectral_rho']:+.4f} | "
        f"{di['macro_spectral_lift20']:.3f}x | "
        f"{dg['rho']:+.4f}"
    )
PY
