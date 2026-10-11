#!/bin/bash
#SBATCH --job-name=q38r7-load
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=logs/q38r7-load-%j.out
#SBATCH --error=logs/q38r7-load-%j.err

set -euo pipefail
cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

PY="$HOME/.conda/envs/masdetective/bin/python"

"$PY" -u <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("analysis").resolve()))

import ten_system_belief_trace_control_v1 as ctrl

class _UnusedGenerator:
    def __init__(self, *args, **kwargs):
        pass

ctrl.CurriculumAdaptiveProbePolicyV11 = _UnusedGenerator

print("LOADING...")
pipe = ctrl.load_pipeline()

print("PIPE KEYS =", sorted(pipe.keys()))
print("EMBEDDER =", type(pipe["embedder"]).__name__)
print("FP_MODELS =", type(pipe["fp_models"]).__name__)
print("FP_CALS =", type(pipe["fp_cals"]).__name__)
print("CPU_VISIBLE_PIPELINE_OK")
PY
