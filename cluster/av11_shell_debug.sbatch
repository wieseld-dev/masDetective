#!/bin/bash
#SBATCH --job-name=av11-shell
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:03:00
#SBATCH --output=logs/av11-shell-%j.out
#SBATCH --error=logs/av11-shell-%j.err

set -x
set -u

echo "STEP 1: SHELL START"
date
hostname

echo "STEP 2: CD"
cd "$HOME/mas_rl_detective"
pwd

echo "STEP 3: BEFORE CONDA.SH"
ls -l /storage/modules/packages/anaconda/etc/profile.d/conda.sh

echo "STEP 4: SOURCE CONDA"
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
RC=$?
echo "SOURCE_RC=$RC"

echo "STEP 5: ACTIVATE"
conda activate masdetective
RC=$?
echo "ACTIVATE_RC=$RC"

echo "STEP 6: PYTHON LOCATION"
command -v python
python --version

echo "STEP 7: TINY PYTHON"
python -u - <<'PY'
import sys
print("PYTHON ACTUALLY STARTED", flush=True)
print(sys.executable, flush=True)
PY

echo "STEP 8: IMPORT OUR PACKAGE"
python -u - <<'PY'
print("before import", flush=True)
import adaptive_v11_active
print("after import adaptive_v11_active", flush=True)
PY

echo "PASS SHELL DEBUG"
