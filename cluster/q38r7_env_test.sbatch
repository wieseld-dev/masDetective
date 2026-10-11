#!/bin/bash
#SBATCH --job-name=q38r7-env
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=10:00:00
#SBATCH --output=logs/q38r7-env-%j.out
#SBATCH --error=logs/q38r7-env-%j.err

set -euo pipefail
cd "$HOME/mas_rl_detective"

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

: "${OPENAI_API_KEY:?OPENAI_API_KEY missing}"

python - <<'PY'
import os
assert os.environ.get("OPENAI_API_KEY")
print("OPENAI_CREDENTIAL_ENV_OK")
PY
