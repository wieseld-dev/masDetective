#!/bin/bash
#SBATCH --job-name=av11-snap
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=logs/av11-snap-%j.out
#SBATCH --error=logs/av11-snap-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

ROOT="${AV11_ROLE_ROOT:?}"
DEST="${AV11_SNAPSHOT_DIR:?}"

mkdir -p "$DEST"

for f in \
  "$ROOT"/candidates_dev_round_*.jsonl \
  "$ROOT"/responses_dev_round_*.jsonl \
  "$ROOT"/scored_dev_round_*.jsonl \
  "$ROOT"/states_dev_round_*.jsonl \
  "$ROOT"/metrics_dev_round_*.jsonl \
  "$ROOT"/advance_dev_round_*_summary.json
do
    [[ -e "$f" ]] && cp -f "$f" "$DEST/"
done

echo "SNAPSHOT=$DEST"
