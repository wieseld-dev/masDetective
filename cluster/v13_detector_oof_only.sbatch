#!/bin/bash
#SBATCH --job-name=v13-oof
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/v13-oof-%j.out

set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

python - <<'PY'
import json

from v8.embedding import FrozenOutputEmbedderV8
from v11.role_universe import TRAIN_ROLES, ROLE_DESCRIPTIONS
from v13.fingerprint import (
    BehavioralFingerprintDetectorV13,
    load_fingerprint_examples,
)

examples = load_fingerprint_examples([
    ("artifacts/v13_probe_disambiguation.jsonl", 1.0),
    ("artifacts/v11_probe_discovery.jsonl", 0.35),
], set(TRAIN_ROLES))

print("examples:", len(examples))

embedder = FrozenOutputEmbedderV8(device="cpu")
detector = BehavioralFingerprintDetectorV13(
    embedder,
    ROLE_DESCRIPTIONS,
    c=0.05,
)

report = detector.fit(examples, TRAIN_ROLES, folds=5)

with open("artifacts/v13_detector_oof_only.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n=== V13 DETECTOR OOF ===")
print("Macro AUC:", report["macro_oof_auc"])
print("Macro BA :", report["macro_oof_balanced_accuracy"])

print("\nPer role:")
for role, x in sorted(
    report["roles"].items(),
    key=lambda kv: kv[1]["oof_auc"]
):
    print(
        f"{role:28s} "
        f"AUC={x['oof_auc']:.3f} "
        f"BA={x['oof_balanced_accuracy']:.3f} "
        f"n={x['examples']} "
        f"groups={x['groups']}"
    )
PY
