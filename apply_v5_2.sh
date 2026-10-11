#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p detective rl training tests
# Files are already overlaid by unzip; this script is a sanity marker.
echo "V5.2 benign-RL patch present."
python -m pytest -q
