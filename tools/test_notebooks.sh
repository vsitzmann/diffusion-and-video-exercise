#!/usr/bin/env bash
# Run every notebook top to bottom with the reference solutions switched on and export the
# results (with all outputs) to dev/*.html. Fails if a notebook raises.
# QC_AUTORUN=1 presses notebook 3's "go" button; it loads checkpoints/mine_*.pt if present.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dev
for nb in 01_regression_to_drifting 02_flow_matching 03_video_world_model; do
  echo "== $nb"
  QC_SOLUTIONS=1 QC_AUTORUN=1 ./run .venv/bin/python $nb.py
  QC_SOLUTIONS=1 QC_AUTORUN=1 ./run .venv/bin/marimo export html $nb.py -o dev/$nb.html
done
