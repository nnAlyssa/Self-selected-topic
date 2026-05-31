#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs experiments/trial_level_s01_s08

SUBJECTS="s01,s02,s03,s04,s05,s06,s07,s08"
COMMON_ARGS=(
  --subjects "$SUBJECTS"
  --raw-data-root data/data_preprocessed_python
  --result-root experiments/trial_level_s01_s08/full_acrnn
  --split-protocol trial
  --folds 5
  --epochs 200
  --batch-size 10
  --train-keep-prob 0.5
)

for dimension in valence arousal; do
  echo "===== full_acrnn trial-level ${dimension} started: $(date) ====="
  python -u code/ACRNN/ACRNN_deap_dat.py \
    "${COMMON_ARGS[@]}" \
    --dimension "$dimension"
  echo "===== full_acrnn trial-level ${dimension} finished: $(date) ====="
done

python summarize_acrnn_results.py --root . --out-dir result_summaries_trial_level_s01_s08
