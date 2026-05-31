#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs experiments

python -u code/ACRNN/ACRNN_deap_dat.py \
  --subjects s01 \
  --dimension valence \
  --raw-data-root data/data_preprocessed_python \
  --result-root experiments/trial_level_smoke/full_acrnn \
  --split-protocol trial \
  --folds 5 \
  --fold-start 1 \
  --fold-end 1 \
  --epochs 1 \
  --batch-size 10 \
  --train-keep-prob 0.5
