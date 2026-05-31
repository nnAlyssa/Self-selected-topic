#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs experiments

python -u code/ACRNN/ACRNN_deap_dat.py \
  --subjects s01 \
  --dimension valence \
  --data-root data/deap_shuffled_data_3s \
  --result-root experiments/ablation_smoke/no_both \
  --folds 10 \
  --fold-start 1 \
  --fold-end 1 \
  --epochs 1 \
  --batch-size 10 \
  --train-keep-prob 0.5 \
  --no-channel-attention \
  --no-self-attention
