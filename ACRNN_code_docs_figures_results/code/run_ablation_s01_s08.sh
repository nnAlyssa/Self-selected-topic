#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs experiments/ablation_s01_s08

SUBJECTS="s01,s02,s03,s04,s05,s06,s07,s08"
COMMON_ARGS=(
  --subjects "$SUBJECTS"
  --data-root data/deap_shuffled_data_3s
  --folds 10
  --epochs 200
  --batch-size 10
  --train-keep-prob 0.5
)

run_variant () {
  local variant="$1"
  shift
  local extra_args=("$@")

  for dimension in valence arousal; do
    echo "===== ${variant} ${dimension} started: $(date) ====="
    python -u code/ACRNN/ACRNN_deap_dat.py \
      "${COMMON_ARGS[@]}" \
      --dimension "$dimension" \
      --result-root "experiments/ablation_s01_s08/${variant}" \
      "${extra_args[@]}"
    echo "===== ${variant} ${dimension} finished: $(date) ====="
  done
}

run_variant no_channel_attention --no-channel-attention
run_variant no_self_attention --no-self-attention
run_variant no_both --no-channel-attention --no-self-attention

python summarize_acrnn_results.py --root . --out-dir result_summaries_ablation_s01_s08
