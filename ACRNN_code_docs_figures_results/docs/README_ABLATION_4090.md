# ACRNN Ablation on RTX 4090D

This package runs three ACRNN ablations on DEAP subjects s01-s08:

- `no_channel_attention`: w/o CWA, keeps self-attention.
- `no_self_attention`: w/o SA, keeps channel-wise attention.
- `no_both`: removes both attention modules, CNN-RNN baseline.

The full ACRNN baseline for the same subjects should use the existing strict
200 epoch result with `train_keep_prob=0.5`.

## Quick Start

```bash
cd /root/autodl-tmp/ACRNN_ablation_project
bash run_ablation_smoke_s01.sh
nohup bash run_ablation_s01_s08.sh > logs/ablation_s01_s08.nohup.log 2>&1 &
```

Check progress:

```bash
ps -ef | grep ACRNN_deap_dat.py
tail -n 80 logs/ablation_s01_s08.nohup.log
find experiments/ablation_s01_s08 -name "*_summary.csv" | wc -l
```

Expected final summary count is `48`: 3 ablation variants * 2 dimensions * 8 subjects.

After it finishes:

```bash
python summarize_acrnn_results.py --root . --out-dir result_summaries_ablation_s01_s08
tar -czf ACRNN_ablation_s01_s08_results.tar.gz \
  experiments/ablation_s01_s08 \
  result_summaries_ablation_s01_s08 \
  logs \
  run_ablation_s01_s08.sh \
  run_ablation_smoke_s01.sh \
  README_ABLATION_4090.md
```
