# ACRNN Trial-Level Split Check on RTX 4090D

This package runs Full ACRNN under a stricter trial-level split on DEAP
subjects s01-s08. In this protocol, all 20 segments from the same trial stay
together in either train or test.

It is intended as a protocol robustness check against the sample-level split.

## Quick Start

```bash
cd /root/autodl-tmp/ACRNN_trial_level_s01_s08_upload
bash run_trial_level_smoke_s01.sh
nohup bash run_trial_level_s01_s08.sh > logs/trial_level_s01_s08.nohup.log 2>&1 &
```

Check progress:

```bash
ps -ef | grep ACRNN_deap_dat.py
tail -n 80 logs/trial_level_s01_s08.nohup.log
find experiments/trial_level_s01_s08 -name "*_summary.csv" | wc -l
```

Expected final summary count is `16`: 8 subjects * 2 dimensions.

After it finishes:

```bash
python summarize_acrnn_results.py --root . --out-dir result_summaries_trial_level_s01_s08
tar -czf ACRNN_trial_level_s01_s08_results.tar.gz \
  experiments/trial_level_s01_s08 \
  result_summaries_trial_level_s01_s08 \
  logs \
  run_trial_level_s01_s08.sh \
  run_trial_level_smoke_s01.sh \
  README_TRIAL_LEVEL_4090.md
```
