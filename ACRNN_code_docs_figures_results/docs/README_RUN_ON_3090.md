# ACRNN 3090 Run Package

This package is for running the DEAP ACRNN reproduction on a rented GPU server.

## What Is Included

- `code/ACRNN/`: runnable ACRNN TensorFlow 1 code.
- `code/summarize_acrnn_results.py`: aggregates per-subject CSV results.
- `data/deap_shuffled_data_3s/`: preprocessed DEAP pkl files used directly by the runner.
- `output/`: empty workspace for ad-hoc outputs.
- `run_*.sh`: command scripts for smoke test and full runs.

Ignore any local folders named `代跑` or `ijcnn-master`; they are not part of this package.

## Expected Server Location

Upload/extract this folder as:

```bash
/root/autodl-tmp/ACRNN_project
```

Then:

```bash
cd /root/autodl-tmp/ACRNN_project
```

## 1. Check GPU And TensorFlow

```bash
nvidia-smi
python - <<'PY'
import tensorflow as tf
print("tf version:", tf.__version__)
print("gpu available:", tf.test.is_gpu_available())
PY
```

Your screenshot already shows RTX 3090 is visible to TensorFlow, so this should print `gpu available: True`.

## 2. Smoke Test First

Run only `s01`, `valence`, fold 1, epoch 1:

```bash
bash run_smoke_s01.sh
```

If this finishes and writes a CSV under `experiments/smoke_test/valence/`, the package path and data path are correct.

## 3. Strict Paper-Parameter Run

This uses the closest original ACRNN settings in this runnable code:

- 10-fold cross-validation
- 200 epochs
- batch size 10
- training keep probability 0.5
- valence and arousal
- all 32 subjects

```bash
bash run_strict_full.sh
```

## 4. Diagnostic Run Matching Previous Local s01-s03 Tests

This is the line that previously ran successfully on local subjects `s01`, `s02`, and `s03`.

## 5. Summarize Results

After any run:

```bash
bash summarize_results.sh
```

Summary CSVs will appear in:

```bash
result_summaries/
```

## What To Download Back

Download these folders after training:

- `experiments/`
- `result_summaries/`
- `logs/`

