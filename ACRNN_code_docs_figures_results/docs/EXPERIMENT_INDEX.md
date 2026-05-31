# Experiment Index

This project keeps raw DEAP data, preprocessing outputs, model code, and result
folders separated. Result folders are organized under `experiments/`.

## Directory Layout

### `experiments/dropout_disabled/`

Diagnostic runs with dropout disabled. These runs are useful for checking
whether the public ACRNN graph can learn at all, but they are not strict
original-parameter reproductions.

Current contents:

- `result_dropout_disabled_30ep_3fold`
- `result_dropout_disabled_30ep_folds4_6`
- `result_dropout_disabled_30ep_folds7_10`
- `result_dropout_disabled_30ep_arousal_folds1_3`
- `result_dropout_disabled_30ep_arousal_folds4_6`
- `result_dropout_disabled_30ep_arousal_folds7_10`

Core conclusion: disabling dropout avoids majority-class collapse and confirms
that the reconstructed public-code graph can train.

### `experiments/keep_prob_08/`

Main diagnostic reproduction runs with dropout kept enabled but relaxed to
`keep_prob=0.8`. These are the current best working runs for the DEAP `.dat`
pipeline.

Completed subjects/dimensions:

- `s01 valence`: final `0.9350 +/- 0.0325`; best epoch `0.9550 +/- 0.0195`
- `s01 arousal`: final `0.9212 +/- 0.0419`; best epoch `0.9475 +/- 0.0208`
- `s02 valence`: final `0.8062 +/- 0.0372`; best epoch `0.8225 +/- 0.0450`
- `s02 arousal`: final `0.7800 +/- 0.0392`; best epoch `0.8100 +/- 0.0357`
- `s03 valence`: final `0.9238 +/- 0.0288`; best epoch `0.9350 +/- 0.0242`
- `s03 arousal`: final `0.9462 +/- 0.0280`; best epoch `0.9525 +/- 0.0236`

Core conclusion: `keep_prob=0.8` substantially mitigates the majority-class
collapse seen with the public `keep_prob=0.5` setting while preserving dropout.
Treat it as a diagnostic correction, not an unchanged original-parameter result.

### `experiments/keep_prob_scan/`

Single-fold dropout-strength scan on `s01 valence`, fold 1.

Current contents:

- `result_keep_prob_scan_valence_kp05_fold1`
- `result_keep_prob_scan_valence_kp07_fold1`
- `result_keep_prob_scan_valence_kp08_fold1`
- `result_keep_prob_scan_valence_kp09_fold1`

Core conclusion: `keep_prob=0.5` tends to collapse to a majority-class
prediction in this reconstructed environment, while milder dropout values
recover learning.

### `experiments/reference_short/`

Short baseline run using the reference-style path.

Current contents:

- `result_reference_ijcnn_short`

Core conclusion: the short reference run stayed near chance on `s01 valence`,
so it is preserved as a diagnostic comparison rather than the main result.

### `experiments/old_or_misc_results/`

Reserved for result folders whose purpose is unclear. It is currently empty.

## Unified Summaries

The summary script reads from `experiments/**` and writes aggregate CSVs to
`result_summaries/`:

- `all_experiment_folds.csv`
- `per_subject_summary.csv`
- `overall_summary.csv`
- `result_group_summary.csv`

Compatibility aliases are also written:

- `all_fold_results.csv`
- `group_summary.csv`

## Current Next Step

Continue with the same controlled setup before expanding claims:

```powershell
D:\Ana\Scripts\conda.exe run --live-stream -n acrnn_tf1 python summarize_acrnn_results.py
```

For new experiments, place new result roots under `experiments/keep_prob_08/`
or a new clearly named subdirectory.
