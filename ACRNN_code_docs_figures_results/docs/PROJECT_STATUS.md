# PROJECT_STATUS

## Current Goal

The current project goal is to reproduce ACRNN for the paper **EEG-Based Emotion Recognition via Channel-Wise Attention and Self Attention**.

The main reproduction line should return to the DEAP protocol as closely as possible:

`DEAP .dat -> DEAP preprocessing -> .pkl -> original ACRNN.py -> 10-fold cross-validation -> valence/arousal accuracy`

## Data Status

- Main data: `data_preprocessed_python/data_preprocessed_python/s01.dat` through `s32.dat`.
- These files are the DEAP Python preprocessed format and should be read with `pickle.load(..., encoding="latin1")`.
- Expected fields are `subject["data"]` and `subject["labels"]`.
- Expected shape is likely:
  - `data`: `(40, 40, 8064)`
  - `labels`: `(40, 4)`
- Old H5 data: `DEAP_h5/train.h5`, `DEAP_h5/val.h5`, `DEAP_h5/test.h5`.
- Old H5 data and checkpoints are now classified as **H5-adapted reproduction / failure analysis**, not strict reproduction results.

## Mainline Files

- `ACRNN/ACRNN.py`: original TensorFlow ACRNN training script. It expects pre-generated `.pkl` data files, not raw `.dat` files.
- `ACRNN/cnn_class.py`: CNN helper used by the original model.
- `ACRNN/Attention/`: original attention modules used by ACRNN.
- `ACRNN/readme.txt`: original project notes.
- `README.md`: original project README.
- `EEG-Based_Emotion_Recognition_via_Channel-Wise_Attention_and_Self_Attention.pdf`: paper PDF.
- `data_preprocessed_python/`: current main DEAP data source.
- `inspect_deap_dat.py`: local inspection script for the DEAP `.dat` data.
- `deap_pre_process_from_dat.py`: reconstructed `.dat -> .pkl` preprocessing script for the main DEAP line.
- `ACRNN/ACRNN_deap_dat.py`: runnable local training entry that keeps the ACRNN model line but fixes local runtime issues in `ACRNN.py` without editing the original file.
- `deap_shuffled_data_3s/`: generated ACRNN-compatible `.pkl` data for valence and arousal.

## Existing Script Notes

- No standalone `deap_pre_process.py` was found in the current project tree.
- No raw `.mat -> .pkl` preprocessing script was found under a similar name.
- `ACRNN/ACRNN.py` contains a function named `deap_preprocess`, but it only loads existing `.pkl` files from a hard-coded Linux path:
  - `/home/taozi12345/deap_shuffled_data_3s/yes_<dimension>/`
  - suffixes `.mat_win_384_rnn_dataset.pkl` and `.mat_win_384_labels.pkl`
- Because the original preprocessing script is missing, `deap_pre_process_from_dat.py` was created as a conservative reconstruction, not as a recovered original file.
- `deap_pre_process_from_dat.py` reads DEAP `.dat` files, uses the first 384 samples as three one-second baseline slices, averages them into a one-second baseline template, subtracts that template from each one-second trial slice, splits the following 7680 samples into 20 windows of 384 samples, keeps the first 32 EEG channels, and writes pickle files matching the names expected by `ACRNN/ACRNN.py`.
- The generated `.pkl` files are written with pickle protocol 4 and were regenerated under the `acrnn_tf1` conda environment so Python 3.7 / NumPy 1.21 can read them.
- `ACRNN/ACRNN_deap_dat.py` was added because `ACRNN/ACRNN.py` has runtime blockers: hard-coded Linux paths, invalid import syntax for `channel-wise_attention.py`, undefined `subjects`, and a DEAP data reshape that still uses `14` channels.

## H5-Adapted Reproduction / Failure Analysis

The following files/directories belong to the old H5 adaptation line and are not the strict reproduction mainline:

- `inspect_h5.py`: checks `train.h5`, `val.h5`, `test.h5` shape, dtype, statistics, and label distribution.
- `inspect_h5_detail.py`: more detailed H5 inspection. Its hard-coded path appears to contain mojibake and may be invalid.
- `prepare_h5_for_acrnn.py`: converts H5 `X: (N, 32, 2000)` into `X: (N, 5, 32, 400, 1)` and one-hot labels.
- `ACRNN/ACRNN_h5.py`: H5 engineering adaptation using full 2000 time points as `5 x 400`.
- `ACRNN/ACRNN_h5_repro_minimal.py`: H5 minimal reproduction attempt using a 384-point window; not a protocol-level reproduction.
- `ACRNN/ACRNN_h5_diagnostic.py`: H5 diagnostics for `first384`, `center384`, and `full2000_5x400`, with confusion matrix and macro-F1.
- `checkpoints_h5/`: checkpoints from the H5 engineering adaptation.
- `checkpoints_h5_repro_minimal/`: checkpoints from the H5 minimal adaptation.

These files have been archived under `archive_h5_adaptation/` instead of deleted.

Archived locations:

- `archive_h5_adaptation/inspect_h5.py`
- `archive_h5_adaptation/inspect_h5_detail.py`
- `archive_h5_adaptation/prepare_h5_for_acrnn.py`
- `archive_h5_adaptation/ACRNN/ACRNN_h5.py`
- `archive_h5_adaptation/ACRNN/ACRNN_h5_repro_minimal.py`
- `archive_h5_adaptation/ACRNN/ACRNN_h5_diagnostic.py`
- `archive_h5_adaptation/checkpoints_h5/`
- `archive_h5_adaptation/checkpoints_h5_repro_minimal/`

## Uncertain / Secondary Files

- `ACRNN-main/`: PyTorch implementation/training scripts using `.pth` files from `../../data/data_preprocessed_ACRNN/`. It is not the original TensorFlow ACRNN mainline and was not moved.
- `data_preprocessed_python.zip`: source archive for the DEAP Python preprocessed data. Keep for traceability.
- `DEAP_h5/`: old H5 split data. It is H5-related, but it was not included in the explicit move list yet because it is source data rather than generated script/checkpoint output.

## Next Run Order

1. Inspect DEAP `.dat` data:

   ```powershell
   D:\Ana\python.exe inspect_deap_dat.py
   ```

2. Generate ACRNN-compatible `.pkl` files for one subject first:

   ```powershell
   D:\Ana\python.exe deap_pre_process_from_dat.py --subjects s01 --dimension all
   ```

3. If the single-subject output is correct, generate all subjects:

   ```powershell
   D:\Ana\python.exe deap_pre_process_from_dat.py --subjects all --dimension all
   ```

   For TensorFlow 1 compatibility, prefer regenerating with:

   ```powershell
   $env:PYTHONIOENCODING='utf-8'
   D:\Ana\Scripts\conda.exe run -n acrnn_tf1 python deap_pre_process_from_dat.py --subjects all --dimension all --overwrite
   ```

4. Smoke-test the local ACRNN runner:

   ```powershell
   $env:PYTHONIOENCODING='utf-8'
   D:\Ana\Scripts\conda.exe run -n acrnn_tf1 python ACRNN\ACRNN_deap_dat.py --subjects s01 --dimension valence --folds 2 --epochs 1 --batch-size 10
   ```

5. Run 10-fold cross-validation for valence and arousal:

   ```powershell
   $env:PYTHONIOENCODING='utf-8'
   D:\Ana\Scripts\conda.exe run -n acrnn_tf1 python ACRNN\ACRNN_deap_dat.py --subjects all --dimension valence --folds 10 --epochs 200 --batch-size 10
   D:\Ana\Scripts\conda.exe run -n acrnn_tf1 python ACRNN\ACRNN_deap_dat.py --subjects all --dimension arousal --folds 10 --epochs 200 --batch-size 10
   ```

## Smoke Test

- Command: `D:\Ana\Scripts\conda.exe run -n acrnn_tf1 python ACRNN\ACRNN_deap_dat.py --subjects s01 --dimension valence --folds 2 --epochs 1 --batch-size 10`
- Result: completed successfully.
- Output: `result_att_cnn_rnn_att_deap/valence/s01_summary.csv`
- This is only a runtime smoke test, not a reproduction result.

## Baseline Revision Test

- Previous generated `.pkl` and short-test results were archived under `archive_preprocess_versions/`.
- Baseline removal was revised to match the paper description more closely: average the three one-second baseline slices into a one-second baseline template, then subtract it from every one-second trial slice before 3-second windowing.
- Regenerated all valence/arousal `.pkl` files with `acrnn_tf1`.
- Short test command: `D:\Ana\Scripts\conda.exe run --live-stream -n acrnn_tf1 python -u ACRNN\ACRNN_deap_dat.py --subjects s01 --dimension valence --folds 10 --epochs 5 --batch-size 10`
- Short test result: `s01 mean accuracy=0.5162 std=0.0559`.
- Interpretation: the revised baseline path runs, but the short test remains near chance; more diagnostics are needed before treating results as reproduction-quality.

## Prediction Diagnostics

- `ACRNN/ACRNN_deap_dat.py` now writes per-fold true label counts, prediction counts, final confusion matrix, and best-epoch confusion matrix to the summary CSV.
- Diagnostic command: `D:\Ana\Scripts\conda.exe run --live-stream -n acrnn_tf1 python -u ACRNN\ACRNN_deap_dat.py --subjects s01 --dimension valence --folds 10 --epochs 5 --batch-size 10`
- Diagnostic output: `result_att_cnn_rnn_att_deap/valence/s01_summary.csv`
- Finding: several folds collapse toward predicting class 0 almost exclusively, for example:
  - fold 1 best `pred_count_0=80`, `pred_count_1=0`
  - fold 3 final `pred_count_0=80`, `pred_count_1=0`
  - fold 4 final `pred_count_0=79`, `pred_count_1=1`
  - fold 6 final `pred_count_0=80`, `pred_count_1=0`
  - fold 10 final `pred_count_0=79`, `pred_count_1=1`
- Interpretation: the weak accuracy is not just display noise; the current training/data reconstruction frequently collapses to majority-class prediction. Further investigation should focus on preprocessing equivalence, sample ordering/shuffling, normalization, and the exact original training graph behavior.

## Diagnostic Variants

- Added diagnostic flags to `ACRNN/ACRNN_deap_dat.py`:
  - `--standardize`: standardize each fold using training-set mean/std.
  - `--train-phase-train`: feed `train_phase=True` during optimizer steps.
- Added `ACRNN/Attention/channel_wise_attention_compat.py`.
  - This keeps the same channel attention computation but replaces the original long `tf.concat([x] * (H * W))` expansion with `tf.tile`.
  - Purpose: remove TensorFlow 1.15 Grappler `concat self cycle` errors without editing the original `channel-wise_attention.py`.
- Quick diagnostic results on `s01 valence`, `2 folds`, `3 epochs`:
  - baseline runner: mean accuracy `0.4925`, collapsed to single-class prediction in final folds.
  - `--standardize`: mean accuracy `0.5037`, less extreme but still near chance.
  - `--train-phase-train`: mean accuracy `0.5275`, still collapsed toward class 0.
  - compat `tf.tile` channel attention: mean accuracy `0.4950`, Grappler self-cycle error removed, still near chance.
  - compat `tf.tile` + `--standardize`: mean accuracy `0.5263`, still near chance.
- Interpretation: graph compatibility, simple standardization, and BN training mode do not recover the paper-level trend. The remaining gap most likely comes from missing original `deap_pre_process.py` details, non-obvious training/evaluation protocol differences, or another unrecovered code path from the authors' environment.

## ijcnn Reference Preprocessing

- The ACRNN README cites `ynulonger/ijcnn` as a reference. A local copy was added under `ijcnn-master/`, and it contains `ijcnn-master/deap_pre_process.py`.
- `deap_pre_process_from_dat.py` was updated to adapt the ijcnn reference preprocessing to DEAP Python `.dat` files:
  - load `.dat` with pickle
  - transpose data from `(40, 40, 8064)` to `(40, 8064, 40)`
  - use labels `> 5` for valence/arousal
  - subtract the average of the three one-second baseline slices from every one-second trial slice
  - normalize every time point across the 32 EEG channels using `norm_dataset`
  - use the ijcnn window routine, including the `<` end condition and duplicated first window
  - shuffle generated samples with `np.random.seed(0)` during preprocessing
- Existing paper-baseline `.pkl` and diagnostics were archived under `archive_preprocess_versions/`.
- Regenerated all `deap_shuffled_data_3s/yes_valence` and `deap_shuffled_data_3s/yes_arousal` files with the `acrnn_tf1` environment.
- Validation for `s01` valence after ijcnn-compatible preprocessing:
  - data shape `(800, 384, 32)`
  - global mean approximately `0`
  - global std `1.0`
  - first time point channel std approximately `1.0`
- Short diagnostic command:

  ```powershell
  D:\Ana\Scripts\conda.exe run --live-stream -n acrnn_tf1 python -u ACRNN\ACRNN_deap_dat.py --subjects s01 --dimension valence --folds 10 --epochs 5 --batch-size 10 --no-load-shuffle --result-root result_reference_ijcnn_short
  ```

- Short diagnostic result: `s01 mean accuracy=0.5250 std=0.0464`.
- Finding: even with ijcnn-compatible preprocessing, the current ACRNN runner still mostly predicts class 0 in short tests, for example many folds have `pred_counts=[80, 0]`.
- Interpretation: ijcnn preprocessing was a key missing piece and is now incorporated, but the current TensorFlow ACRNN runner still does not reproduce the paper trend in short tests. Remaining suspects include model-code differences from the actual ACRNN run path, the hacked/partial nature of `ACRNN.py`, the use of the compatibility channel attention implementation, and potentially needing longer training only after resolving prediction collapse.

## Single-Batch and Fold Diagnostics

- Added `ACRNN/ACRNN_ablation_runner.py`.
- Purpose: diagnose whether majority-class collapse comes from a hard model/training graph bug or from full-fold training settings.
- Supported model modes:
  - `cnn_only`
  - `cnn_lstm`
  - `cwa_cnn_lstm`
  - `cnn_lstm_selfatt`
  - `full_acrnn`
- Single-batch overfit test on `s01 valence`, first 40 samples, `300` epochs, dropout disabled:
  - `cnn_only`: final accuracy `1.0000`
  - `cnn_lstm`: final accuracy `1.0000`
  - `cwa_cnn_lstm`: final accuracy `1.0000`
  - `cnn_lstm_selfatt`: final accuracy `1.0000`
  - `full_acrnn`: final accuracy `1.0000`
- Interpretation: CNN, LSTM, channel-wise attention, self-attention, labels, loss, optimizer, and feed path can all memorize a tiny batch. This rules out a simple hard training-graph bug.
- Added fold diagnostic mode to `ACRNN/ACRNN_ablation_runner.py`.
- Fold diagnostic results on `s01 valence`, fold `1/10`, `30` epochs, batch size `10`, no extra load shuffle:
  - `full_acrnn`, dropout disabled: train accuracy `1.0000`, test accuracy `0.8750`, test prediction counts `[36, 44]`
  - `full_acrnn`, dropout enabled, `train_phase=False`: train accuracy `0.5306`, test accuracy `0.4750`, test prediction counts `[80, 0]`
  - `full_acrnn`, dropout enabled, `train_phase=True`: train accuracy `0.5306`, test accuracy `0.4750`, test prediction counts `[80, 0]`
- Interpretation: the current collapse is strongly tied to training with `keep_prob=0.5`; changing batch norm `train_phase` does not fix it. With dropout disabled, the same fold learns quickly and generalizes well in this diagnostic.
- Added `--disable-dropout` to `ACRNN/ACRNN_deap_dat.py` for controlled 10-fold dropout ablation. This is diagnostic and should be reported separately from strict original-parameter reproduction.
- Added `--max-folds` to `ACRNN/ACRNN_deap_dat.py` so long diagnostic runs can be split safely instead of losing all results on timeout.
- Dropout-disabled 3-fold diagnostic:
  - command: `D:\Ana\Scripts\conda.exe run --live-stream -n acrnn_tf1 python -u ACRNN\ACRNN_deap_dat.py --subjects s01 --dimension valence --folds 10 --max-folds 3 --epochs 30 --batch-size 10 --no-load-shuffle --disable-dropout --result-root result_dropout_disabled_30ep_3fold`
  - fold 1 final accuracy `0.9375`, prediction counts `[41, 39]`
  - fold 2 final accuracy `0.9125`, prediction counts `[38, 42]`
  - fold 3 final accuracy `0.8375`, prediction counts `[38, 42]`
  - mean accuracy over first 3 folds: `0.8958`, std `0.0425`
  - summary CSV: `result_dropout_disabled_30ep_3fold/valence/s01_summary.csv`
- Interpretation: disabling dropout removes the majority-class collapse in the first 3 folds and produces strong fold-level performance. The full 10-fold run should be executed in smaller chunks or after adding resumable per-fold output.
- Added `--fold-start` and `--fold-end` to `ACRNN/ACRNN_deap_dat.py` for resumable fold ranges.
- Completed `s01 valence`, 10-fold, 30 epochs, dropout disabled, no extra load shuffle, split across three runs:
  - folds 1-3: `result_dropout_disabled_30ep_3fold/valence/s01_summary.csv`
  - folds 4-6: `result_dropout_disabled_30ep_folds4_6/valence/s01_summary.csv`
  - folds 7-10: `result_dropout_disabled_30ep_folds7_10/valence/s01_summary.csv`
- Final per-fold accuracies:
  - fold 1: `0.9375`
  - fold 2: `0.9125`
  - fold 3: `0.8375`
  - fold 4: `0.9000`
  - fold 5: `0.9125`
  - fold 6: `0.8875`
  - fold 7: `0.9125`
  - fold 8: `0.9375`
  - fold 9: `0.9375`
  - fold 10: `0.9250`
- Full 10-fold mean accuracy: `0.9100`, std `0.0289`.
- Interpretation: for `s01 valence`, the DEAP `.dat` -> ijcnn-style preprocessing -> ACRNN runner path can produce strong 10-fold performance when dropout is disabled. This is a diagnostic ablation result, not a strict original-parameter reproduction, because original/public code uses `keep_prob=0.5`.
- Completed `s01 arousal`, 10-fold, 30 epochs, dropout disabled, no extra load shuffle, split across three runs:
  - folds 1-3: `result_dropout_disabled_30ep_arousal_folds1_3/arousal/s01_summary.csv`
  - folds 4-6: `result_dropout_disabled_30ep_arousal_folds4_6/arousal/s01_summary.csv`
  - folds 7-10: `result_dropout_disabled_30ep_arousal_folds7_10/arousal/s01_summary.csv`
- Final per-fold arousal accuracies:
  - fold 1: `0.9500`
  - fold 2: `0.9125`
  - fold 3: `0.9625`
  - fold 4: `0.9125`
  - fold 5: `0.9375`
  - fold 6: `0.9000`
  - fold 7: `0.9625`
  - fold 8: `0.9625`
  - fold 9: `0.9625`
  - fold 10: `0.9000`
- Full 10-fold arousal mean accuracy: `0.9362`, std `0.0259`.
- Current `s01` dropout-disabled diagnostic summary:
  - valence: `0.9100 +/- 0.0289`
  - arousal: `0.9362 +/- 0.0259`
- Interpretation: both valence and arousal show strong, non-collapsed subject-dependent 10-fold performance for `s01` when dropout is disabled. The key unresolved reproduction issue remains why the public `keep_prob=0.5` setting collapses in this environment.
- Added `--train-keep-prob` to `ACRNN/ACRNN_deap_dat.py` for dropout-strength scanning without changing model code.
- Dropout keep-prob scan on `s01 valence`, fold `1/10`, 30 epochs, no extra load shuffle:
  - `keep_prob=0.5`: final accuracy `0.4750`, best accuracy `0.4875`, prediction counts `[80, 0]`
  - `keep_prob=0.7`: final accuracy `0.8625`, best accuracy `0.8625`, prediction counts `[39, 41]`
  - `keep_prob=0.8`: final accuracy `0.8875`, best accuracy `0.9375`, prediction counts `[29, 51]`
  - `keep_prob=0.9`: final accuracy `0.8750`, best accuracy `0.9125`, prediction counts `[42, 38]`
  - `keep_prob=1.0`: final accuracy `0.9375`, best accuracy `0.9375`, prediction counts `[41, 39]`
- Interpretation: collapse is not caused by dropout as a concept, but by the original/public `keep_prob=0.5` being too strong for this reconstructed TF1/preprocessing path. Milder dropout values recover learning, and disabling dropout remains the strongest/cleanest diagnostic setting so far.
- Completed `s01 valence`, `keep_prob=0.8`, 10-fold, 30 epochs, no extra load shuffle:
  - folds 1-3: `result_keep_prob_08_30ep_valence_folds1_3/valence/s01_summary.csv`
  - folds 4-6: `result_keep_prob_08_30ep_valence_folds4_6/valence/s01_summary.csv`
  - folds 7-10: `result_keep_prob_08_30ep_valence_folds7_10/valence/s01_summary.csv`
- Final per-fold accuracies with `keep_prob=0.8`:
  - fold 1: `0.9500`
  - fold 2: `0.8750`
  - fold 3: `0.9625`
  - fold 4: `0.9125`
  - fold 5: `0.9750`
  - fold 6: `0.9625`
  - fold 7: `0.8875`
  - fold 8: `0.9625`
  - fold 9: `0.9375`
  - fold 10: `0.9250`
- Full 10-fold `keep_prob=0.8` final mean accuracy: `0.9350`, std `0.0325`.
- Full 10-fold `keep_prob=0.8` best-epoch mean accuracy: `0.9550`, std `0.0195`.
- Interpretation: `keep_prob=0.8` is currently the strongest compromise setting for `s01 valence`: it preserves dropout, avoids collapse, and outperforms the no-dropout final-epoch diagnostic on this subject.
- Completed `s01 arousal`, `keep_prob=0.8`, 10-fold, 30 epochs, no extra load shuffle:
  - folds 1-3: `result_keep_prob_08_30ep_arousal_folds1_3/arousal/s01_summary.csv`
  - folds 4-6: `result_keep_prob_08_30ep_arousal_folds4_6/arousal/s01_summary.csv`
  - folds 7-10: `result_keep_prob_08_30ep_arousal_folds7_10/arousal/s01_summary.csv`
- Final per-fold arousal accuracies with `keep_prob=0.8`:
  - fold 1: `0.8875`
  - fold 2: `0.9500`
  - fold 3: `0.9500`
  - fold 4: `0.9000`
  - fold 5: `0.9375`
  - fold 6: `0.9375`
  - fold 7: `0.9500`
  - fold 8: `0.8125`
  - fold 9: `0.9375`
  - fold 10: `0.9500`
- Full 10-fold arousal `keep_prob=0.8` final mean accuracy: `0.9212`, std `0.0419`.
- Full 10-fold arousal `keep_prob=0.8` best-epoch mean accuracy: `0.9475`, std `0.0208`.
- Current `s01` `keep_prob=0.8` diagnostic summary:
  - valence final: `0.9350 +/- 0.0325`; best epoch: `0.9550 +/- 0.0195`
  - arousal final: `0.9212 +/- 0.0419`; best epoch: `0.9475 +/- 0.0208`
- Interpretation: `keep_prob=0.8` preserves dropout and avoids the majority-class collapse on both `s01` valence and arousal. It should be treated as a diagnostic correction to the public-code dropout setting, not as the unchanged original-parameter reproduction.
- Completed `s02 valence`, `keep_prob=0.8`, 10-fold, 30 epochs, no extra load shuffle:
  - final mean accuracy: `0.8062`, std `0.0372`
  - best-epoch mean accuracy: `0.8225`, std `0.0450`
  - result roots: `result_keep_prob_08_30ep_s02_valence_folds1_3`, `result_keep_prob_08_30ep_s02_valence_folds4_6`, `result_keep_prob_08_30ep_s02_valence_folds7_10`
- Completed `s02 arousal`, `keep_prob=0.8`, 10-fold, 30 epochs, no extra load shuffle:
  - final per-fold accuracies: `0.8500`, `0.7875`, `0.8000`, `0.7875`, `0.7750`, `0.7500`, `0.8000`, `0.7250`, `0.7125`, `0.8125`
  - final mean accuracy: `0.7800`, std `0.0392`
  - best-epoch mean accuracy: `0.8100`, std `0.0357`
  - result roots: `result_keep_prob_08_30ep_s02_arousal_fold1_retry`, `result_keep_prob_08_30ep_s02_arousal_fold2`, `result_keep_prob_08_30ep_s02_arousal_fold3`, `result_keep_prob_08_30ep_s02_arousal_fold4`, `result_keep_prob_08_30ep_s02_arousal_fold5`, `result_keep_prob_08_30ep_s02_arousal_fold6`, `result_keep_prob_08_30ep_s02_arousal_fold7`, `result_keep_prob_08_30ep_s02_arousal_fold8`, `result_keep_prob_08_30ep_s02_arousal_fold9`, `result_keep_prob_08_30ep_s02_arousal_fold10`
- Current `keep_prob=0.8` two-subject diagnostic summary:
  - `s01 valence`: `0.9350 +/- 0.0325`
  - `s01 arousal`: `0.9212 +/- 0.0419`
  - `s02 valence`: `0.8062 +/- 0.0372`
  - `s02 arousal`: `0.7800 +/- 0.0392`
- Completed `s03 valence`, `keep_prob=0.8`, 10-fold, 30 epochs, no extra load shuffle:
  - final per-fold accuracies: `0.8875`, `0.9375`, `0.9500`, `0.8875`, `0.9000`, `0.9000`, `0.9125`, `0.9750`, `0.9375`, `0.9500`
  - final mean accuracy: `0.9238`, std `0.0288`
  - best-epoch mean accuracy: `0.9350`, std `0.0242`
  - result roots: `result_keep_prob_08_30ep_s03_valence_folds1_3`, `result_keep_prob_08_30ep_s03_valence_folds4_6`, `result_keep_prob_08_30ep_s03_valence_folds7_10`
- Current `keep_prob=0.8` diagnostic summary:
  - `s01 valence`: `0.9350 +/- 0.0325`
  - `s01 arousal`: `0.9212 +/- 0.0419`
  - `s02 valence`: `0.8062 +/- 0.0372`
  - `s02 arousal`: `0.7800 +/- 0.0392`
  - `s03 valence`: `0.9238 +/- 0.0288`

## Constraints

- Do not treat H5 adaptation results as strict reproduction results.
- Do not delete files; archive old material under `archive_h5_adaptation/`.
- Do not modify the original model structure, loss, optimizer, window length, or training logic unless required only for runtime compatibility.

## Latest Completed Run: s03 arousal

- Completed `s03 arousal`, `keep_prob=0.8`, 10 folds, 30 epochs, no extra load shuffle.
- Final 10-fold mean accuracy: `0.9462`, std `0.0280`.
- Best-epoch 10-fold mean accuracy: `0.9525`, std `0.0236`.
- Per-fold final accuracies: `0.9625`, `0.9375`, `0.9875`, `0.9625`, `0.9750`, `0.9000`, `0.9500`, `0.9125`, `0.9125`, `0.9625`.
- Result roots: `result_keep_prob_08_30ep_s03_arousal_folds1_3`, `result_keep_prob_08_30ep_s03_arousal_folds4_6`, `result_keep_prob_08_30ep_s03_arousal_folds7_10`.
- Combined fold CSV: `result_summaries/s03_arousal_keep_prob_08_30ep_10fold.csv`.
- Updated summary CSVs: `result_summaries/all_fold_results.csv`, `result_summaries/group_summary.csv`.
- Caveat: `s03 arousal` is strongly class-imbalanced (`[640, 160]`), so use confusion matrices/prediction counts together with accuracy.

## Non-destructive result directory cleanup

- Created `experiments/` with subdirectories: `dropout_disabled`, `keep_prob_08`, `keep_prob_scan`, `reference_short`, and `old_or_misc_results`.
- Moved root `result_*` experiment folders into `experiments/` by category. No files were deleted.
- Left core project directories and files in place: `ACRNN/`, `ijcnn-master/`, `data_preprocessed_python/`, `deap_shuffled_data_3s/`, `deap_pre_process_from_dat.py`, `inspect_deap_dat.py`, `summarize_acrnn_results.py`, `PROJECT_STATUS.md`, and `RUNNING_NOTES.md`.
- `result_summaries/` remains at the project root as the unified summary output directory.
- No uncertain `result_*` folders were found; `experiments/old_or_misc_results/` is currently empty.
- Updated `summarize_acrnn_results.py` to scan `experiments/**/*_summary.csv` and write:
  - `result_summaries/all_experiment_folds.csv`
  - `result_summaries/per_subject_summary.csv`
  - `result_summaries/overall_summary.csv`
  - `result_summaries/result_group_summary.csv`
- Also keeps compatibility aliases: `all_fold_results.csv` and `group_summary.csv`.
- The requested plain `conda run -n acrnn_tf1 python summarize_acrnn_results.py` command hit a Conda stdout GBK encoding error. The summary itself wrote files successfully, and rerunning with `--live-stream` completed successfully.
- Added `EXPERIMENT_INDEX.md`.

## Summary command check

- Re-ran the requested exact command after cleanup:
  - `D:\Ana\Scripts\conda.exe run -n acrnn_tf1 python summarize_acrnn_results.py`
- It failed in Conda's stdout wrapper with `UnicodeEncodeError: 'gbk' codec can't encode character '\ufffd'`.
- This is the same Conda/terminal encoding issue observed during cleanup. The script itself had already written the summary CSVs successfully, and `conda run --live-stream` succeeds.
- No additional project files or experiment results were moved after this failure.
