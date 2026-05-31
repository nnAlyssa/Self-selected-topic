# python3
"""
Runnable DEAP .dat -> .pkl training entry for the original ACRNN model line.

This file keeps the original ACRNN model structure but fixes local runtime
issues in ACRNN.py: hard-coded paths, invalid hyphenated import, undefined
subject list, and Python/TensorFlow compatibility details.
"""

import argparse
import csv
import importlib.util
import os
from pathlib import Path
import pickle
import time

import numpy as np
import tensorflow as tf

from cnn_class import cnn
from Attention.disan import multi_dimensional_attention


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "deap_shuffled_data_3s"
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "result_att_cnn_rnn_att_deap"
DEFAULT_RAW_DATA_ROOT = PROJECT_ROOT / "data_preprocessed_python"

WINDOW_SIZE = 384
BASELINE_SIZE = 384
TRIAL_SIGNAL_SIZE = 7680
WINDOWS_PER_TRIAL = TRIAL_SIGNAL_SIZE // WINDOW_SIZE
SAMPLING_RATE = 128
N_CHANNEL = 32
INPUT_CHANNEL_NUM = 1
INPUT_HEIGHT = 32
INPUT_WIDTH = 384
NUM_LABELS = 2

KERNEL_HEIGHT_1ST = 32
KERNEL_WIDTH_1ST = 45
KERNEL_STRIDE = 1
CONV_CHANNEL_NUM = 40

POOLING_HEIGHT_1ST = 1
POOLING_WIDTH_1ST = 75
POOLING_STRIDE_1ST = 10

N_HIDDEN_STATE = 64
NUM_TIMESTEP = 1
LEARNING_RATE = 1e-4
DROPOUT_PROB = 0.5
PADDING = "VALID"

DEAP_SUBJECTS = [f"s{index:02d}" for index in range(1, 33)]
DIMENSION_TO_LABEL_INDEX = {
    "valence": 0,
    "arousal": 1,
}


def _load_channel_wise_attention():
    module_path = SCRIPT_DIR / "Attention" / "channel_wise_attention_compat.py"
    spec = importlib.util.spec_from_file_location("channel_wise_attention_module", str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.channel_wise_attention


channel_wise_attention = _load_channel_wise_attention()


def one_hot(labels):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    result = np.zeros((labels.shape[0], NUM_LABELS), dtype=np.float32)
    result[np.arange(labels.shape[0]), labels] = 1.0
    return result


def load_subject_pickles(subject, dimension, data_root):
    data_dir = Path(data_root) / f"yes_{dimension}"
    data_path = data_dir / f"{subject}.mat_win_384_rnn_dataset.pkl"
    label_path = data_dir / f"{subject}.mat_win_384_labels.pkl"
    with data_path.open("rb") as f:
        datasets = pickle.load(f)
    with label_path.open("rb") as f:
        labels = pickle.load(f)

    datasets = np.asarray(datasets, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if datasets.shape[1:] != (WINDOW_SIZE, N_CHANNEL):
        raise ValueError(f"{data_path} expected shape (N, 384, 32), got {datasets.shape}")
    if labels.shape[0] != datasets.shape[0]:
        raise ValueError(f"{label_path} label count {labels.shape[0]} does not match data {datasets.shape[0]}")

    datasets = np.transpose(datasets, (0, 2, 1))
    datasets = datasets[..., np.newaxis]
    return datasets, one_hot(labels), labels


def find_subject_dat(subject, raw_data_root):
    root = Path(raw_data_root)
    candidates = [
        root / f"{subject}.dat",
        root / "data_preprocessed_python" / f"{subject}.dat",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find raw DEAP file for {subject}. Searched:\n{searched}")


def feature_normalize(data):
    data = np.asarray(data, dtype=np.float32).copy()
    nonzero = data.nonzero()
    if len(nonzero[0]) == 0:
        return data
    values = data[nonzero]
    mean = values.mean()
    sigma = values.std()
    if sigma < 1e-8:
        sigma = 1.0
    data[nonzero] = (values - mean) / sigma
    return data


def norm_dataset(dataset_1d):
    norm_dataset_1d = np.zeros((dataset_1d.shape[0], N_CHANNEL), dtype=np.float32)
    for i in range(dataset_1d.shape[0]):
        norm_dataset_1d[i] = feature_normalize(dataset_1d[i])
    return norm_dataset_1d


def iter_window_bounds(data, size):
    start = 0
    while (start + size) < data.shape[0]:
        yield int(start), int(start + size)
        start += size


def baseline_correct_trial(trial):
    base_signal = (
        trial[0:128, 0:N_CHANNEL]
        + trial[128:256, 0:N_CHANNEL]
        + trial[256:384, 0:N_CHANNEL]
    ) / 3.0
    signal = trial[BASELINE_SIZE:8064, 0:N_CHANNEL].copy()
    for i in range(60):
        signal[i * SAMPLING_RATE:(i + 1) * SAMPLING_RATE, 0:N_CHANNEL] -= base_signal
    return norm_dataset(signal)


def split_trial_into_windows(signal):
    segments = []
    for start, end in iter_window_bounds(signal, WINDOW_SIZE):
        if len(signal[start:end]) == WINDOW_SIZE:
            if start == 0:
                segments.append(signal[start:end])
                segments.append(signal[start:end])
            else:
                segments.append(signal[start:end])
    if len(segments) != WINDOWS_PER_TRIAL:
        raise ValueError(f"Expected {WINDOWS_PER_TRIAL} windows, got {len(segments)}")
    return np.ascontiguousarray(np.stack(segments, axis=0), dtype=np.float32)


def load_subject_trials(subject, dimension, raw_data_root):
    file_path = find_subject_dat(subject, raw_data_root)
    with file_path.open("rb") as f:
        subject_data = pickle.load(f, encoding="latin1")
    data = np.asarray(subject_data["data"], dtype=np.float32).transpose(0, 2, 1)
    labels = np.asarray(subject_data["labels"], dtype=np.float32)
    if data.shape != (40, 8064, 40):
        raise ValueError(f"{file_path} expected transposed data shape (40, 8064, 40), got {data.shape}")
    if labels.shape != (40, 4):
        raise ValueError(f"{file_path} expected labels shape (40, 4), got {labels.shape}")

    label_index = DIMENSION_TO_LABEL_INDEX[dimension]
    trial_datasets = []
    trial_labels = []
    raw_trial_labels = []
    for trial_index in range(data.shape[0]):
        corrected_signal = baseline_correct_trial(data[trial_index])
        windows = split_trial_into_windows(corrected_signal)
        windows = np.transpose(windows, (0, 2, 1))[..., np.newaxis]
        label = 1 if labels[trial_index, label_index] > 5.0 else 0
        trial_datasets.append(windows)
        trial_labels.append(one_hot(np.full(WINDOWS_PER_TRIAL, label, dtype=np.int64)))
        raw_trial_labels.append(label)

    return (
        np.asarray(trial_datasets, dtype=np.float32),
        np.asarray(trial_labels, dtype=np.float32),
        np.asarray(raw_trial_labels, dtype=np.int64),
    )


def shuffle_data(datasets, labels, raw_labels, seed):
    rng = np.random.RandomState(seed)
    indexes = np.arange(labels.shape[0])
    rng.shuffle(indexes)
    return datasets[indexes], labels[indexes], raw_labels[indexes]


def iter_batches(x, y, batch_size, shuffle=False):
    indexes = np.arange(x.shape[0])
    if shuffle:
        np.random.shuffle(indexes)
    full_count = (x.shape[0] // batch_size) * batch_size
    for start in range(0, full_count, batch_size):
        batch_indexes = indexes[start:start + batch_size]
        yield x[batch_indexes], y[batch_indexes]


def build_graph(use_channel_attention=True, use_self_attention=True):
    tf.reset_default_graph()
    cnn_2d = cnn(padding=PADDING)

    x_ph = tf.placeholder(tf.float32, shape=[None, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL_NUM], name="X")
    y_ph = tf.placeholder(tf.float32, shape=[None, NUM_LABELS], name="Y")
    train_phase = tf.placeholder(tf.bool, name="train_phase")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    if use_channel_attention:
        x_1 = tf.transpose(x_ph, [0, 3, 2, 1])
        conv = channel_wise_attention(
            x_1,
            1,
            WINDOW_SIZE,
            N_CHANNEL,
            weight_decay=0.00004,
            scope="",
            reuse=None,
        )
        conv_1 = tf.transpose(conv, [0, 3, 2, 1])
    else:
        conv_1 = x_ph

    conv_1 = cnn_2d.apply_conv2d(
        conv_1,
        KERNEL_HEIGHT_1ST,
        KERNEL_WIDTH_1ST,
        INPUT_CHANNEL_NUM,
        CONV_CHANNEL_NUM,
        KERNEL_STRIDE,
        train_phase,
    )
    pool_1 = cnn_2d.apply_max_pooling(
        conv_1,
        POOLING_HEIGHT_1ST,
        POOLING_WIDTH_1ST,
        POOLING_STRIDE_1ST,
    )
    pool_1_shape = pool_1.get_shape().as_list()
    pool1_flat = tf.reshape(pool_1, [-1, pool_1_shape[1] * pool_1_shape[2] * pool_1_shape[3]])
    fc_drop = tf.nn.dropout(pool1_flat, keep_prob)

    lstm_in = tf.reshape(fc_drop, [-1, NUM_TIMESTEP, pool_1_shape[1] * pool_1_shape[2] * pool_1_shape[3]])
    cells = []
    for _ in range(2):
        cell = tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_STATE, forget_bias=1.0, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        cells.append(cell)
    lstm_cell = tf.contrib.rnn.MultiRNNCell(cells)
    init_state = lstm_cell.zero_state(tf.shape(lstm_in)[0], dtype=tf.float32)
    rnn_op, _ = tf.nn.dynamic_rnn(lstm_cell, lstm_in, initial_state=init_state, time_major=False)

    with tf.name_scope("Attention_layer"):
        if use_self_attention:
            rep_mask = tf.fill(tf.stack([tf.shape(rnn_op)[0], NUM_TIMESTEP]), True)
            attention_op = multi_dimensional_attention(
                rnn_op,
                rep_mask,
                scope=None,
                keep_prob=1.0,
                is_train=None,
                wd=0.0,
                activation="elu",
                tensor_dict=None,
                name=None,
            )
        else:
            attention_op = rnn_op[:, -1, :]
        attention_drop = tf.nn.dropout(attention_op, keep_prob)
        logits = cnn_2d.apply_readout(attention_drop, rnn_op.shape[2].value, NUM_LABELS)

    y_prob = tf.nn.softmax(logits, name="y_prob")
    y_pred = tf.argmax(y_prob, 1, name="y_pred")
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_ph), name="loss")
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(logits), 1), tf.argmax(y_ph, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

    return {
        "X": x_ph,
        "Y": y_ph,
        "train_phase": train_phase,
        "keep_prob": keep_prob,
        "optimizer": optimizer,
        "cost": cost,
        "accuracy": accuracy,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


def evaluate(session, graph, x, y, batch_size):
    accuracies = []
    losses = []
    predictions = []
    true_labels = []
    for batch_x, batch_y in iter_batches(x, y, batch_size, shuffle=False):
        acc, loss, pred = session.run(
            [graph["accuracy"], graph["cost"], graph["y_pred"]],
            feed_dict={
                graph["X"]: batch_x,
                graph["Y"]: batch_y,
                graph["keep_prob"]: 1.0,
                graph["train_phase"]: False,
            },
        )
        accuracies.append(acc)
        losses.append(loss)
        predictions.append(pred)
        true_labels.append(np.argmax(batch_y, axis=1))

    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    return {
        "accuracy": float(np.mean(accuracies)),
        "loss": float(np.mean(losses)),
        "predictions": predictions,
        "true_labels": true_labels,
    }


def confusion_matrix(true_labels, predictions):
    matrix = np.zeros((NUM_LABELS, NUM_LABELS), dtype=np.int64)
    for true_label, pred_label in zip(true_labels, predictions):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def standardize_by_train(train_x, test_x):
    mean = np.mean(train_x, dtype=np.float64)
    std = np.std(train_x, dtype=np.float64)
    if std < 1e-8:
        std = 1.0
    train_x = ((train_x - mean) / std).astype(np.float32)
    test_x = ((test_x - mean) / std).astype(np.float32)
    return train_x, test_x, float(mean), float(std)


def run_fold(
    train_x,
    train_y,
    test_x,
    test_y,
    epochs,
    batch_size,
    train_phase_value,
    disable_dropout,
    train_keep_prob,
    use_channel_attention,
    use_self_attention,
):
    graph = build_graph(
        use_channel_attention=use_channel_attention,
        use_self_attention=use_self_attention,
    )
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    best = {
        "epoch": 0,
        "accuracy": -1.0,
        "loss": None,
        "predictions": None,
        "true_labels": None,
    }

    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            losses = []
            for batch_x, batch_y in iter_batches(train_x, train_y, batch_size, shuffle=True):
                _, loss = session.run(
                    [graph["optimizer"], graph["cost"]],
                    feed_dict={
                        graph["X"]: batch_x,
                        graph["Y"]: batch_y,
                        graph["keep_prob"]: 1.0 if disable_dropout else train_keep_prob,
                        graph["train_phase"]: train_phase_value,
                    },
                )
                losses.append(loss)
            eval_result = evaluate(session, graph, test_x, test_y, batch_size)
            test_acc = eval_result["accuracy"]
            test_loss = eval_result["loss"]
            if test_acc > best["accuracy"]:
                best = {
                    "epoch": epoch + 1,
                    "accuracy": test_acc,
                    "loss": test_loss,
                    "predictions": eval_result["predictions"],
                    "true_labels": eval_result["true_labels"],
                }
            print(
                f"    epoch {epoch + 1}/{epochs}: "
                f"train_loss={float(np.mean(losses)):.6f}, "
                f"test_loss={test_loss:.6f}, test_acc={test_acc:.4f}"
            )
        final_result = evaluate(session, graph, test_x, test_y, batch_size)
    return final_result, best


def cross_validate_subject(
    subject,
    dimension,
    data_root,
    raw_data_root,
    folds,
    epochs,
    batch_size,
    seed,
    standardize,
    train_phase_value,
    load_shuffle,
    disable_dropout,
    max_folds,
    fold_start,
    fold_end,
    train_keep_prob,
    use_channel_attention,
    use_self_attention,
    split_protocol,
):
    if split_protocol == "sample":
        datasets, labels, raw_labels = load_subject_pickles(subject, dimension, data_root)
        if load_shuffle:
            datasets, labels, raw_labels = shuffle_data(datasets, labels, raw_labels, seed)
        split_indexes = np.arange(datasets.shape[0])
        fold_size = datasets.shape[0] // folds
        print(
            f"{subject} {dimension}: split=sample, data={datasets.shape}, "
            f"labels={labels.shape}, counts={np.bincount(raw_labels, minlength=2).tolist()}"
        )
    elif split_protocol == "trial":
        trial_datasets, trial_labels, raw_trial_labels = load_subject_trials(subject, dimension, raw_data_root)
        split_indexes = np.arange(trial_datasets.shape[0])
        if load_shuffle:
            rng = np.random.RandomState(seed)
            rng.shuffle(split_indexes)
        fold_size = trial_datasets.shape[0] // folds
        print(
            f"{subject} {dimension}: split=trial, trials={trial_datasets.shape}, "
            f"trial_label_counts={np.bincount(raw_trial_labels, minlength=2).tolist()}"
        )
    else:
        raise ValueError(f"Unknown split protocol: {split_protocol}")

    fold_results = []
    if fold_start < 1 or fold_start > folds:
        raise ValueError("--fold-start must be between 1 and --folds")
    if fold_end is not None and (fold_end < fold_start or fold_end > folds):
        raise ValueError("--fold-end must be between --fold-start and --folds")
    end_fold = folds if fold_end is None else fold_end
    if max_folds is not None:
        end_fold = min(end_fold, fold_start + max_folds - 1)
    for fold in range(fold_start - 1, end_fold):
        start = fold * fold_size
        end = start + fold_size
        test_indexes = split_indexes[start:end]
        train_indexes = np.setdiff1d(split_indexes, test_indexes, assume_unique=True)
        if split_protocol == "sample":
            train_x, train_y = datasets[train_indexes], labels[train_indexes]
            test_x, test_y = datasets[test_indexes], labels[test_indexes]
            train_units = train_x.shape[0]
            test_units = test_x.shape[0]
        else:
            train_x = trial_datasets[train_indexes].reshape(-1, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL_NUM)
            train_y = trial_labels[train_indexes].reshape(-1, NUM_LABELS)
            test_x = trial_datasets[test_indexes].reshape(-1, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL_NUM)
            test_y = trial_labels[test_indexes].reshape(-1, NUM_LABELS)
            train_units = len(train_indexes)
            test_units = len(test_indexes)
        standardize_mean = ""
        standardize_std = ""
        if standardize:
            train_x, test_x, standardize_mean, standardize_std = standardize_by_train(train_x, test_x)
        train_counts = np.bincount(np.argmax(train_y, axis=1), minlength=NUM_LABELS)
        test_counts = np.bincount(np.argmax(test_y, axis=1), minlength=NUM_LABELS)

        print(
            f"  fold {fold + 1}/{folds}: train={train_x.shape[0]}, test={test_x.shape[0]}, "
            f"train_{split_protocol}s={train_units}, test_{split_protocol}s={test_units}, "
            f"train_counts={train_counts.tolist()}, test_counts={test_counts.tolist()}"
        )
        final_result, best = run_fold(
            train_x,
            train_y,
            test_x,
            test_y,
            epochs,
            batch_size,
            train_phase_value,
            disable_dropout,
            train_keep_prob,
            use_channel_attention,
            use_self_attention,
        )
        final_counts = np.bincount(final_result["predictions"], minlength=NUM_LABELS)
        final_cm = confusion_matrix(final_result["true_labels"], final_result["predictions"])
        best_counts = np.bincount(best["predictions"], minlength=NUM_LABELS)
        best_cm = confusion_matrix(best["true_labels"], best["predictions"])
        print(
            f"  fold {fold + 1}/{folds} final: "
            f"loss={final_result['loss']:.6f}, accuracy={final_result['accuracy']:.4f}, "
            f"pred_counts={final_counts.tolist()}"
        )
        print(f"  fold {fold + 1}/{folds} final confusion_matrix={final_cm.tolist()}")
        print(
            f"  fold {fold + 1}/{folds} best: "
            f"epoch={best['epoch']}, loss={best['loss']:.6f}, "
            f"accuracy={best['accuracy']:.4f}, pred_counts={best_counts.tolist()}"
        )
        print(f"  fold {fold + 1}/{folds} best confusion_matrix={best_cm.tolist()}")
        fold_results.append({
            "fold": fold + 1,
            "test_loss": final_result["loss"],
            "test_accuracy": final_result["accuracy"],
            "true_count_0": int(test_counts[0]),
            "true_count_1": int(test_counts[1]),
            "pred_count_0": int(final_counts[0]),
            "pred_count_1": int(final_counts[1]),
            "cm_00": int(final_cm[0, 0]),
            "cm_01": int(final_cm[0, 1]),
            "cm_10": int(final_cm[1, 0]),
            "cm_11": int(final_cm[1, 1]),
            "best_epoch": int(best["epoch"]),
            "best_test_loss": best["loss"],
            "best_test_accuracy": best["accuracy"],
            "best_pred_count_0": int(best_counts[0]),
            "best_pred_count_1": int(best_counts[1]),
            "best_cm_00": int(best_cm[0, 0]),
            "best_cm_01": int(best_cm[0, 1]),
            "best_cm_10": int(best_cm[1, 0]),
            "best_cm_11": int(best_cm[1, 1]),
            "standardize": int(standardize),
            "standardize_mean": standardize_mean,
            "standardize_std": standardize_std,
            "train_phase_train": int(train_phase_value),
            "disable_dropout": int(disable_dropout),
            "train_keep_prob": 1.0 if disable_dropout else train_keep_prob,
            "use_channel_attention": int(use_channel_attention),
            "use_self_attention": int(use_self_attention),
            "split_protocol": split_protocol,
            "train_split_units": int(train_units),
            "test_split_units": int(test_units),
        })
    return fold_results


def write_summary(result_root, dimension, subject, fold_results):
    output_dir = Path(result_root) / dimension
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{subject}_summary.csv"
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_results[0].keys()))
        writer.writeheader()
        writer.writerows(fold_results)
    return output_path


def parse_subjects(subject_arg):
    if subject_arg == "all":
        return DEAP_SUBJECTS
    return [item.strip() for item in subject_arg.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="s01", help="all, or comma-separated IDs such as s01,s02")
    parser.add_argument("--dimension", default="valence", choices=["valence", "arousal"])
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--raw-data-root", default=str(DEFAULT_RAW_DATA_ROOT))
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--split-protocol", default="sample", choices=["sample", "trial"])
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--standardize", action="store_true", help="Standardize each fold using train mean/std")
    parser.add_argument(
        "--train-phase-train",
        action="store_true",
        help="Feed train_phase=True during optimizer steps so batch norm uses training mode",
    )
    parser.add_argument(
        "--no-load-shuffle",
        action="store_true",
        help="Do not shuffle after loading .pkl files; use the order generated by preprocessing",
    )
    parser.add_argument(
        "--disable-dropout",
        action="store_true",
        help="Diagnostic mode: feed keep_prob=1.0 during training",
    )
    parser.add_argument(
        "--train-keep-prob",
        type=float,
        default=1.0 - DROPOUT_PROB,
        help="Diagnostic mode: keep_prob fed during training when dropout is enabled",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Diagnostic mode: run only the first N folds",
    )
    parser.add_argument(
        "--fold-start",
        type=int,
        default=1,
        help="Diagnostic mode: first 1-based fold to run",
    )
    parser.add_argument(
        "--fold-end",
        type=int,
        default=None,
        help="Diagnostic mode: last 1-based fold to run",
    )
    parser.add_argument(
        "--no-channel-attention",
        action="store_true",
        help="Ablation: remove the channel-wise attention module before CNN",
    )
    parser.add_argument(
        "--no-self-attention",
        action="store_true",
        help="Ablation: replace self-attention after LSTM with the last LSTM output",
    )
    args = parser.parse_args()

    print("started:", time.asctime())
    use_channel_attention = not args.no_channel_attention
    use_self_attention = not args.no_self_attention
    print(
        "model:",
        f"use_channel_attention={int(use_channel_attention)}",
        f"use_self_attention={int(use_self_attention)}",
    )
    subjects = parse_subjects(args.subjects)
    for subject in subjects:
        fold_results = cross_validate_subject(
            subject,
            args.dimension,
            args.data_root,
            args.raw_data_root,
            args.folds,
            args.epochs,
            args.batch_size,
            args.seed,
            args.standardize,
            args.train_phase_train,
            not args.no_load_shuffle,
            args.disable_dropout,
            args.max_folds,
            args.fold_start,
            args.fold_end,
            args.train_keep_prob,
            use_channel_attention,
            use_self_attention,
            args.split_protocol,
        )
        output_path = write_summary(args.result_root, args.dimension, subject, fold_results)
        accuracies = [item["test_accuracy"] for item in fold_results]
        print(f"{subject} mean accuracy={np.mean(accuracies):.4f} std={np.std(accuracies):.4f}")
        print("summary:", output_path)
    print("finished:", time.asctime())


if __name__ == "__main__":
    main()
