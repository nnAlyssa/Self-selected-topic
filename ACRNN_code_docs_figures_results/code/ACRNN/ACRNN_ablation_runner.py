# python3
"""
Minimal ACRNN ablation runner for overfit and fold diagnostics.

This is intentionally separate from ACRNN.py and ACRNN_deap_dat.py. It is not a
paper-result runner; it exists to identify whether public-code model components
can memorize a tiny fixed batch and whether a real CV fold learns its train set.
"""

import argparse
import importlib.util
from pathlib import Path
import pickle

import numpy as np
import tensorflow as tf

from cnn_class import cnn
from Attention.disan import multi_dimensional_attention


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "deap_shuffled_data_3s"

WINDOW_SIZE = 384
N_CHANNEL = 32
INPUT_CHANNEL_NUM = 1
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
PADDING = "VALID"
MODEL_MODES = [
    "cnn_only",
    "cnn_lstm",
    "cwa_cnn_lstm",
    "cnn_lstm_selfatt",
    "full_acrnn",
]


def _load_channel_wise_attention():
    module_path = SCRIPT_DIR / "Attention" / "channel-wise_attention.py"
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


def load_subject_pickles(subject, dimension, data_root, load_shuffle, seed):
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

    datasets = np.transpose(datasets, (0, 2, 1))[..., np.newaxis]
    if load_shuffle:
        rng = np.random.RandomState(seed)
        indexes = np.arange(labels.shape[0])
        rng.shuffle(indexes)
        datasets = datasets[indexes]
        labels = labels[indexes]
    return datasets, one_hot(labels), labels


def iter_batches(x, y, batch_size, shuffle):
    indexes = np.arange(x.shape[0])
    if shuffle:
        np.random.shuffle(indexes)
    for start in range(0, x.shape[0], batch_size):
        batch_indexes = indexes[start:start + batch_size]
        if batch_indexes.shape[0] == batch_size:
            yield x[batch_indexes], y[batch_indexes]


def confusion_matrix(true_labels, predictions):
    matrix = np.zeros((NUM_LABELS, NUM_LABELS), dtype=np.int64)
    for true_label, pred_label in zip(true_labels, predictions):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def build_graph(model_mode, disable_dropout):
    tf.reset_default_graph()
    cnn_2d = cnn(padding=PADDING)

    x_ph = tf.placeholder(tf.float32, shape=[None, N_CHANNEL, WINDOW_SIZE, INPUT_CHANNEL_NUM], name="X")
    y_ph = tf.placeholder(tf.float32, shape=[None, NUM_LABELS], name="Y")
    train_phase = tf.placeholder(tf.bool, name="train_phase")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    effective_keep_prob = tf.constant(1.0, dtype=tf.float32) if disable_dropout else keep_prob

    conv_input = x_ph
    if model_mode in ("cwa_cnn_lstm", "full_acrnn"):
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
        conv_input = tf.transpose(conv, [0, 3, 2, 1])

    conv_1 = cnn_2d.apply_conv2d(
        conv_input,
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
    feature_size = pool_1_shape[1] * pool_1_shape[2] * pool_1_shape[3]
    pool1_flat = tf.reshape(pool_1, [-1, feature_size])
    fc_drop = tf.nn.dropout(pool1_flat, effective_keep_prob)

    if model_mode == "cnn_only":
        logits_input = fc_drop
        logits_input_size = feature_size
    else:
        lstm_in = tf.reshape(fc_drop, [-1, NUM_TIMESTEP, feature_size])
        cells = []
        for _ in range(2):
            cell = tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_STATE, forget_bias=1.0, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=effective_keep_prob)
            cells.append(cell)
        lstm_cell = tf.contrib.rnn.MultiRNNCell(cells)
        init_state = lstm_cell.zero_state(tf.shape(lstm_in)[0], dtype=tf.float32)
        rnn_op, _ = tf.nn.dynamic_rnn(lstm_cell, lstm_in, initial_state=init_state, time_major=False)

        if model_mode in ("cnn_lstm_selfatt", "full_acrnn"):
            rep_mask = tf.fill(tf.stack([tf.shape(rnn_op)[0], NUM_TIMESTEP]), True)
            logits_input = multi_dimensional_attention(
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
            logits_input = tf.reshape(rnn_op[:, -1, :], [-1, N_HIDDEN_STATE])
        logits_input = tf.nn.dropout(logits_input, effective_keep_prob)
        logits_input_size = N_HIDDEN_STATE

    logits = cnn_2d.apply_readout(logits_input, logits_input_size, NUM_LABELS)
    y_prob = tf.nn.softmax(logits, name="y_prob")
    y_pred = tf.argmax(y_prob, 1, name="y_pred")
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_ph), name="loss")
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_ph, 1), y_pred), tf.float32), name="accuracy")

    return {
        "X": x_ph,
        "Y": y_ph,
        "train_phase": train_phase,
        "keep_prob": keep_prob,
        "optimizer": optimizer,
        "cost": cost,
        "accuracy": accuracy,
        "y_pred": y_pred,
        "trainable_vars": tf.trainable_variables(),
    }


def variable_norms(session, variables):
    values = session.run(variables[:10])
    return [float(np.linalg.norm(value)) for value in values]


def evaluate(session, graph, x, y):
    loss, acc, pred = session.run(
        [graph["cost"], graph["accuracy"], graph["y_pred"]],
        feed_dict={
            graph["X"]: x,
            graph["Y"]: y,
            graph["keep_prob"]: 1.0,
            graph["train_phase"]: False,
        },
    )
    true_labels = np.argmax(y, axis=1)
    pred_counts = np.bincount(pred, minlength=NUM_LABELS)
    true_counts = np.bincount(true_labels, minlength=NUM_LABELS)
    return {
        "loss": float(loss),
        "accuracy": float(acc),
        "predictions": pred,
        "pred_counts": pred_counts,
        "true_counts": true_counts,
        "confusion": confusion_matrix(true_labels, pred),
    }


def print_eval(prefix, epoch, epochs, result):
    print(
        "{} epoch {}/{}: loss={:.6f}, acc={:.4f}, true_counts={}, pred_counts={}, confusion_matrix={}".format(
            prefix,
            epoch,
            epochs,
            result["loss"],
            result["accuracy"],
            result["true_counts"].tolist(),
            result["pred_counts"].tolist(),
            result["confusion"].tolist(),
        )
    )


def run_overfit(args):
    x, y, raw_labels = load_subject_pickles(
        args.subjects.split(",")[0],
        args.dimension,
        args.data_root,
        load_shuffle=not args.no_load_shuffle,
        seed=args.seed,
    )
    x = x[:args.overfit_n]
    y = y[:args.overfit_n]
    raw_labels = raw_labels[:args.overfit_n]

    graph = build_graph(args.model_mode, args.disable_dropout)
    print("model_mode:", args.model_mode)
    print("overfit samples:", x.shape)
    print("label distribution:", np.bincount(raw_labels, minlength=NUM_LABELS).tolist())
    print("trainable variables:")
    for var in graph["trainable_vars"]:
        print("  ", var.name, var.shape.as_list())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    before = None
    after = None
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        before = variable_norms(session, graph["trainable_vars"])
        print("first_10_var_norm_before:", before)
        for epoch in range(1, args.epochs + 1):
            for batch_x, batch_y in iter_batches(x, y, args.batch_size, shuffle=True):
                session.run(
                    graph["optimizer"],
                    feed_dict={
                        graph["X"]: batch_x,
                        graph["Y"]: batch_y,
                        graph["keep_prob"]: 1.0 if args.disable_dropout else 0.5,
                        graph["train_phase"]: True,
                    },
                )
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                result = evaluate(session, graph, x, y)
                print_eval("", epoch, args.epochs, result)
        final_result = evaluate(session, graph, x, y)
        after = variable_norms(session, graph["trainable_vars"])

    deltas = [abs(a - b) for a, b in zip(after, before)]
    print("first_10_var_norm_after:", after)
    print("first_10_var_norm_delta:", deltas)
    can_overfit = final_result["accuracy"] >= 0.95
    suspected_issue = "none" if can_overfit else "cannot overfit tiny batch"
    print("\nmodel_mode | can_overfit | final_acc | final_loss | pred_counts | suspected_issue")
    print(
        "{} | {} | {:.4f} | {:.6f} | {} | {}".format(
            args.model_mode,
            can_overfit,
            final_result["accuracy"],
            final_result["loss"],
            final_result["pred_counts"].tolist(),
            suspected_issue,
        )
    )
    return final_result


def split_fold(x, y, raw_labels, folds, fold_index):
    if folds < 2:
        raise ValueError("--folds must be at least 2 for --fold-diagnostic")
    if fold_index < 1 or fold_index > folds:
        raise ValueError("--fold-index must be between 1 and --folds")
    fold_size = x.shape[0] // folds
    start = (fold_index - 1) * fold_size
    end = start + fold_size
    test_x = x[start:end]
    test_y = y[start:end]
    test_raw = raw_labels[start:end]
    train_x = np.concatenate([x[:start], x[end:]], axis=0)
    train_y = np.concatenate([y[:start], y[end:]], axis=0)
    train_raw = np.concatenate([raw_labels[:start], raw_labels[end:]], axis=0)
    return train_x, train_y, train_raw, test_x, test_y, test_raw


def standardize_by_train(train_x, test_x):
    mean = np.mean(train_x, dtype=np.float64)
    std = np.std(train_x, dtype=np.float64)
    if std < 1e-8:
        std = 1.0
    return (
        ((train_x - mean) / std).astype(np.float32),
        ((test_x - mean) / std).astype(np.float32),
        float(mean),
        float(std),
    )


def run_fold_diagnostic(args):
    x, y, raw_labels = load_subject_pickles(
        args.subjects.split(",")[0],
        args.dimension,
        args.data_root,
        load_shuffle=not args.no_load_shuffle,
        seed=args.seed,
    )
    train_x, train_y, train_raw, test_x, test_y, test_raw = split_fold(
        x,
        y,
        raw_labels,
        args.folds,
        args.fold_index,
    )
    if args.standardize:
        train_x, test_x, mean, std = standardize_by_train(train_x, test_x)
        print("standardize_by_train: mean={:.6f}, std={:.6f}".format(mean, std))

    graph = build_graph(args.model_mode, args.disable_dropout)
    print("model_mode:", args.model_mode)
    print("fold_diagnostic:", True)
    print("subject:", args.subjects.split(",")[0])
    print("dimension:", args.dimension)
    print("fold:", "{}/{}".format(args.fold_index, args.folds))
    print("train samples:", train_x.shape)
    print("test samples:", test_x.shape)
    print("train label distribution:", np.bincount(train_raw, minlength=NUM_LABELS).tolist())
    print("test label distribution:", np.bincount(test_raw, minlength=NUM_LABELS).tolist())
    print("disable_dropout:", args.disable_dropout)
    print("train_phase_value:", bool(args.train_phase_train))
    print("trainable variable count:", len(graph["trainable_vars"]))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    before = None
    after = None
    final_train = None
    final_test = None
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        before = variable_norms(session, graph["trainable_vars"])
        print("first_10_var_norm_before:", before)
        for epoch in range(1, args.epochs + 1):
            for batch_x, batch_y in iter_batches(train_x, train_y, args.batch_size, shuffle=True):
                session.run(
                    graph["optimizer"],
                    feed_dict={
                        graph["X"]: batch_x,
                        graph["Y"]: batch_y,
                        graph["keep_prob"]: 1.0 if args.disable_dropout else 0.5,
                        graph["train_phase"]: bool(args.train_phase_train),
                    },
                )
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                train_result = evaluate(session, graph, train_x, train_y)
                test_result = evaluate(session, graph, test_x, test_y)
                print_eval("train", epoch, args.epochs, train_result)
                print_eval("test ", epoch, args.epochs, test_result)
        final_train = evaluate(session, graph, train_x, train_y)
        final_test = evaluate(session, graph, test_x, test_y)
        after = variable_norms(session, graph["trainable_vars"])

    deltas = [abs(a - b) for a, b in zip(after, before)]
    print("first_10_var_norm_after:", after)
    print("first_10_var_norm_delta:", deltas)
    train_learns = final_train["accuracy"] >= 0.90
    suspected_issue = "generalization/protocol" if train_learns else "train-fold learning failure"
    print("\nmodel_mode | train_acc | test_acc | train_loss | test_loss | test_pred_counts | suspected_issue")
    print(
        "{} | {:.4f} | {:.4f} | {:.6f} | {:.6f} | {} | {}".format(
            args.model_mode,
            final_train["accuracy"],
            final_test["accuracy"],
            final_train["loss"],
            final_test["loss"],
            final_test["pred_counts"].tolist(),
            suspected_issue,
        )
    )
    return final_train, final_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="s01")
    parser.add_argument("--dimension", default="valence", choices=["valence", "arousal"])
    parser.add_argument("--model-mode", required=True, choices=MODEL_MODES)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-load-shuffle", action="store_true")
    parser.add_argument("--overfit-test", action="store_true")
    parser.add_argument("--overfit-n", type=int, default=40)
    parser.add_argument("--fold-diagnostic", action="store_true")
    parser.add_argument("--fold-index", type=int, default=1)
    parser.add_argument("--disable-dropout", action="store_true")
    parser.add_argument("--train-phase-train", action="store_true")
    parser.add_argument("--standardize", action="store_true")
    args = parser.parse_args()

    if args.overfit_test and args.fold_diagnostic:
        raise ValueError("Choose only one of --overfit-test or --fold-diagnostic")
    if args.overfit_test:
        run_overfit(args)
    elif args.fold_diagnostic:
        run_fold_diagnostic(args)
    else:
        raise ValueError("Choose --overfit-test or --fold-diagnostic")


if __name__ == "__main__":
    main()
