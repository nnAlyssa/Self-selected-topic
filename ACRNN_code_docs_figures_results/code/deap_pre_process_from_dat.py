"""
Build ACRNN-compatible DEAP pickle files from the official Python .dat files.

The original ACRNN deap_pre_process.py is not present in this project. This
script adapts the reference preprocessing logic from ynulonger/ijcnn, which is
explicitly cited in the ACRNN README, to the official DEAP Python .dat files.

Pipeline per subject:
  data: (40 trials, 40 channels, 8064 samples)
  -> transpose to (40 trials, 8064 samples, 40 channels)
  -> first 384 samples are treated as 3 one-second baseline slices
  -> a one-second baseline template is subtracted from each trial second
  -> normalize each time point across 32 EEG channels
  -> following 7680 samples are segmented using the ijcnn window routine
  -> output dataset shape: (800, 384, 32)
  -> output label shape: (800,)
"""

from argparse import ArgumentParser
from pathlib import Path
import pickle

import numpy as np


np.random.seed(0)
PROJECT_ROOT = Path(__file__).resolve().parent
WINDOW_SIZE = 384
BASELINE_SIZE = 384
TRIAL_SIGNAL_SIZE = 7680
EEG_CHANNELS = 32
SAMPLING_RATE = 128
WINDOWS_PER_TRIAL = TRIAL_SIGNAL_SIZE // WINDOW_SIZE
DIMENSIONS = {
    "valence": 0,
    "arousal": 1,
}


def find_deap_dat_dir(root):
    candidates = [
        root / "data_preprocessed_python",
        root / "data_preprocessed_python" / "data_preprocessed_python",
    ]
    for candidate in candidates:
        if (candidate / "s01.dat").exists():
            return candidate
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find DEAP .dat files. Searched:\n{searched}")


def load_subject(file_path):
    with file_path.open("rb") as f:
        subject = pickle.load(f, encoding="latin1")
    data = np.asarray(subject["data"], dtype=np.float32).transpose(0, 2, 1)
    labels = np.asarray(subject["labels"], dtype=np.float32)
    validate_subject_arrays(data, labels, file_path)
    return data, labels


def validate_subject_arrays(data, labels, file_path):
    if data.shape != (40, 8064, 40):
        raise ValueError(f"{file_path} expected transposed data shape (40, 8064, 40), got {data.shape}")
    if labels.shape != (40, 4):
        raise ValueError(f"{file_path} expected labels shape (40, 4), got {labels.shape}")


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
    norm_dataset_1d = np.zeros((dataset_1d.shape[0], EEG_CHANNELS), dtype=np.float32)
    for i in range(dataset_1d.shape[0]):
        norm_dataset_1d[i] = feature_normalize(dataset_1d[i])
    return norm_dataset_1d


def windows(data, size):
    start = 0
    while (start + size) < data.shape[0]:
        yield int(start), int(start + size)
        start += size


def baseline_correct_trial(trial):
    base_signal = (
        trial[0:128, 0:EEG_CHANNELS]
        + trial[128:256, 0:EEG_CHANNELS]
        + trial[256:384, 0:EEG_CHANNELS]
    ) / 3.0
    signal = trial[384:8064, 0:EEG_CHANNELS].copy()
    for i in range(60):
        signal[i * SAMPLING_RATE:(i + 1) * SAMPLING_RATE, 0:EEG_CHANNELS] -= base_signal
    return norm_dataset(signal)


def split_trial_into_windows(signal):
    segments = []
    for start, end in windows(signal, WINDOW_SIZE):
        if len(signal[start:end]) == WINDOW_SIZE:
            if start == 0:
                segments.append(signal[start:end])
                segments.append(signal[start:end])
            else:
                segments.append(signal[start:end])
    if len(segments) != WINDOWS_PER_TRIAL:
        raise ValueError(f"Expected {WINDOWS_PER_TRIAL} windows, got {len(segments)}")
    return np.ascontiguousarray(np.stack(segments, axis=0), dtype=np.float32)


def preprocess_subject(data, labels, dimension):
    label_index = DIMENSIONS[dimension]
    datasets = []
    binary_labels = []

    for trial_index in range(data.shape[0]):
        corrected_signal = baseline_correct_trial(data[trial_index])
        windows = split_trial_into_windows(corrected_signal)
        label = 1 if labels[trial_index, label_index] > 5.0 else 0
        datasets.append(windows)
        binary_labels.extend([label] * WINDOWS_PER_TRIAL)

    datasets = np.concatenate(datasets, axis=0).astype(np.float32)
    binary_labels = np.asarray(binary_labels, dtype=np.int64)
    indexes = np.arange(binary_labels.shape[0])
    np.random.shuffle(indexes)
    datasets = datasets[indexes]
    binary_labels = binary_labels[indexes]
    return datasets, binary_labels


def output_paths(output_root, dimension, subject_id):
    output_dir = output_root / f"yes_{dimension}"
    data_path = output_dir / f"{subject_id}.mat_win_384_rnn_dataset.pkl"
    label_path = output_dir / f"{subject_id}.mat_win_384_labels.pkl"
    return output_dir, data_path, label_path


def write_pickle(path, value):
    with path.open("wb") as f:
        pickle.dump(value, f, protocol=4)


def process_subject(dat_dir, output_root, subject_id, dimensions, overwrite):
    file_path = dat_dir / f"{subject_id}.dat"
    data, labels = load_subject(file_path)

    results = []
    for dimension in dimensions:
        datasets, binary_labels = preprocess_subject(data, labels, dimension)
        output_dir, data_path, label_path = output_paths(output_root, dimension, subject_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not overwrite and (data_path.exists() or label_path.exists()):
            raise FileExistsError(
                f"Output already exists for {subject_id} {dimension}. "
                "Use --overwrite to replace it."
            )

        write_pickle(data_path, datasets)
        write_pickle(label_path, binary_labels)
        results.append((dimension, datasets.shape, binary_labels.shape, np.bincount(binary_labels, minlength=2)))
    return results


def parse_subjects(subject_arg):
    if subject_arg == "all":
        return [f"s{index:02d}" for index in range(1, 33)]
    return [item.strip() for item in subject_arg.split(",") if item.strip()]


def parse_dimensions(dimension_arg):
    if dimension_arg == "all":
        return ["valence", "arousal"]
    if dimension_arg not in DIMENSIONS:
        raise ValueError(f"Unknown dimension: {dimension_arg}")
    return [dimension_arg]


def main():
    parser = ArgumentParser()
    parser.add_argument("--subjects", default="all", help="all, or comma-separated IDs such as s01,s02")
    parser.add_argument("--dimension", default="all", choices=["all", "valence", "arousal"])
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "deap_shuffled_data_3s"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dat_dir = find_deap_dat_dir(PROJECT_ROOT)
    output_root = Path(args.output_root)
    subjects = parse_subjects(args.subjects)
    dimensions = parse_dimensions(args.dimension)

    print("DEAP dat dir:", dat_dir)
    print("output root:", output_root)
    print("subjects:", ", ".join(subjects))
    print("dimensions:", ", ".join(dimensions))

    for subject_id in subjects:
        results = process_subject(dat_dir, output_root, subject_id, dimensions, args.overwrite)
        for dimension, data_shape, label_shape, counts in results:
            print(
                f"{subject_id} {dimension}: "
                f"data {data_shape}, labels {label_shape}, counts {counts.tolist()}"
            )


if __name__ == "__main__":
    main()
