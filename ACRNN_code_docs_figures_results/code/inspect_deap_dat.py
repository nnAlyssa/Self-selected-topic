from pathlib import Path
import pickle

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    PROJECT_ROOT / "data_preprocessed_python" / "s01.dat",
    PROJECT_ROOT / "data_preprocessed_python" / "data_preprocessed_python" / "s01.dat",
]


def find_subject_file():
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    searched = "\n".join(str(path) for path in DATA_CANDIDATES)
    raise FileNotFoundError(f"Could not find s01.dat. Searched:\n{searched}")


def binary_counts(values, threshold=5.0):
    values = np.asarray(values)
    labels = (values > threshold).astype(np.int64)
    unique, counts = np.unique(labels, return_counts=True)
    return {int(label): int(count) for label, count in zip(unique, counts)}


def main():
    file_path = find_subject_file()

    with file_path.open("rb") as f:
        subject = pickle.load(f, encoding="latin1")

    data = np.asarray(subject["data"])
    labels = np.asarray(subject["labels"])

    print("file path:", file_path)
    print("keys:", sorted(subject.keys()))
    print("data shape:", data.shape)
    print("labels shape:", labels.shape)
    print("data dtype:", data.dtype)
    print("labels dtype:", labels.dtype)
    print("data min:", np.min(data))
    print("data max:", np.max(data))
    print("data mean:", np.mean(data))
    print("data std:", np.std(data))
    print("labels first 5 rows:")
    print(labels[:5])

    print("label dimension stats:")
    for index in range(labels.shape[1]):
        column = labels[:, index]
        print(
            f"  label[{index}] min/max/mean: "
            f"{np.min(column):.6f}/{np.max(column):.6f}/{np.mean(column):.6f}"
        )

    print("binary label distribution with threshold=5:")
    print("  valence label[0]:", binary_counts(labels[:, 0], threshold=5.0))
    print("  arousal label[1]:", binary_counts(labels[:, 1], threshold=5.0))


if __name__ == "__main__":
    main()
