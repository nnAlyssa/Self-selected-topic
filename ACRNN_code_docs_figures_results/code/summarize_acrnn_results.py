"""Summarize ACRNN diagnostic result CSV files.

This script only reads existing experiment result directories and writes
aggregate CSVs. It does not touch training code, raw data, or source summary
CSVs produced by the training runner.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Project root")
    parser.add_argument("--out-dir", default="result_summaries")
    return parser.parse_args()


def fnum(value: str) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def find_summary_csvs(root: Path) -> list[Path]:
    csv_paths = []
    experiments_dir = root / "experiments"
    if experiments_dir.exists():
        csv_paths.extend(experiments_dir.glob("**/*_summary.csv"))

    # Compatibility for older layouts that still keep result_* directories at
    # the project root.
    csv_paths.extend(root.glob("result_*/*/*_summary.csv"))
    return sorted({path.resolve() for path in csv_paths if path.is_file()})


def describe_source(root: Path, csv_path: Path) -> dict[str, str]:
    rel_parts = csv_path.relative_to(root).parts
    if rel_parts[0] == "experiments":
        experiment_group = rel_parts[1]
        result_root = rel_parts[2]
        dimension = rel_parts[-2]
    else:
        experiment_group = "root_legacy"
        result_root = rel_parts[0]
        dimension = rel_parts[-2]

    return {
        "experiment_group": experiment_group,
        "result_root": result_root,
        "dimension": dimension,
        "subject": csv_path.stem.replace("_summary", ""),
        "source_csv": str(csv_path.relative_to(root)),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_group(
    key: tuple[str, ...],
    group: list[dict[str, str]],
    key_names: list[str],
) -> dict[str, object]:
    group = sorted(group, key=lambda r: (int(r["fold"]), r["source_csv"]))
    acc = [fnum(r["test_accuracy"]) for r in group]
    best_acc = [fnum(r["best_test_accuracy"]) for r in group]
    folds = [int(r["fold"]) for r in group]
    row = dict(zip(key_names, key))
    row.update(
        {
            "folds": " ".join(str(f) for f in folds),
            "n_folds": len(group),
            "mean_accuracy": f"{mean(acc):.6f}",
            "std_accuracy": f"{pstdev(acc):.6f}" if len(acc) > 1 else "0.000000",
            "mean_best_accuracy": f"{mean(best_acc):.6f}",
            "std_best_accuracy": f"{pstdev(best_acc):.6f}" if len(best_acc) > 1 else "0.000000",
            "source_csv_count": len({r["source_csv"] for r in group}),
        }
    )
    return row


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for csv_path in find_summary_csvs(root):
        source = describe_source(root, csv_path)
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row.update(source)
                rows.append(row)

    all_fieldnames = [
        "experiment_group",
        "result_root",
        "subject",
        "dimension",
        "fold",
        "test_accuracy",
        "test_loss",
        "best_test_accuracy",
        "best_test_loss",
        "best_epoch",
        "true_count_0",
        "true_count_1",
        "pred_count_0",
        "pred_count_1",
        "best_pred_count_0",
        "best_pred_count_1",
        "disable_dropout",
        "train_keep_prob",
        "standardize",
        "train_phase_train",
        "use_channel_attention",
        "use_self_attention",
        "split_protocol",
        "train_split_units",
        "test_split_units",
        "source_csv",
    ]
    all_rows_path = out_dir / "all_experiment_folds.csv"
    write_csv(all_rows_path, all_fieldnames, rows)

    result_groups = defaultdict(list)
    subject_groups = defaultdict(list)
    overall_groups = defaultdict(list)
    for row in rows:
        result_key = (
            row["experiment_group"],
            row["result_root"],
            row["subject"],
            row["dimension"],
            row.get("train_keep_prob", ""),
            row.get("disable_dropout", ""),
        )
        subject_key = (
            row["experiment_group"],
            row["subject"],
            row["dimension"],
            row.get("train_keep_prob", ""),
            row.get("disable_dropout", ""),
        )
        overall_key = (
            row["experiment_group"],
            row["dimension"],
            row.get("train_keep_prob", ""),
            row.get("disable_dropout", ""),
        )
        result_groups[result_key].append(row)
        subject_groups[subject_key].append(row)
        overall_groups[overall_key].append(row)

    result_summary_rows = [
        summarize_group(
            key,
            group,
            [
                "experiment_group",
                "result_root",
                "subject",
                "dimension",
                "train_keep_prob",
                "disable_dropout",
            ],
        )
        for key, group in sorted(result_groups.items())
    ]
    subject_summary_rows = [
        summarize_group(
            key,
            group,
            [
                "experiment_group",
                "subject",
                "dimension",
                "train_keep_prob",
                "disable_dropout",
            ],
        )
        for key, group in sorted(subject_groups.items())
    ]
    overall_summary_rows = [
        summarize_group(
            key,
            group,
            ["experiment_group", "dimension", "train_keep_prob", "disable_dropout"],
        )
        for key, group in sorted(overall_groups.items())
    ]

    result_fieldnames = [
        "experiment_group",
        "result_root",
        "subject",
        "dimension",
        "train_keep_prob",
        "disable_dropout",
        "folds",
        "n_folds",
        "mean_accuracy",
        "std_accuracy",
        "mean_best_accuracy",
        "std_best_accuracy",
        "source_csv_count",
    ]
    subject_fieldnames = [
        "experiment_group",
        "subject",
        "dimension",
        "train_keep_prob",
        "disable_dropout",
        "folds",
        "n_folds",
        "mean_accuracy",
        "std_accuracy",
        "mean_best_accuracy",
        "std_best_accuracy",
        "source_csv_count",
    ]
    overall_fieldnames = [
        "experiment_group",
        "dimension",
        "train_keep_prob",
        "disable_dropout",
        "folds",
        "n_folds",
        "mean_accuracy",
        "std_accuracy",
        "mean_best_accuracy",
        "std_best_accuracy",
        "source_csv_count",
    ]

    result_summary_path = out_dir / "result_group_summary.csv"
    per_subject_path = out_dir / "per_subject_summary.csv"
    overall_path = out_dir / "overall_summary.csv"
    write_csv(result_summary_path, result_fieldnames, result_summary_rows)
    write_csv(per_subject_path, subject_fieldnames, subject_summary_rows)
    write_csv(overall_path, overall_fieldnames, overall_summary_rows)

    # Compatibility aliases used by earlier notes.
    write_csv(out_dir / "all_fold_results.csv", all_fieldnames, rows)
    write_csv(out_dir / "group_summary.csv", result_fieldnames, result_summary_rows)

    print(f"read_fold_rows={len(rows)}")
    print(f"wrote={all_rows_path}")
    print(f"wrote={per_subject_path}")
    print(f"wrote={overall_path}")
    print(f"wrote={result_summary_path}")
    for row in subject_summary_rows:
        print(
            "{experiment_group} {subject} {dimension} folds={folds} "
            "mean={mean_accuracy} std={std_accuracy} "
            "best_mean={mean_best_accuracy}".format(**row)
        )


if __name__ == "__main__":
    main()
