"""Plot full strict-parameter ACRNN results for DEAP subjects s01-s32."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("ACRNN_strict_200ep_full_results")
EXP_ROOT = ROOT / "experiments" / "strict_200ep_full"
OUT_DIR = ROOT / "figures"

PAPER = {
    "valence": {"mean": 93.72, "std": 3.21},
    "arousal": {"mean": 93.38, "std": 3.73},
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def macro_f1_from_cm(cm: np.ndarray) -> float:
    tp0, fn0 = cm[0, 0], cm[0, 1]
    fp0 = cm[1, 0]
    tp1, fn1 = cm[1, 1], cm[1, 0]
    fp1 = cm[0, 1]

    def f1(tp: int, fp: int, fn: int) -> float:
        denom = 2 * tp + fp + fn
        return 0.0 if denom == 0 else (2 * tp) / denom

    return (f1(tp0, fp0, fn0) + f1(tp1, fp1, fn1)) / 2


def load_results() -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for dimension in ("valence", "arousal"):
        for path in sorted((EXP_ROOT / dimension).glob("s*_summary.csv")):
            rows = read_rows(path)
            acc = [float(row["test_accuracy"]) * 100 for row in rows]
            best = [float(row["best_test_accuracy"]) * 100 for row in rows]
            cms = [
                np.array(
                    [
                        [int(row["cm_00"]), int(row["cm_01"])],
                        [int(row["cm_10"]), int(row["cm_11"])],
                    ],
                    dtype=int,
                )
                for row in rows
            ]
            best_cms = [
                np.array(
                    [
                        [int(row["best_cm_00"]), int(row["best_cm_01"])],
                        [int(row["best_cm_10"]), int(row["best_cm_11"])],
                    ],
                    dtype=int,
                )
                for row in rows
            ]
            f1 = [macro_f1_from_cm(cm) * 100 for cm in cms]
            best_f1 = [macro_f1_from_cm(cm) * 100 for cm in best_cms]
            results.append(
                {
                    "subject": path.stem.replace("_summary", ""),
                    "dimension": dimension,
                    "mean_accuracy": mean(acc),
                    "std_accuracy": pstdev(acc),
                    "mean_best_accuracy": mean(best),
                    "std_best_accuracy": pstdev(best),
                    "mean_macro_f1": mean(f1),
                    "std_macro_f1": pstdev(f1),
                    "mean_best_macro_f1": mean(best_f1),
                    "std_best_macro_f1": pstdev(best_f1),
                    "cm": sum(cms, np.zeros((2, 2), dtype=int)),
                }
            )
    if len(results) != 64:
        raise RuntimeError(f"Expected 64 subject-dimension result rows, found {len(results)}")
    return results


def write_clean_tables(results: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subject_fields = [
        "dimension",
        "subject",
        "mean_accuracy",
        "std_accuracy",
        "mean_best_accuracy",
        "std_best_accuracy",
        "mean_macro_f1",
        "std_macro_f1",
        "mean_best_macro_f1",
        "std_best_macro_f1",
    ]
    with (OUT_DIR / "strict_200ep_full_per_subject.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=subject_fields)
        writer.writeheader()
        for row in sorted(results, key=lambda r: (r["dimension"], r["subject"])):
            writer.writerow({field: row[field] for field in subject_fields})

    overall_fields = [
        "dimension",
        "final_mean_subject",
        "final_std_subject",
        "best_mean_subject",
        "best_std_subject",
        "paper_mean",
        "paper_std",
        "macro_f1_mean_subject",
        "macro_f1_std_subject",
    ]
    with (OUT_DIR / "strict_200ep_full_overall.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=overall_fields)
        writer.writeheader()
        for dimension in ("valence", "arousal"):
            subset = [r for r in results if r["dimension"] == dimension]
            writer.writerow(
                {
                    "dimension": dimension,
                    "final_mean_subject": mean(r["mean_accuracy"] for r in subset),
                    "final_std_subject": pstdev(r["mean_accuracy"] for r in subset),
                    "best_mean_subject": mean(r["mean_best_accuracy"] for r in subset),
                    "best_std_subject": pstdev(r["mean_best_accuracy"] for r in subset),
                    "paper_mean": PAPER[dimension]["mean"],
                    "paper_std": PAPER[dimension]["std"],
                    "macro_f1_mean_subject": mean(r["mean_macro_f1"] for r in subset),
                    "macro_f1_std_subject": pstdev(r["mean_macro_f1"] for r in subset),
                }
            )


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def style_axes(ax, ylabel: str, ymin: float = 70) -> None:
    ax.set_ylabel(ylabel)
    ax.set_ylim(ymin, 101.5)
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def plot_subject_accuracy(results: list[dict[str, object]], dimension: str) -> None:
    subset = [r for r in results if r["dimension"] == dimension]
    subjects = [r["subject"] for r in subset]
    means = np.array([r["mean_accuracy"] for r in subset])
    stds = np.array([r["std_accuracy"] for r in subset])
    best_means = np.array([r["mean_best_accuracy"] for r in subset])
    x = np.arange(len(subjects))

    fig, ax = plt.subplots(figsize=(13.8, 4.8))
    ax.bar(x, means, yerr=stds, capsize=2.5, width=0.68, color="#4C78A8", label="Final epoch")
    ax.plot(x, best_means, color="#E45756", marker="o", linewidth=1.3, markersize=3.4, label="Best epoch")
    ax.axhline(PAPER[dimension]["mean"], color="#222222", linestyle="--", linewidth=1.2, label="Paper ACRNN mean")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha="right", fontsize=8.5)
    ax.set_xlabel("Subject")
    ax.set_title(f"DEAP {dimension.title()} Classification Accuracy (s01-s32)")
    style_axes(ax, "Accuracy (%)", ymin=70)
    ax.legend(frameon=False, ncols=3, loc="lower center")
    save(fig, f"strict_200ep_full_{dimension}_subject_accuracy")


def plot_subject_fused_accuracy(results: list[dict[str, object]]) -> None:
    subjects = [f"s{i:02d}" for i in range(1, 33)]
    x = np.arange(len(subjects) + 1)
    x_labels = subjects + ["Mean"]
    styles = {
        ("valence", "Final"): {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
        ("valence", "Best"): {"color": "#1f77b4", "linestyle": "--", "marker": "s"},
        ("arousal", "Final"): {"color": "#d95f02", "linestyle": "-", "marker": "o"},
        ("arousal", "Best"): {"color": "#d95f02", "linestyle": "--", "marker": "s"},
    }

    fig, ax = plt.subplots(figsize=(14.8, 4.8))
    for dimension in ("valence", "arousal"):
        subset = {r["subject"]: r for r in results if r["dimension"] == dimension}
        final_values = [subset[subject]["mean_accuracy"] for subject in subjects]
        best_values = [subset[subject]["mean_best_accuracy"] for subject in subjects]
        final_values.append(mean(final_values))
        best_values.append(mean(best_values))

        for label, values in [("Final", final_values), ("Best", best_values)]:
            style = styles[(dimension, label)]
            ax.plot(
                x,
                values,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=1.9,
                markersize=4,
                label=f"{dimension.title()} {label}",
            )

    ax.axvline(len(subjects) - 0.5, color="#888888", linestyle=":", linewidth=1.1)
    ax.text(len(subjects), 100.7, "Overall", ha="center", va="bottom", fontsize=10)
    ax.axhline(PAPER["valence"]["mean"], color="#1f77b4", linestyle=":", linewidth=1.0, alpha=0.55)
    ax.axhline(PAPER["arousal"]["mean"], color="#d95f02", linestyle=":", linewidth=1.0, alpha=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8.5)
    ax.set_title("Strict Full ACRNN Accuracy on DEAP (s01-s32)")
    style_axes(ax, "Accuracy (%)", ymin=70)
    ax.legend(frameon=False, ncols=4, loc="lower left")
    save(fig, "strict_200ep_full_poster_fused_accuracy")


def plot_overall(results: list[dict[str, object]]) -> None:
    dims = ["valence", "arousal"]
    final_mean = [mean(r["mean_accuracy"] for r in results if r["dimension"] == dim) for dim in dims]
    final_std = [pstdev(r["mean_accuracy"] for r in results if r["dimension"] == dim) for dim in dims]
    best_mean = [mean(r["mean_best_accuracy"] for r in results if r["dimension"] == dim) for dim in dims]
    best_std = [pstdev(r["mean_best_accuracy"] for r in results if r["dimension"] == dim) for dim in dims]
    paper_mean = [PAPER[dim]["mean"] for dim in dims]
    paper_std = [PAPER[dim]["std"] for dim in dims]

    x = np.arange(len(dims))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.bar(x - width, final_mean, width, yerr=final_std, capsize=3, color="#4C78A8", label="Final epoch")
    ax.bar(x, best_mean, width, yerr=best_std, capsize=3, color="#E45756", label="Best epoch")
    ax.bar(x + width, paper_mean, width, yerr=paper_std, capsize=3, color="#F2A541", label="Paper ACRNN")
    ax.set_xticks(x)
    ax.set_xticklabels(["Valence", "Arousal"])
    ax.set_title("DEAP Overall Accuracy (s01-s32)")
    style_axes(ax, "Accuracy (%)", ymin=85)
    ax.legend(frameon=False, ncols=3, loc="lower center")
    save(fig, "strict_200ep_full_overall_accuracy")


def plot_paper_table_comparison(results: list[dict[str, object]]) -> None:
    methods = ["DT", "SVM", "Conti-CNN", "CRAM", "GCNN", "CNN-RNN", "A-CNN-RNN", "CNN-RNN-A", "ACRNN", "Ours"]
    valence = [75.95, 89.33, 82.77, 87.09, 88.24, 62.75, 91.48, 89.15, 93.72]
    arousal = [78.18, 89.99, 81.55, 84.46, 87.72, 67.12, 91.59, 89.96, 93.38]
    valence.append(mean(r["mean_accuracy"] for r in results if r["dimension"] == "valence"))
    arousal.append(mean(r["mean_accuracy"] for r in results if r["dimension"] == "arousal"))

    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.plot(x, valence, marker="o", linewidth=2.0, color="#1f77b4", label="Valence")
    ax.plot(x, arousal, marker="s", linewidth=2.0, color="#d95f02", label="Arousal")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha="right")
    ax.set_title("Comparison with Reported DEAP Results")
    style_axes(ax, "Accuracy (%)", ymin=60)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "strict_200ep_full_paper_table_comparison")


def plot_confusion_matrix(results: list[dict[str, object]], dimension: str) -> None:
    cm = sum((r["cm"] for r in results if r["dimension"] == dimension), np.zeros((2, 2), dtype=int))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.25))
    image = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Low", "High"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Low", "High"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{dimension.title()} Confusion Matrix (s01-s32)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{cm_norm[i, j] * 100:.1f}%", ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save(fig, f"strict_200ep_full_{dimension}_confusion_matrix")


def main() -> None:
    results = load_results()
    write_clean_tables(results)
    plot_overall(results)
    plot_subject_fused_accuracy(results)
    plot_paper_table_comparison(results)
    for dimension in ("valence", "arousal"):
        plot_subject_accuracy(results, dimension)
        plot_confusion_matrix(results, dimension)
    print(f"wrote strict full figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
