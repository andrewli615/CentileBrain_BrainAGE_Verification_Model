#!/usr/bin/env python3
"""
Plot self-dataset adjusted BrainAGE summaries.

Run from this folder:
    python3 plot.py

Save without opening plot windows:
    python3 plot.py --no-show
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT.parent / "ethnoracial_data_outputs"
DEFAULT_PLOT_DIR = ROOT / "outputs"

DATASET_LABELS = {
    "self_black": "Black",
    "self_chilean": "Chilean",
    "self_chinese": "Chinese",
    "self_mexican": "Mexican",
    "self_southasian": "South Asian",
    "self_turkish": "Turkish",
    "self_white": "White",
}

GROUP_ORDER = ["Female 5-40", "Female 40-90", "Male 5-40", "Male 40-90"]
GROUP_COLORS = {
    "Female 5-40": "#2f6fbb",
    "Female 40-90": "#83b8e8",
    "Male 5-40": "#b94f4f",
    "Male 40-90": "#e6a45f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--no-show", action="store_true", help="Save plots without opening windows.")
    return parser.parse_args()


def configure_matplotlib(no_show: bool, plot_dir: Path):
    mpl_config = plot_dir / ".mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    import matplotlib

    if no_show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def latest_adjusted_brainage_file(group_dir: Path) -> Path | None:
    files = sorted(group_dir.glob("*Adjusted_BrainAGE_*.csv"))
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def load_adjusted_brainage(output_root: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    values: list[pd.DataFrame] = []

    for dataset_dir in sorted(output_root.glob("self_*")):
        if not dataset_dir.is_dir():
            continue

        dataset_label = DATASET_LABELS.get(dataset_dir.name, dataset_dir.name.removeprefix("self_"))
        for group_dir in sorted(dataset_dir.iterdir()):
            if not group_dir.is_dir():
                continue

            group_match = re.fullmatch(r"(Female|Male)_(.+)", group_dir.name)
            if not group_match:
                continue

            sex, age_group = group_match.groups()
            group = f"{sex} {age_group}"
            source_file = latest_adjusted_brainage_file(group_dir)
            if source_file is None:
                continue

            df = pd.read_csv(source_file)
            if "Adjusted_BrainAGE" not in df.columns:
                continue

            adjusted = pd.to_numeric(df["Adjusted_BrainAGE"], errors="coerce").dropna()
            if adjusted.empty:
                continue

            records.append(
                {
                    "dataset_key": dataset_dir.name,
                    "dataset": dataset_label,
                    "sex": sex,
                    "age_group": age_group,
                    "group": group,
                    "n": int(adjusted.size),
                    "mean_adjusted_brainage": float(adjusted.mean()),
                    "mae_adjusted_brainage": float(adjusted.abs().mean()),
                    "median_abs_adjusted_brainage": float(adjusted.abs().median()),
                    "source_file": str(source_file),
                }
            )

            values.append(
                pd.DataFrame(
                    {
                        "dataset": dataset_label,
                        "group": group,
                        "Adjusted_BrainAGE": adjusted.to_numpy(),
                    }
                )
            )

    if not records:
        raise FileNotFoundError(f"No self_* Adjusted_BrainAGE CSV files found in {output_root}")

    summary = pd.DataFrame(records).sort_values(["dataset", "sex", "age_group"]).reset_index(drop=True)
    all_values = pd.concat(values, ignore_index=True)
    return summary, all_values


def plot_grouped_bars(summary: pd.DataFrame, plot_dir: Path, plt):
    datasets = summary["dataset"].drop_duplicates().tolist()
    x = np.arange(len(datasets))
    width = 0.18

    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    for i, group in enumerate(GROUP_ORDER):
        y_values: list[float] = []
        n_labels: list[str] = []
        for dataset in datasets:
            row = summary[(summary["dataset"] == dataset) & (summary["group"] == group)]
            if row.empty:
                y_values.append(np.nan)
                n_labels.append("")
            else:
                y_values.append(float(row["mae_adjusted_brainage"].iloc[0]))
                n_labels.append(str(int(row["n"].iloc[0])))

        offset = (i - (len(GROUP_ORDER) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            y_values,
            width=width,
            label=group,
            color=GROUP_COLORS[group],
            edgecolor="#333333",
            linewidth=0.4,
        )
        for bar, label, value in zip(bars, n_labels, y_values):
            if label and not np.isnan(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.08,
                    f"n={label}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    ax.set_title("Mean Absolute Adjusted BrainAGE by Self Dataset and Sex/Age Group", fontsize=15, pad=14)
    ax.set_ylabel("Mean absolute adjusted BrainAGE")
    ax.set_xlabel("Self dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(title="Group", ncols=2, frameon=False)
    ax.margins(x=0.02)
    fig.tight_layout()

    path = plot_dir / "self_dataset_adjusted_brainage_mae.png"
    fig.savefig(path, dpi=200)
    return fig, path


def plot_sorted_bars(summary: pd.DataFrame, plot_dir: Path, plt):
    summary_sorted = summary.sort_values("mae_adjusted_brainage", ascending=True)
    height = max(6, 0.35 * len(summary_sorted) + 1.5)
    labels = summary_sorted.apply(
        lambda row: f"{row['dataset']} - {row['group']} (n={int(row['n'])})",
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(10.5, height))
    ax.barh(
        labels,
        summary_sorted["mae_adjusted_brainage"],
        color="#587c95",
        edgecolor="#333333",
        linewidth=0.35,
    )
    ax.set_title("Mean Absolute Adjusted BrainAGE by Group", fontsize=14, pad=12)
    ax.set_xlabel("Mean absolute adjusted BrainAGE")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()

    path = plot_dir / "self_dataset_adjusted_brainage_mae_sorted.png"
    fig.savefig(path, dpi=200)
    return fig, path


def plot_normal_distribution(all_values: pd.DataFrame, plot_dir: Path, plt):
    values = pd.to_numeric(all_values["Adjusted_BrainAGE"], errors="coerce").dropna().to_numpy()
    mean = values.mean()
    std = values.std(ddof=1)

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.hist(values, bins=35, density=True, color="#88a9bd", edgecolor="white", alpha=0.85, label="Observed")

    x_min = min(values.min(), mean - 4 * std)
    x_max = max(values.max(), mean + 4 * std)
    x = np.linspace(x_min, x_max, 500)
    normal_pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    ax.plot(x, normal_pdf, color="#9f3535", linewidth=2.2, label=f"Normal fit (mean={mean:.2f}, sd={std:.2f})")

    ax.axvline(mean, color="#333333", linestyle="-", linewidth=1, label="Mean")
    ax.axvline(mean - std, color="#333333", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.axvline(mean + std, color="#333333", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_title("Distribution of Adjusted BrainAGE Across Self Dataset Outputs", fontsize=14, pad=12)
    ax.set_xlabel("Adjusted BrainAGE")
    ax.set_ylabel("Density")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()

    path = plot_dir / "self_dataset_adjusted_brainage_distribution_normal_fit.png"
    fig.savefig(path, dpi=200)
    return fig, path


def main() -> None:
    args = parse_args()
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    plt = configure_matplotlib(args.no_show, args.plot_dir)

    summary, all_values = load_adjusted_brainage(args.output_root)
    summary_path = args.plot_dir / "self_dataset_adjusted_brainage_mae_summary.csv"
    summary.to_csv(summary_path, index=False)

    figures_and_paths = [
        plot_grouped_bars(summary, args.plot_dir, plt),
        plot_sorted_bars(summary, args.plot_dir, plt),
        plot_normal_distribution(all_values, args.plot_dir, plt),
    ]

    print(f"Summary: {summary_path}")
    for _, path in figures_and_paths:
        print(f"Plot: {path}")
    print()
    print(summary[["dataset", "group", "n", "mae_adjusted_brainage", "mean_adjusted_brainage"]].to_string(index=False))

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
