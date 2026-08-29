#!/usr/bin/env python3
"""Adjusted and unadjusted BrainAGE ridgeline plots for active datasets."""

from __future__ import annotations

import os
import re
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from brainage_dataset_config import BRAINAGE_DATASETS, BRAINAGE_ROW_ORDER, COLORS, age_panel_rows


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "ethnoracial_data_outputs"
TRAINING_ROOT = ROOT / "training_data"
GRAPH_ROOT = ROOT / "graph_outputs"

PANELS = [("Female", "5-40"), ("Female", "40-90"), ("Male", "5-40"), ("Male", "40-90")]
AGE_PANELS = ["5-40", "40-90"]
SEXES = ["Female", "Male"]

X_LIMITS = (-10, 10)
RIDGE_HEIGHT = 0.75
TRAINING_BANDWIDTH = 2.5
DATASET_BANDWIDTH = 1.15


def setup_matplotlib():
    cache = GRAPH_ROOT / ".mplconfig"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    return plt, MultipleLocator


def newest_csv(folder: Path, pattern: str) -> Path | None:
    files = list(folder.glob(pattern))
    return max(files, key=lambda path: path.stat().st_mtime) if files else None


def training_file(sex: str, age_group: str) -> Path:
    age_key = "age1234" if age_group == "5-40" else "age56789"
    return TRAINING_ROOT / f"training_{age_key}_{sex.lower()}.csv"


def load_training_rows(measure: str) -> list[pd.DataFrame]:
    rows = []
    for sex, age_group in PANELS:
        df = pd.read_csv(training_file(sex, age_group))
        predicted = "predicted brain age_adjusted" if measure == "Adjusted" else "predicted brain age"
        brainage = df[predicted] - df["age"]
        rows.append(
            pd.DataFrame(
                {
                    "dataset": "Training",
                    "sex": sex,
                    "age_group": age_group,
                    "BrainAGE": brainage,
                }
            )
        )
    return rows


def load_dataset_rows(measure: str) -> list[pd.DataFrame]:
    rows = []
    for dataset_key, label, _ in BRAINAGE_DATASETS:
        for group_dir in sorted(path for path in (OUTPUT_ROOT / dataset_key).iterdir() if path.is_dir()):
            match = re.fullmatch(r"(Female|Male)_(5-40|40-90)", group_dir.name)
            if not match:
                continue
            pattern = "*_Adjusted_BrainAGE_*.csv" if measure == "Adjusted" else "*_MR_predicted_age_*.csv"
            bag_file = newest_csv(group_dir, pattern)
            if bag_file is None:
                continue
            if measure == "Unadjusted" and "_Adjusted_" in bag_file.name:
                files = [path for path in group_dir.glob(pattern) if "_Adjusted_" not in path.name]
                bag_file = max(files, key=lambda path: path.stat().st_mtime) if files else None
            if bag_file is None:
                continue
            sex, age_group = match.groups()
            column = "Adjusted_BrainAGE" if measure == "Adjusted" else "Unadjusted_BrainAGE"
            values = age_panel_rows(pd.read_csv(bag_file), age_group)[column]
            rows.append(
                pd.DataFrame(
                    {
                        "dataset_key": dataset_key,
                        "dataset": label,
                        "sex": sex,
                        "age_group": age_group,
                        "BrainAGE": values,
                    }
                )
            )
    return rows


def build_plot_data(measure: str = "Adjusted") -> pd.DataFrame:
    data = pd.concat(load_training_rows(measure) + load_dataset_rows(measure), ignore_index=True)
    data = data.dropna(subset=["BrainAGE"])
    data["dataset"] = pd.Categorical(data["dataset"], categories=BRAINAGE_ROW_ORDER, ordered=True)
    return data.sort_values(["sex", "age_group", "dataset"]).reset_index(drop=True)


def density(values: np.ndarray, x_grid: np.ndarray, dataset: str) -> np.ndarray:
    if len(values) < 2 or np.std(values) == 0:
        center = values[0] if len(values) else 0
        y = np.exp(-0.5 * ((x_grid - center) / 0.2) ** 2)
    elif dataset == "Training":
        mean = values.mean()
        sd = values.std(ddof=1)
        y = np.exp(-0.5 * ((x_grid - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    else:
        kde = gaussian_kde(values)
        kde.set_bandwidth(kde.factor * DATASET_BANDWIDTH)
        y = kde(x_grid)

    return y / y.max() * RIDGE_HEIGHT


def draw_panel(ax, data: pd.DataFrame, sex: str, age_group: str, x_grid: np.ndarray, locator) -> None:
    row_positions = {label: len(BRAINAGE_ROW_ORDER) - 1 - i for i, label in enumerate(BRAINAGE_ROW_ORDER)}
    panel = data[(data["sex"] == sex) & (data["age_group"] == age_group)]

    for dataset in BRAINAGE_ROW_ORDER:
        baseline = row_positions[dataset]
        values = panel.loc[panel["dataset"].eq(dataset), "BrainAGE"].to_numpy(float)
        if len(values) == 0:
            ax.text(x_grid.min(), baseline + 0.08, "no data", color="#9a9a9a", fontsize=8)
            continue

        y = density(values, x_grid, dataset)
        color = COLORS[dataset]
        ax.fill_between(x_grid, baseline, baseline + y, color=color, alpha=0.62, linewidth=0)
        ax.plot(x_grid, baseline + y, color=color, linewidth=1.1)
        mean = values.mean()
        ax.vlines(mean, baseline, baseline + max(np.interp(mean, x_grid, y), 0.22), color="#2f2f2f", linestyle=(0, (4, 3)), linewidth=1)
        ax.text(
            0.985,
            baseline + 0.08,
            f"n={len(values)}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8,
            color="#333333",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )

    ax.set_title(f"{sex} {age_group}", fontsize=13, pad=10)
    ax.set_yticks([row_positions[label] for label in BRAINAGE_ROW_ORDER])
    ax.set_yticklabels(BRAINAGE_ROW_ORDER)
    ax.set_ylim(-0.55, len(BRAINAGE_ROW_ORDER) - 0.1)
    ax.set_xlim(*X_LIMITS)
    ax.grid(axis="x", linestyle="--", alpha=0.22)
    ax.xaxis.set_major_locator(locator(10))
    ax.tick_params(axis="x", labelbottom=True)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)


def make_plot(data: pd.DataFrame, panels: list[tuple[str, str]], measure: str, output_name: str, plt, locator) -> Path:
    x_grid = np.linspace(*X_LIMITS, 500)
    n_cols = 2
    n_rows = math.ceil(len(panels) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (sex, age_group) in zip(axes, panels):
        draw_panel(ax, data, sex, age_group, x_grid, locator)
    for ax in axes[len(panels) :]:
        ax.axis("off")

    fig.suptitle(f"{measure} BrainAGE Distributions", fontsize=18, y=0.98)
    fig.supxlabel("BrainAGE (years)", fontsize=13)
    fig.tight_layout(rect=(0.04, 0.06, 1, 0.94), w_pad=3.2)

    output = GRAPH_ROOT / output_name
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def main() -> None:
    GRAPH_ROOT.mkdir(exist_ok=True)
    plt, locator = setup_matplotlib()
    outputs = []
    for measure in ("Adjusted", "Unadjusted"):
        data = build_plot_data(measure)
        key = measure.lower()
        outputs.extend([
            make_plot(data, PANELS, measure, f"self_{key}_brainage_ridgelines.png", plt, locator),
            make_plot(data, [("Female", age) for age in AGE_PANELS], measure, f"self_female_{key}_brainage_ridgelines.png", plt, locator),
            make_plot(data, [("Male", age) for age in AGE_PANELS], measure, f"self_male_{key}_brainage_ridgelines.png", plt, locator),
        ])
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
