#!/usr/bin/env python3
"""Create adjusted and unadjusted BrainAGE ridgeline plots for self datasets."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "ethnoracial_data_outputs"
TRAINING_ROOT = ROOT / "training_data"
GRAPH_ROOT = ROOT / "graph_outputs"

DATASET_LABELS = {
    "self_black": "Black",
    "self_chilean": "Chilean",
    "self_chinese": "Chinese",
    "self_mexican": "Mexican",
    "self_southasian": "South Asian",
    "self_turkish": "Turkish",
    "self_white": "White",
}

ROW_ORDER = ["Training", "Black", "Chilean", "Chinese", "Mexican", "South Asian", "Turkish", "White"]
PANEL_ORDER = [
    ("Female", "5-40"),
    ("Female", "40-90"),
    ("Male", "5-40"),
    ("Male", "40-90"),
]

COLORS = {
    "Training": "#585858",
    "Black": "#4C78A8",
    "Chilean": "#F58518",
    "Chinese": "#54A24B",
    "Mexican": "#E45756",
    "South Asian": "#72B7B2",
    "Turkish": "#B279A2",
    "White": "#9D755D",
}

# --------------------------- adjustable plot settings ---------------------------
# Change these values, then rerun:
#     python3 plot_brainage_ridges.py
#
# TRAINING_DENSITY_STYLE:
#   "normal" = draw training rows as a smooth fitted normal curve.
#   "kde"    = draw training rows with KDE, using TRAINING_KDE_BANDWIDTH.
TRAINING_DENSITY_STYLE = "normal"

# KDE bandwidth multipliers. Larger values make smoother, more normal-like curves.
# Use values around 1.0-2.5 for smooth plots; lower values show more bumps.
TRAINING_KDE_BANDWIDTH = 2.5
SELF_KDE_BANDWIDTH = 1.15

# Ridge height and spacing. Lower RIDGE_HEIGHT creates more whitespace.
RIDGE_HEIGHT = 0.68

# X-axis padding around the observed min/max BrainAGE values.
X_AXIS_MARGIN_FRACTION = 0.10
X_AXIS_MIN_MARGIN = 1.0
X_AXIS_LABEL = "BrainAGE (years)"
X_AXIS_TICK_STEP = 10
X_AXIS_LIMITS = (-20, 20)

# The ridgeline input table is large and can be regenerated from the output CSVs.
# Turn this on only when you want the underlying plotted values saved for checking.
SAVE_PLOT_DATA = False
# -------------------------------------------------------------------------------


def configure_matplotlib():
    mpl_config = GRAPH_ROOT / ".mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    os.environ.setdefault("XDG_CACHE_HOME", str(mpl_config))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    return plt, MultipleLocator


def latest(paths):
    paths = list(paths)
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None


def training_file(sex: str, age_group: str) -> Path:
    sex_key = sex.lower()
    age_key = "age1234" if age_group == "5-40" else "age56789"
    return TRAINING_ROOT / f"training_{age_key}_{sex_key}.csv"


def load_training_records() -> list[pd.DataFrame]:
    records = []
    for sex, age_group in PANEL_ORDER:
        path = training_file(sex, age_group)
        df = pd.read_csv(path)
        age = pd.to_numeric(df["age"], errors="coerce")
        unadjusted = pd.to_numeric(df["predicted brain age"], errors="coerce") - age
        adjusted = pd.to_numeric(df["predicted brain age_adjusted"], errors="coerce") - age

        records.append(
            pd.DataFrame(
                {
                    "dataset": "Training",
                    "sex": sex,
                    "age_group": age_group,
                    "BrainAGE": unadjusted,
                    "kind": "Unadjusted",
                    "source_file": str(path),
                }
            )
        )
        records.append(
            pd.DataFrame(
                {
                    "dataset": "Training",
                    "sex": sex,
                    "age_group": age_group,
                    "BrainAGE": adjusted,
                    "kind": "Adjusted",
                    "source_file": str(path),
                }
            )
        )
    return records


def load_self_records() -> list[pd.DataFrame]:
    records = []
    for dataset_dir in sorted(OUTPUT_ROOT.glob("self_*")):
        if not dataset_dir.is_dir():
            continue
        dataset = DATASET_LABELS.get(dataset_dir.name, dataset_dir.name.removeprefix("self_"))
        for group_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            match = re.fullmatch(r"(Female|Male)_(.+)", group_dir.name)
            if not match:
                continue
            sex, age_group = match.groups()

            adjusted_file = latest(group_dir.glob("*_Adjusted_BrainAGE_*.csv"))
            unadjusted_file = latest(
                path
                for path in group_dir.glob("*_MR_predicted_age_*.csv")
                if "_Adjusted_MR_predicted_age_" not in path.name
            )

            if adjusted_file is not None:
                adjusted = pd.read_csv(adjusted_file)
                records.append(
                    pd.DataFrame(
                        {
                            "dataset": dataset,
                            "sex": sex,
                            "age_group": age_group,
                            "BrainAGE": pd.to_numeric(adjusted["Adjusted_BrainAGE"], errors="coerce"),
                            "kind": "Adjusted",
                            "source_file": str(adjusted_file),
                        }
                    )
                )

            if unadjusted_file is not None:
                unadjusted = pd.read_csv(unadjusted_file)
                records.append(
                    pd.DataFrame(
                        {
                            "dataset": dataset,
                            "sex": sex,
                            "age_group": age_group,
                            "BrainAGE": pd.to_numeric(unadjusted["Unadjusted_BrainAGE"], errors="coerce"),
                            "kind": "Unadjusted",
                            "source_file": str(unadjusted_file),
                        }
                    )
                )
    return records


def build_plot_data() -> pd.DataFrame:
    data = pd.concat(load_training_records() + load_self_records(), ignore_index=True)
    data = data.dropna(subset=["BrainAGE"])
    data["dataset"] = pd.Categorical(data["dataset"], categories=ROW_ORDER, ordered=True)
    data["panel"] = data["sex"] + " " + data["age_group"]
    return data.sort_values(["kind", "sex", "age_group", "dataset"]).reset_index(drop=True)


def normal_density(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values, ddof=1))
    if not np.isfinite(std) or std <= 0:
        std = max((x_grid.max() - x_grid.min()) / 120, 0.2)
    return np.exp(-0.5 * ((x_grid - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def kde_density(values: np.ndarray, x_grid: np.ndarray, bandwidth: float) -> np.ndarray:
    kde = gaussian_kde(values)
    kde.set_bandwidth(kde.factor * bandwidth)
    return kde(x_grid)


def density_values(values: np.ndarray, x_grid: np.ndarray, label: str) -> np.ndarray:
    if len(values) < 2 or np.nanstd(values) == 0:
        center = values[0] if len(values) else 0.0
        width = max((x_grid.max() - x_grid.min()) / 120, 0.2)
        density = np.exp(-0.5 * ((x_grid - center) / width) ** 2)
    elif label == "Training" and TRAINING_DENSITY_STYLE == "normal":
        density = normal_density(values, x_grid)
    else:
        bandwidth = TRAINING_KDE_BANDWIDTH if label == "Training" else SELF_KDE_BANDWIDTH
        density = kde_density(values, x_grid, bandwidth)
    max_density = np.nanmax(density)
    if max_density <= 0 or not np.isfinite(max_density):
        return np.zeros_like(x_grid)
    return density / max_density


def plot_kind(data: pd.DataFrame, kind: str, plt, multiple_locator):
    subset = data[data["kind"] == kind]
    values = subset["BrainAGE"].to_numpy()
    if X_AXIS_LIMITS is None:
        x_min = float(np.nanmin(values))
        x_max = float(np.nanmax(values))
        span = x_max - x_min
        margin = max(span * X_AXIS_MARGIN_FRACTION, X_AXIS_MIN_MARGIN)
        x_min, x_max = x_min - margin, x_max + margin
    else:
        x_min, x_max = X_AXIS_LIMITS
    x_grid = np.linspace(x_min, x_max, 500)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.ravel()
    row_positions = {label: len(ROW_ORDER) - 1 - idx for idx, label in enumerate(ROW_ORDER)}

    for ax, (sex, age_group) in zip(axes, PANEL_ORDER):
        panel = subset[(subset["sex"] == sex) & (subset["age_group"] == age_group)]
        for label in ROW_ORDER:
            baseline = row_positions[label]
            row = panel[panel["dataset"] == label]
            if row.empty:
                ax.text(
                    x_grid.min(),
                    baseline + 0.08,
                    "no data",
                    color="#9a9a9a",
                    fontsize=8,
                    va="center",
                )
                continue

            row_values = row["BrainAGE"].to_numpy(dtype=float)
            scaled_density = density_values(row_values, x_grid, label) * RIDGE_HEIGHT
            color = COLORS[label]
            ax.fill_between(
                x_grid,
                baseline,
                baseline + scaled_density,
                color=color,
                alpha=0.68 if label != "Training" else 0.52,
                linewidth=0,
            )
            ax.plot(x_grid, baseline + scaled_density, color=color, linewidth=1.1)
            mean = float(np.nanmean(row_values))
            mean_height = np.interp(mean, x_grid, scaled_density)
            ax.vlines(
                mean,
                baseline,
                baseline + max(mean_height, 0.22),
                color="#2f2f2f",
                linestyle=(0, (4, 3)),
                linewidth=1.0,
            )
            ax.text(
                0.985,
                baseline + 0.08,
                f"n={len(row_values)}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=8,
                color="#333333",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                },
            )

        ax.set_title(f"{sex} {age_group}", fontsize=13, pad=10)
        ax.set_yticks([row_positions[label] for label in ROW_ORDER])
        ax.set_yticklabels(ROW_ORDER)
        ax.set_ylim(-0.55, len(ROW_ORDER) - 0.1)
        ax.set_xlim(x_min, x_max)
        ax.grid(axis="x", linestyle="--", alpha=0.22)
        ax.xaxis.set_major_locator(multiple_locator(X_AXIS_TICK_STEP))
        ax.tick_params(axis="x", labelbottom=True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    fig.suptitle(f"{kind} BrainAGE Distributions by Sex/Age Group", fontsize=18, y=0.98)
    fig.supxlabel(X_AXIS_LABEL, fontsize=13)
    fig.tight_layout(rect=(0.04, 0.04, 1, 0.95), w_pad=3.2, h_pad=2.4)

    output = GRAPH_ROOT / f"self_{kind.lower()}_brainage_ridgelines.png"
    fig.savefig(output, dpi=220)
    return output


def main() -> None:
    GRAPH_ROOT.mkdir(parents=True, exist_ok=True)
    plt, multiple_locator = configure_matplotlib()
    data = build_plot_data()
    if SAVE_PLOT_DATA:
        data.to_csv(GRAPH_ROOT / "self_brainage_ridgeline_plot_data.csv", index=False)

    outputs = [
        plot_kind(data, "Adjusted", plt, multiple_locator),
        plot_kind(data, "Unadjusted", plt, multiple_locator),
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
