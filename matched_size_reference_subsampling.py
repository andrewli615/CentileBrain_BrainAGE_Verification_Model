#!/usr/bin/env python3
"""Adjusted BAG matched-size reference subsampling.

For each active dataset/sex/age group:
1. Read observed adjusted BrainAGE values.
2. Draw 10,000 same-size samples from the matching training reference data.
3. Compare the observed mean BAG and SD BAG to those reference distributions.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from brainage_dataset_config import COLORS, STATS_DATASETS


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "ethnoracial_data_outputs"
TRAINING_ROOT = ROOT / "training_data"
GRAPH_ROOT = ROOT / "graph_outputs"

N_ITER = 10_000
SEED = 2005
PANELS = [("Female", "5-40"), ("Female", "40-90"), ("Male", "5-40"), ("Male", "40-90")]


def setup_matplotlib():
    cache = GRAPH_ROOT / ".mplconfig"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def newest_csv(folder: Path, pattern: str) -> Path | None:
    files = list(folder.glob(pattern))
    return max(files, key=lambda path: path.stat().st_mtime) if files else None


def training_bag(sex: str, age_group: str) -> np.ndarray:
    age_key = "age1234" if age_group == "5-40" else "age56789"
    path = TRAINING_ROOT / f"training_{age_key}_{sex.lower()}.csv"
    df = pd.read_csv(path)
    return (df["predicted brain age_adjusted"] - df["age"]).dropna().to_numpy(float)


def observed_groups():
    for dataset_key, label, _ in STATS_DATASETS:
        dataset_dir = OUTPUT_ROOT / dataset_key
        for group_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            match = re.fullmatch(r"(Female|Male)_(5-40|40-90)", group_dir.name)
            if not match:
                continue
            bag_file = newest_csv(group_dir, "*_Adjusted_BrainAGE_*.csv")
            if bag_file is None:
                continue
            sex, age_group = match.groups()
            values = pd.read_csv(bag_file)["Adjusted_BrainAGE"].dropna().to_numpy(float)
            yield dataset_key, label, sex, age_group, values, bag_file


def reference_distributions(reference: np.ndarray, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    samples = np.array([rng.choice(reference, size=n, replace=False) for _ in range(N_ITER)])
    return {
        "mean_BAG": samples.mean(axis=1),
        "sd_BAG": samples.std(axis=1, ddof=1),
    }


def empirical_p(reference_stats: np.ndarray, observed: float) -> float:
    lower = (np.sum(reference_stats <= observed) + 1) / (len(reference_stats) + 1)
    upper = (np.sum(reference_stats >= observed) + 1) / (len(reference_stats) + 1)
    return min(1.0, 2 * min(lower, upper))


def p_label(p_value: float) -> str:
    if p_value < 0.001:
        return "p<0.001"
    if p_value < 0.01:
        return f"p={p_value:.3f}"
    return f"p={p_value:.2f}"


def summarize(reference_stats: np.ndarray, observed: float) -> dict:
    low, high = np.percentile(reference_stats, [2.5, 97.5])
    return {
        "observed": observed,
        "reference_mean": reference_stats.mean(),
        "reference_p2_5": low,
        "reference_p97_5": high,
        "percentile": 100 * np.mean(reference_stats <= observed),
        "two_sided_empirical_p": empirical_p(reference_stats, observed),
        "outside_95_reference_interval": observed < low or observed > high,
    }


def run_analysis():
    rng = np.random.default_rng(SEED)
    rows = []
    sample_rows = []
    plot_panels = []
    reference_cache = {}

    for dataset_key, dataset, sex, age_group, values, source_file in observed_groups():
        reference_key = (sex, age_group, len(values))
        if reference_key not in reference_cache:
            reference_cache[reference_key] = reference_distributions(
                training_bag(sex, age_group), len(values), rng
            )

        observed_stats = {
            "mean_BAG": values.mean(),
            "sd_BAG": values.std(ddof=1),
        }

        sample_table = pd.DataFrame(
            {
                "iteration": np.arange(1, N_ITER + 1),
                "dataset_key": dataset_key,
                "dataset": dataset,
                "sex": sex,
                "age_group": age_group,
                "n": len(values),
                "random_seed": SEED,
                "reference_mean_BAG": reference_cache[reference_key]["mean_BAG"],
                "reference_sd_BAG": reference_cache[reference_key]["sd_BAG"],
                "observed_mean_BAG": observed_stats["mean_BAG"],
                "observed_sd_BAG": observed_stats["sd_BAG"],
            }
        )
        sample_rows.append(sample_table)

        for statistic, observed in observed_stats.items():
            reference_stats = reference_cache[reference_key][statistic]
            summary = summarize(reference_stats, observed)
            row = {
                "dataset_key": dataset_key,
                "dataset": dataset,
                "sex": sex,
                "age_group": age_group,
                "kind": "Adjusted",
                "statistic": statistic,
                "n": len(values),
                "reference_n": len(training_bag(sex, age_group)),
                "iterations": N_ITER,
                "random_seed": SEED,
                "source_file": str(source_file.relative_to(ROOT)),
                **summary,
            }
            rows.append(row)
            plot_panels.append({**row, "reference_values": reference_stats})

    order = {panel: i for i, panel in enumerate(PANELS)}
    rows.sort(key=lambda row: (row["statistic"], order[(row["sex"], row["age_group"])], row["dataset"]))
    plot_panels.sort(key=lambda row: (row["statistic"], order[(row["sex"], row["age_group"])], row["dataset"]))
    samples = pd.concat(sample_rows, ignore_index=True)
    return pd.DataFrame(rows), samples, plot_panels


def plot_distributions(plot_panels: list[dict], statistic: str, plt) -> Path:
    panels = [panel for panel in plot_panels if panel["statistic"] == statistic]
    x_values = np.concatenate([panel["reference_values"] for panel in panels] + [[p["observed"] for p in panels]])
    x_min, x_max = np.percentile(x_values, [0.5, 99.5])
    x_min = min(x_min, *(panel["observed"] for panel in panels))
    x_max = max(x_max, *(panel["observed"] for panel in panels))
    margin = max((x_max - x_min) * 0.08, 0.05)
    x_min, x_max = x_min - margin, x_max + margin

    n_cols = 3
    n_rows = math.ceil(len(panels) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15.5, 3 * n_rows + 0.4), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for index, (ax, panel) in enumerate(zip(axes, panels)):
        color = COLORS[panel["dataset"]]
        reference = panel["reference_values"]
        ax.hist(reference, bins=35, density=True, color=color, alpha=0.32, edgecolor="white")
        ax.axvspan(panel["reference_p2_5"], panel["reference_p97_5"], color=color, alpha=0.10)
        ax.axvline(panel["reference_mean"], color="#2f2f2f", linestyle=(0, (4, 3)), linewidth=1.2)
        ax.axvline(panel["observed"], color=color, linewidth=2.2)
        ax.set_xlim(x_min, x_max)
        ax.set_title(f"{panel['dataset']} | {panel['sex']} {panel['age_group']} | n={panel['n']}", fontsize=10.5)
        if index % n_cols == 0:
            ax.set_ylabel("Density")
        ax.text(
            0.98,
            0.94,
            f"observed = {panel['observed']:.2f}\n{p_label(panel['two_sided_empirical_p'])}",
            transform=ax.transAxes,
            color=color,
            fontsize=8.5,
            ha="right",
            va="top",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
        ax.grid(axis="x", linestyle="--", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[len(panels) :]:
        ax.axis("off")

    x_label = "Mean BrainAGE (years)" if statistic == "mean_BAG" else "BrainAGE SD (years)"
    for ax in axes[-n_cols:]:
        if ax.has_data():
            ax.set_xlabel(x_label)

    stat_label = "mean BAG" if statistic == "mean_BAG" else "SD BAG"
    fig.suptitle(f"Adjusted {stat_label}: matched-size reference distributions", fontsize=16, y=0.995)
    fig.text(
        0.5,
        0.968,
        "Histogram = 10,000 matched-size reference samples; shaded band = 95% reference interval",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.0, w_pad=1.8)

    output = GRAPH_ROOT / f"matched_size_reference_adjusted_{statistic.lower()}.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def main() -> None:
    GRAPH_ROOT.mkdir(exist_ok=True)
    summary, samples, plot_panels = run_analysis()
    summary.to_csv(GRAPH_ROOT / "matched_size_reference_subsampling_summary.csv", index=False)
    samples.to_csv(GRAPH_ROOT / "matched_size_reference_subsampling_samples.csv", index=False)

    plt = setup_matplotlib()
    print(GRAPH_ROOT / "matched_size_reference_subsampling_summary.csv")
    print(GRAPH_ROOT / "matched_size_reference_subsampling_samples.csv")
    print(plot_distributions(plot_panels, "mean_BAG", plt))
    print(plot_distributions(plot_panels, "sd_BAG", plt))


if __name__ == "__main__":
    main()
