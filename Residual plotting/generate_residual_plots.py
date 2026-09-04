#!/usr/bin/env python3
"""Residual-vs-age plots with segmented observed regressions."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brainage_dataset_config import BRAINAGE_DATASETS, age_panel_rows  # noqa: E402


PLOT_ROOT = ROOT / "Residual plotting"
OUTPUT_ROOT = PLOT_ROOT / "Chinese"
TRAINING_ROOT = ROOT / "training_data"
COHORT_ROOT = ROOT / "ethnoracial_data_outputs"

PANELS = [("Female", "5-40"), ("Female", "40-90"), ("Male", "5-40"), ("Male", "40-90")]
AGE_LIMITS = {"5-40": (5, 40), "40-90": (40, 90)}
MEASURES = ("unadjusted", "adjusted")

TRAINING_COLOR = "#0072B2"
OBSERVED_COLOR = "#D62728"
OBSERVED_FIT_COLOR = "#8B1A1A"
ZERO_COLOR = "#6E6E6E"


@dataclass(frozen=True)
class PlotData:
    group_key: str
    group_label: str
    sex: str
    age_group: str
    training: pd.DataFrame
    observed: pd.DataFrame


def setup_matplotlib() -> None:
    cache = PLOT_ROOT / ".mplconfig"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib

    matplotlib.use("Agg")


def newest_csv(folder: Path, pattern: str, *, exclude_adjusted: bool = False) -> Path | None:
    files = list(folder.glob(pattern))
    if exclude_adjusted:
        files = [path for path in files if "_Adjusted_" not in path.name]
    return max(files, key=lambda path: path.stat().st_mtime) if files else None


def training_file(sex: str, age_group: str) -> Path:
    age_key = "age1234" if age_group == "5-40" else "age56789"
    return TRAINING_ROOT / f"training_{age_key}_{sex.lower()}.csv"


def load_training(sex: str, age_group: str, measure: str) -> pd.DataFrame:
    df = pd.read_csv(training_file(sex, age_group))
    predicted_col = "predicted brain age_adjusted" if measure == "adjusted" else "predicted brain age"
    plot_df = pd.DataFrame(
        {
            "age": pd.to_numeric(df["age"], errors="coerce"),
            "residual": pd.to_numeric(df[predicted_col], errors="coerce")
            - pd.to_numeric(df["age"], errors="coerce"),
        }
    )
    return age_panel_rows(plot_df.rename(columns={"age": "chronological_age"}), age_group).rename(
        columns={"chronological_age": "age"}
    )


def load_observed(group_key: str, sex: str, age_group: str, measure: str) -> pd.DataFrame:
    folder = COHORT_ROOT / group_key / f"{sex}_{age_group}"
    if measure == "adjusted":
        path = newest_csv(folder, "*_Adjusted_BrainAGE_*.csv")
        if path is None:
            return pd.DataFrame(columns=["age", "residual"])
        df = pd.read_csv(path)
        residual = pd.to_numeric(df["Adjusted_BrainAGE"], errors="coerce")
    else:
        path = newest_csv(folder, "*_MR_predicted_age_*.csv", exclude_adjusted=True)
        if path is None:
            return pd.DataFrame(columns=["age", "residual"])
        df = pd.read_csv(path)
        if "Unadjusted_BrainAGE" in df:
            residual = pd.to_numeric(df["Unadjusted_BrainAGE"], errors="coerce")
        else:
            residual = pd.to_numeric(df["MR_predicted_age"], errors="coerce") - pd.to_numeric(
                df["chronological_age"], errors="coerce"
            )

    plot_df = pd.DataFrame(
        {
            "chronological_age": pd.to_numeric(df["chronological_age"], errors="coerce"),
            "residual": residual,
        }
    )
    plot_df = age_panel_rows(plot_df, age_group).rename(columns={"chronological_age": "age"})
    return plot_df.dropna(subset=["age", "residual"]).reset_index(drop=True)


def load_plot_data(group_key: str, group_label: str, measure: str) -> list[PlotData]:
    panels = []
    for sex, age_group in PANELS:
        panels.append(
            PlotData(
                group_key=group_key,
                group_label=group_label,
                sex=sex,
                age_group=age_group,
                training=load_training(sex, age_group, measure).dropna(subset=["age", "residual"]),
                observed=load_observed(group_key, sex, age_group, measure),
            )
        )
    return panels


def linear_fit(df: pd.DataFrame) -> tuple[float, float, float] | None:
    clean = df.dropna(subset=["age", "residual"])
    if len(clean) < 2 or clean["age"].nunique() < 2:
        return None
    result = stats.linregress(clean["age"], clean["residual"])
    return result.slope, result.intercept, result.rvalue


def fixed_segments(age_group: str, segment_years: int) -> list[tuple[float, float]]:
    lo, hi = AGE_LIMITS[age_group]
    starts = np.arange(lo, hi, segment_years)
    return [(float(start), float(min(start + segment_years, hi))) for start in starts]


def rows_in_segment(df: pd.DataFrame, start: float, end: float, is_last: bool) -> pd.DataFrame:
    if is_last:
        return df[df["age"].between(start, end, inclusive="both")]
    return df[(df["age"] >= start) & (df["age"] < end)]


def supported_segments(df: pd.DataFrame, age_group: str, segment_years: int) -> list[dict[str, object]]:
    clean = df.dropna(subset=["age", "residual"])
    segments = fixed_segments(age_group, segment_years)
    rows_by_segment = []
    nonempty_indices = []
    for idx, (start, end) in enumerate(segments):
        rows = rows_in_segment(df, start, end, idx == len(segments) - 1)
        rows_by_segment.append(
            {
                "segment_start": start,
                "segment_end": end,
                "n": len(rows),
            }
        )
        if len(rows) > 0:
            nonempty_indices.append(idx)

    if clean.empty or not nonempty_indices:
        return []
    first, last = nonempty_indices[0], nonempty_indices[-1]
    return rows_by_segment[first : last + 1]


def piecewise_design(x: np.ndarray, knots: list[float]) -> np.ndarray:
    columns = [np.ones_like(x), x]
    columns.extend(np.maximum(0.0, x - knot) for knot in knots)
    return np.column_stack(columns)


def piecewise_predict(x: np.ndarray, coefficients: np.ndarray, knots: list[float]) -> np.ndarray:
    return piecewise_design(x, knots) @ coefficients


def continuous_segmented_fit(df: pd.DataFrame, age_group: str, segment_years: int) -> dict[str, object] | None:
    segments = supported_segments(df, age_group, segment_years)
    if not segments:
        return None

    start = float(segments[0]["segment_start"])
    end = float(segments[-1]["segment_end"])
    clean = df.dropna(subset=["age", "residual"])
    clean = clean[clean["age"].between(start, end, inclusive="both")]
    if len(clean) < 2 or clean["age"].nunique() < 2:
        return None

    knots = [float(segment["segment_end"]) for segment in segments[:-1]]
    x = clean["age"].to_numpy(float)
    y = clean["residual"].to_numpy(float)
    coefficients, *_ = np.linalg.lstsq(piecewise_design(x, knots), y, rcond=None)
    predicted = piecewise_predict(x, coefficients, knots)
    model_r = np.corrcoef(y, predicted)[0, 1] if np.std(predicted) > 0 and np.std(y) > 0 else np.nan

    segment_rows = []
    for segment in segments:
        segment_start = float(segment["segment_start"])
        active_knots = [knot for knot in knots if knot <= segment_start]
        active_gammas = coefficients[2 : 2 + len(active_knots)]
        slope = float(coefficients[1] + active_gammas.sum())
        intercept = float(coefficients[0] - sum(gamma * knot for gamma, knot in zip(active_gammas, active_knots)))
        segment_rows.append(
            {
                "segment_start": segment_start,
                "segment_end": float(segment["segment_end"]),
                "n": int(segment["n"]),
                "slope": slope,
                "intercept": intercept,
                "r": float(model_r),
            }
        )

    return {
        "start": start,
        "end": end,
        "knots": knots,
        "coefficients": coefficients,
        "segments": segment_rows,
    }


def rounded_limits(values: np.ndarray, pad: float = 2.0) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return -10.0, 10.0
    ymin = math.floor((values.min() - pad) / 5) * 5
    ymax = math.ceil((values.max() + pad) / 5) * 5
    if ymin == ymax:
        ymin -= 5
        ymax += 5
    return float(ymin), float(ymax)


def y_limits_for_panel(panel: PlotData, segment_years: int) -> tuple[float, float]:
    values = [panel.training["residual"].to_numpy(float), panel.observed["residual"].to_numpy(float)]
    train_fit = linear_fit(panel.training)
    lo, hi = AGE_LIMITS[panel.age_group]
    if train_fit is not None:
        slope, intercept, _ = train_fit
        values.append(np.array([slope * lo + intercept, slope * hi + intercept]))
    observed_fit = continuous_segmented_fit(panel.observed, panel.age_group, segment_years)
    if observed_fit is not None:
        x = np.linspace(float(observed_fit["start"]), float(observed_fit["end"]), 400)
        values.append(piecewise_predict(x, observed_fit["coefficients"], observed_fit["knots"]))
    return rounded_limits(np.concatenate(values) if values else np.array([]))


def style_axis(ax, panel: PlotData, segment_years: int) -> None:
    x_min, x_max = AGE_LIMITS[panel.age_group]
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(*y_limits_for_panel(panel, segment_years))
    ax.axhline(0, color=ZERO_COLOR, linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.75)
    ax.grid(color="#D5D5D5", linewidth=0.7, alpha=0.45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)


def draw_panel(
    ax,
    panel: PlotData,
    segment_years: int,
    measure: str,
    *,
    show_legend: bool = False,
) -> list[dict[str, object]]:
    style_axis(ax, panel, segment_years)
    ax.scatter(
        panel.training["age"],
        panel.training["residual"],
        s=8,
        color=TRAINING_COLOR,
        alpha=0.38,
        edgecolors="none",
        label="Training",
    )
    ax.scatter(
        panel.observed["age"],
        panel.observed["residual"],
        s=18,
        color=OBSERVED_COLOR,
        alpha=0.78,
        edgecolors="none",
        label=panel.group_label,
    )

    stats_rows: list[dict[str, object]] = []
    train_fit = linear_fit(panel.training)
    if train_fit is not None:
        slope, intercept, rvalue = train_fit
        x_min, x_max = AGE_LIMITS[panel.age_group]
        x = np.array([x_min, x_max])
        ax.plot(x, slope * x + intercept, color=TRAINING_COLOR, linewidth=2.4, label="Training fit")
        stats_rows.append(
            {
                "Group": panel.group_label,
                "Sex": panel.sex,
                "Age_Group": panel.age_group,
                "Data": "Training",
                "Fit_Mode": "linear",
                "Segment_Years": "all",
                "Segment_Start": x_min,
                "Segment_End": x_max,
                "N": len(panel.training),
                "Slope": slope,
                "Intercept": intercept,
                "r": rvalue,
            }
        )

    observed_fit = continuous_segmented_fit(panel.observed, panel.age_group, segment_years)
    if observed_fit is not None:
        x = np.linspace(float(observed_fit["start"]), float(observed_fit["end"]), 400)
        y = piecewise_predict(x, observed_fit["coefficients"], observed_fit["knots"])
        label = f"{panel.group_label} connected segmented fit" if show_legend else None
        ax.plot(x, y, color=OBSERVED_FIT_COLOR, linewidth=3.0, label=label)

    for fit in observed_fit["segments"] if observed_fit is not None else []:
        stats_rows.append(
            {
                "Group": panel.group_label,
                "Sex": panel.sex,
                "Age_Group": panel.age_group,
                "Data": "Observed",
                "Fit_Mode": "continuous_piecewise",
                "Segment_Years": segment_years,
                "Segment_Start": fit["segment_start"],
                "Segment_End": fit["segment_end"],
                "N": fit["n"],
                "Slope": fit["slope"],
                "Intercept": fit["intercept"],
                "r": fit["r"],
            }
        )

    ax.set_title(f"{panel.sex}, {panel.age_group}", fontsize=14, pad=8)
    ax.set_xlabel("Chronological age (years)", fontsize=10)
    ylabel = "Raw brain PAD (years)" if measure == "unadjusted" else "Adjusted brain BAG (years)"
    ax.set_ylabel(ylabel, fontsize=10)
    ax.text(
        0.97,
        0.94,
        f"n={len(panel.observed)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#303030",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )
    if show_legend:
        ax.legend(loc="lower left", frameon=True, framealpha=0.88, fontsize=9)
    return stats_rows


def make_combined_plot(
    panels: list[PlotData],
    segment_years: int,
    measure: str,
    group_label: str,
    output_dir: Path,
) -> tuple[Path, list[dict[str, object]]]:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), constrained_layout=True)
    rows: list[dict[str, object]] = []
    for idx, (ax, panel) in enumerate(zip(axes.ravel(), panels)):
        rows.extend(draw_panel(ax, panel, segment_years, measure, show_legend=idx == 0))
    measure_label = "Raw PAD" if measure == "unadjusted" else "Adjusted BAG"
    fig.suptitle(
        f"{group_label}: training fit with {segment_years}-year observed fits ({measure_label})",
        fontsize=17,
    )
    output = output_dir / f"{group_label}_combined_{measure}_segments_{segment_years}yr.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output, rows


def make_individual_plots(
    panels: list[PlotData],
    segment_years: int,
    measure: str,
    group_label: str,
    output_dir: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt

    outputs = []
    for panel in panels:
        fig, ax = plt.subplots(figsize=(7.3, 5.2), constrained_layout=True)
        draw_panel(ax, panel, segment_years, measure, show_legend=True)
        output = output_dir / f"{group_label}_{panel.sex}_{panel.age_group}_{measure}_segments_{segment_years}yr.png"
        fig.savefig(output, dpi=300)
        plt.close(fig)
        outputs.append(output)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group-key",
        default="self_chinese",
        help="Dataset key from brainage_dataset_config.py. Defaults to self_chinese.",
    )
    parser.add_argument(
        "--measure",
        choices=MEASURES,
        default="unadjusted",
        help="Use raw PAD (unadjusted) or adjusted BAG values. Defaults to unadjusted.",
    )
    parser.add_argument(
        "--segment-years",
        type=int,
        nargs="+",
        default=[5, 10],
        help="Observed-data regression segment width(s), in years. Defaults to 5 and 10.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_matplotlib()

    dataset_lookup = {key: label for key, label, _ in BRAINAGE_DATASETS}
    if args.group_key not in dataset_lookup:
        raise ValueError(f"Unknown group key {args.group_key!r}. Expected one of: {sorted(dataset_lookup)}")
    group_label = dataset_lookup[args.group_key]

    panels = load_plot_data(args.group_key, group_label, args.measure)
    output_dir = PLOT_ROOT / group_label / args.measure
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    outputs = []
    for segment_years in args.segment_years:
        if segment_years <= 0:
            raise ValueError("--segment-years values must be positive")
        combined, stats_rows = make_combined_plot(panels, segment_years, args.measure, group_label, output_dir)
        outputs.append(combined)
        outputs.extend(make_individual_plots(panels, segment_years, args.measure, group_label, output_dir))
        all_stats.extend(stats_rows)

    summary = pd.DataFrame(all_stats)
    summary_path = output_dir / f"{group_label}_{args.measure}_segment_fit_summary.csv"
    summary.to_csv(summary_path, index=False)

    for output in outputs:
        print(output)
    print(summary_path)


if __name__ == "__main__":
    main()
