"""Plot corrected BrainAGE using the existing residual plotting style."""

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from correct import ROOT, DEFAULT_OUTPUT, plt
from brainage_dataset_config import BRAINAGE_DATASETS
from compare_controls import corrected_rows

# Reuse the existing panel renderer and segmented fits without changing raw PAD plots.
spec = importlib.util.spec_from_file_location(
    "brainage_residual_plot_style", ROOT / "Residual plotting" / "generate_residual_plots.py")
style = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = style
spec.loader.exec_module(style)


def residual_frame(path):
    frame = corrected_rows(path)
    return pd.DataFrame({"age": pd.to_numeric(frame["chronological_age"]),
                         "residual": pd.to_numeric(frame["BrainAGE_S"])})


def render(panels, label, width, directory):
    nrows, ncols, figsize = style.combined_plot_layout(len(panels))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    rows = []
    for idx, (ax, panel) in enumerate(zip(axes, panels)):
        rows.extend(style.draw_panel(ax, panel, width, show_legend=idx == 0))
        ax.set_ylabel("BrainAGE_S (years)")
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(f"{label}: linear and {width}-year segmented fits (BrainAGE_S)", fontsize=17)
    fig.savefig(directory / f"{label}_combined_BrainAGE_S_segments_{width}yr.png", dpi=300)
    plt.close(fig)
    for panel in panels:
        fig, ax = plt.subplots(figsize=(7.3, 5.2), constrained_layout=True)
        style.draw_panel(ax, panel, width, show_legend=True)
        ax.set_ylabel("BrainAGE_S (years)")
        ax.set_title(f"{panel.sex}, {panel.age_group} — BrainAGE_S")
        fig.savefig(directory / f"{label}_{panel.sex}_{panel.age_group}_BrainAGE_S_segments_{width}yr.png", dpi=300)
        plt.close(fig)
    return rows


def run(input_root=DEFAULT_OUTPUT, output_root=ROOT / "Residual plotting", widths=(5, 10), group_key=None):
    if any(width <= 0 for width in widths):
        raise ValueError("Segment widths must be positive")
    cohort_manifest = json.loads((input_root / "run_manifest.json").read_text())
    control_manifest = json.loads((input_root / "training_controls" / "run_manifest.json").read_text())
    cohorts = {(c["Dataset"], c["Sex"], c["Age_Panel"]): c for c in cohort_manifest["Cohorts"]
               if c["Status"] == "corrected"}
    control_paths = {(c["Sex"], c["Age_Panel"]): Path(c["Participant_File"])
                     for c in control_manifest["Controls"]}
    controls = {panel: residual_frame(path) for panel, path in control_paths.items()}
    group_count, plot_count = 0, 0
    lookup = {key for key, _, _ in BRAINAGE_DATASETS}
    if group_key is not None and group_key not in lookup:
        raise ValueError(f"Unknown group: {group_key}")
    for key, label, _ in BRAINAGE_DATASETS:
        if group_key is not None and key != group_key:
            continue
        panels, sources = [], []
        for sex, panel in style.PANELS:
            entry = cohorts.get((key, sex, panel))
            if entry is None:
                continue
            cohort_path = Path(entry["Participant_File"])
            observed = residual_frame(cohort_path)
            if observed.empty:
                continue
            panels.append(style.PlotData(key, label, sex, panel, controls[(sex, panel)], observed))
            for role, path in (("Training", control_paths[(sex, panel)]), ("Observed", cohort_path)):
                sources.append(dict(Role=role, Sex=sex, Age_Panel=panel, Path=str(path.resolve()),
                                    SHA256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if not panels:
            continue
        directory = output_root / label / "sample_corrected"
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        for width in widths:
            rows.extend(render(panels, label, width, directory))
            plot_count += len(panels) + 1
        summary = pd.DataFrame(rows)
        summary["Measure"] = "BrainAGE_S"
        summary.to_csv(directory / f"{label}_BrainAGE_S_segment_fit_summary.csv", index=False)
        (directory / "plot_manifest.json").write_text(json.dumps(dict(
            Run_Time_UTC=datetime.now(timezone.utc).isoformat(), Measure="BrainAGE_S",
            Segment_Years=list(widths), Sources=sources,
            Note="Segment fits are descriptive diagnostics; no additional segmented correction is applied.",
        ), indent=2) + "\n")
        group_count += 1
    print(f"Generated {plot_count} residual-style BrainAGE_S figures across {group_count} groups.")
    print(f"Output: {output_root.resolve()}/<group>/sample_corrected/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "Residual plotting")
    parser.add_argument("--segment-years", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--group-key", default=None, help="Default: all seven active groups")
    args = parser.parse_args()
    run(args.input_root, args.output_root, args.segment_years, args.group_key)
