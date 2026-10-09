"""Correct training/reference controls and compare them with corrected cohorts."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from correct import ROOT, DEFAULT_OUTPUT, correct_frame, plot_cohort, plt
from brainage_dataset_config import BRAINAGE_DATASETS
import plot_brainage_ridges as ridges


def reference_frame(source):
    """Adapt reference schema; row labels are provenance, not participant IDs."""
    frame = source.copy()
    frame["source_row"] = np.arange(1, len(frame) + 1)
    frame["SITE"] = "not_available"
    frame["SubjectID"] = "source_row_" + frame["source_row"].astype(str)
    frame["chronological_age"] = pd.to_numeric(frame["age"], errors="coerce")
    frame["Adjusted_BrainAGE"] = (
        pd.to_numeric(frame["predicted brain age_adjusted"], errors="coerce")
        - frame["chronological_age"])
    return frame


def corrected_rows(path):
    frame = pd.read_csv(path, dtype={"SITE": str, "SubjectID": str})
    return frame.loc[frame["correction_status"].eq("corrected")].copy()


def comparison_plot(label, color, cohorts, controls, column, output):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (sex, panel) in zip(axes.ravel(), ridges.PANELS):
        cohort = cohorts.get((sex, panel))
        if cohort is None or cohort.empty:
            ax.text(.5, .5, "No ethnoracial cohort available", ha="center", transform=ax.transAxes)
            ax.set_title(f"{sex}, {panel}")
            ax.axis("off")
            continue
        reference = controls[(sex, panel)]
        for frame, name, point_color, alpha in (
                (reference, "Training/reference controls", "#0072B2", .12),
                (cohort, label, "#D62728", .45)):
            x = frame["chronological_age"].to_numpy(float)
            y = frame[column].to_numpy(float)
            ax.scatter(x, y, s=10, color=point_color, alpha=alpha, rasterized=True)
            slope = np.polyfit(x - x.mean(), y, 1)[0]
            span = np.array([x.min(), x.max()])
            ax.plot(span, y.mean() + slope * (span - x.mean()), color=point_color,
                    linewidth=2, label=f"{name}: n={len(y):,}, mean={y.mean():+.2f}")
        ax.axhline(0, color="gray", linewidth=.6)
        ax.set(xlabel="Chronological age (years)", ylabel=f"{column} (years)",
               title=f"{sex}, {'5–40' if panel == '5-40' else '>40–90'}",
               xlim=(5, 40) if panel == "5-40" else (40, 90))
        ax.legend(fontsize=8, loc="best")
    measure = "Sample-specific correction" if column == "BrainAGE_S" else "Original CentileBrain adjustment"
    fig.suptitle(f"{label} and training/reference controls — {measure}", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run(output_root=DEFAULT_OUTPUT, training_root=ROOT / "training_data"):
    controls, summaries, provenance, plot_data = {}, [], [], []
    for sex, panel in ridges.PANELS:
        age_key = "age1234" if panel == "5-40" else "age56789"
        source_path = training_root / f"training_{age_key}_{sex.lower()}.csv"
        original = pd.read_csv(source_path)
        frame, summary = correct_frame(reference_frame(original), panel)
        if summary["Status"] != "corrected":
            raise ValueError(f"Reference controls cannot be corrected: {sex}, {panel}: {summary['Reason']}")
        directory = output_root / "training_controls" / f"{sex}_{panel}"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / "participants_BrainAGE_S.csv"
        frame.to_csv(destination, index=False)
        metadata = dict(Source_Path=str(source_path.resolve()),
                        Source_Mtime_UTC=datetime.fromtimestamp(source_path.stat().st_mtime, timezone.utc).isoformat(),
                        Source_SHA256=hashlib.sha256(source_path.read_bytes()).hexdigest())
        summaries.append(dict(Group="Training/reference controls", Sex=sex, Age_Panel=panel,
                              **summary, **metadata))
        provenance.append(dict(Sex=sex, Age_Panel=panel, **metadata,
                               Participant_File=str(destination.resolve())))
        plot_cohort(frame, summary, f"Training/reference controls, {sex}, {panel}", "#0072B2", directory)
        eligible = frame.loc[frame.correction_status.eq("corrected")].copy()
        controls[(sex, panel)] = eligible
        plot_data.append(pd.DataFrame(dict(dataset="Training", sex=sex, age_group=panel,
                                            BrainAGE=eligible.BrainAGE_S)))
    pd.DataFrame(summaries).to_csv(output_root / "training_controls" / "cohort_summary.csv", index=False)
    comparison_root = output_root / "control_comparisons"
    comparison_root.mkdir(exist_ok=True)
    # Only use cohorts listed by the current correction manifest, not stale files.
    cohort_manifest = json.loads((output_root / "run_manifest.json").read_text())
    available = {(c['Dataset'], c['Sex'], c['Age_Panel']): c
                 for c in cohort_manifest['Cohorts'] if c['Status'] == 'corrected'}
    for key, label, color in BRAINAGE_DATASETS:
        cohorts = {}
        for sex, panel in ridges.PANELS:
            entry = available.get((key, sex, panel))
            if entry is None:
                continue
            cohort = corrected_rows(Path(entry['Participant_File']))
            cohorts[(sex, panel)] = cohort
            plot_data.append(pd.DataFrame(dict(dataset=label, sex=sex, age_group=panel,
                                                BrainAGE=cohort.BrainAGE_S)))
        for column, suffix in (("Adjusted_BrainAGE", "original_adjusted"), ("BrainAGE_S", "sample_corrected")):
            comparison_plot(label, color, cohorts, controls, column,
                            comparison_root / f"{key}_{suffix}_age_vs_controls.png")
    data = pd.concat(plot_data, ignore_index=True)
    data.to_csv(comparison_root / "ridgeline_plot_data.csv", index=False)
    # Reuse previous distribution layout, colors, smoothing and axis limits.
    ridges.GRAPH_ROOT = comparison_root
    _, locator = ridges.setup_matplotlib()
    for panels, name in ((ridges.PANELS, "all"),
                         ([("Female", p) for p in ridges.AGE_PANELS], "female"),
                         ([("Male", p) for p in ridges.AGE_PANELS], "male")):
        ridges.make_plot(data, panels, "Sample-specific corrected",
                         f"self_{name}_BrainAGE_S_vs_controls_ridgelines.png", plt, locator)
    (output_root / "training_controls" / "run_manifest.json").write_text(json.dumps(dict(
        Run_Time_UTC=datetime.now(timezone.utc).isoformat(),
        Method="Separate controls-only OLS correction per sex and age panel; own mean age and slope",
        Identifier_Note="Source files lack IDs and site labels; source_row is a provenance label only",
        Controls=provenance, Cohort_Manifest=str((output_root / "run_manifest.json").resolve()),
        Cohort_Manifest_SHA256=hashlib.sha256((output_root / "run_manifest.json").read_bytes()).hexdigest(),
    ), indent=2) + "\n")
    print(f"Corrected four reference-control panels ({sum(s['N_Eligible'] for s in summaries):,} rows).")
    print(f"Comparisons: {comparison_root.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--training-root", type=Path, default=ROOT / "training_data")
    args = parser.parse_args()
    run(args.output_root, args.training_root)
