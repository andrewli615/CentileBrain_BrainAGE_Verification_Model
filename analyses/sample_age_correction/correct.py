"""Controls-only sample-specific linear age correction of adjusted BrainAGE."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from brainage_dataset_config import BRAINAGE_DATASETS, age_panel_rows

PANELS = [(sex, panel) for sex in ("Female", "Male") for panel in ("5-40", "40-90")]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs"


def discover(input_root):
    """Account for every configured panel, including absent input files."""
    for key, label, color in BRAINAGE_DATASETS:
        for sex, panel in PANELS:
            folder = input_root / key / f"{sex}_{panel}"
            files = list(folder.glob("*_Adjusted_BrainAGE_*.csv"))
            path = max(files, key=lambda p: (p.stat().st_mtime_ns, p.name)) if files else None
            yield key, label, color, sex, panel, path


def correct_frame(frame, panel):
    required = {"SITE", "SubjectID", "chronological_age", "Adjusted_BrainAGE"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    if panel not in ("5-40", "40-90"):
        raise ValueError(f"Unknown age panel: {panel}")
    result = frame.copy()
    age = pd.to_numeric(frame["chronological_age"], errors="coerce")
    bag = pd.to_numeric(frame["Adjusted_BrainAGE"], errors="coerce")
    finite_age, finite_bag = np.isfinite(age), np.isfinite(bag)
    in_panel = frame.index.isin(age_panel_rows(frame, panel).index)
    eligible = finite_age & finite_bag & in_panel
    reasons = []
    for a, b, inside in zip(finite_age, finite_bag, in_panel):
        reasons.append(";".join(reason for check, reason in (
            (not a, "missing_or_nonfinite_age"),
            (not b, "missing_or_nonfinite_brainage"),
            (a and not inside, "outside_age_panel"),
        ) if check))
    result["eligible"] = eligible.to_numpy()
    result["exclusion_reason"] = reasons
    result["correction_status"] = np.where(eligible, "pending", "excluded")
    for column in ("age_centered", "residual_age_effect", "BrainAGE_S"):
        result[column] = np.nan
    n = int(eligible.sum())
    summary = dict(N_Input=len(frame), N_Eligible=n, N_Excluded=len(frame) - n,
                   N_Unique_Ages=int(age[eligible].nunique()))
    if n:
        summary.update(Age_Min=float(age[eligible].min()), Age_Max=float(age[eligible].max()),
                       Mean_Age=float(age[eligible].mean()))
    skip = "fewer_than_three_eligible" if n < 3 else (
        "fewer_than_two_distinct_ages" if age[eligible].nunique() < 2 else None)
    if skip:
        result.loc[eligible, "correction_status"] = f"skipped:{skip}"
        summary.update(Status="skipped", Reason=skip)
        return result, summary
    x = age[eligible].to_numpy(float) - summary["Mean_Age"]
    y = bag[eligible].to_numpy(float)
    fit = stats.linregress(x, y)
    effect = fit.slope * x
    corrected = y - effect
    after = stats.linregress(x, corrected)
    half_ci = stats.t.ppf(.975, n - 2) * fit.stderr
    mean_change = float(corrected.mean() - y.mean())
    if not np.isclose(mean_change, 0, atol=1e-10, rtol=0):
        raise ArithmeticError("Correction failed to preserve the eligible cohort mean")
    if not np.isclose(after.slope, 0, atol=1e-10, rtol=0):
        raise ArithmeticError("Correction failed to remove the fitted linear slope")
    result.loc[eligible, "age_centered"] = x
    result.loc[eligible, "residual_age_effect"] = effect
    result.loc[eligible, "BrainAGE_S"] = corrected
    result.loc[eligible, "correction_status"] = "corrected"
    summary.update(Status="corrected", Reason="", Slope_Before=float(fit.slope),
                   Slope_CI95_Lower=float(fit.slope - half_ci),
                   Slope_CI95_Upper=float(fit.slope + half_ci), Slope_P=float(fit.pvalue),
                   Slope_After=float(after.slope), Mean_Before=float(y.mean()),
                   Mean_After=float(corrected.mean()), Mean_Change=mean_change,
                   SD_Before=float(y.std(ddof=1)), SD_After=float(corrected.std(ddof=1)),
                   Mean_Absolute_Before=float(np.abs(y).mean()),
                   Mean_Absolute_After=float(np.abs(corrected).mean()))
    return result, summary


def plot_cohort(frame, summary, title, color, directory):
    rows = frame.loc[frame["correction_status"] == "corrected"]
    age = pd.to_numeric(rows["chronological_age"]).to_numpy()
    before = pd.to_numeric(rows["Adjusted_BrainAGE"]).to_numpy()
    after = rows["BrainAGE_S"].to_numpy()
    span = np.array([age.min(), age.max()])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    for ax, values, label, slope in zip(axes, (before, after),
            ("CentileBrain adjusted BrainAGE", "BrainAGE_S"),
            (summary["Slope_Before"], summary["Slope_After"])):
        ax.scatter(age, values, s=15, alpha=.45, color=color)
        ax.plot(span, values.mean() + slope * (span - summary["Mean_Age"]), color="black")
        ax.axhline(0, color="gray", linewidth=.6)
        ax.set(title=label, xlabel="Chronological age (years)", ylabel="BrainAGE (years)")
    fig.suptitle(f"{title} | n={len(rows)} | ages {age.min():g}–{age.max():g}\n"
                 f"Original slope {summary['Slope_Before']:.3f}, 95% CI "
                 f"[{summary['Slope_CI95_Lower']:.3f}, {summary['Slope_CI95_Upper']:.3f}] years/year")
    fig.tight_layout()
    fig.savefig(directory / "age_before_after.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.histogram_bin_edges(np.concatenate([before, after]), bins="auto")
    ax.hist(before, bins=bins, alpha=.45, color="gray", label="CentileBrain adjusted BrainAGE")
    ax.hist(after, bins=bins, alpha=.5, color=color, label="BrainAGE_S")
    ax.set(title=f"{title} | n={len(rows)}", xlabel="BrainAGE (years)", ylabel="Participants")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "distribution_before_after.png", dpi=180)
    plt.close(fig)


def run(input_root, output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    summaries, manifest = [], []
    for key, label, color, sex, panel, path in discover(input_root):
        identity = dict(Dataset=key, Group=label, Sex=sex, Age_Panel=panel)
        if path is None:
            manifest.append({**identity, "Status": "missing_input"})
            continue
        metadata = dict(Source_Path=str(path.resolve()),
                        Source_Mtime_UTC=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                        Source_SHA256=hashlib.sha256(path.read_bytes()).hexdigest())
        # Keep identifiers as strings, including leading zeros and literal NA labels.
        frame = pd.read_csv(path, dtype={"SITE": str, "SubjectID": str}, keep_default_na=False)
        corrected, summary = correct_frame(frame, panel)
        directory = output_root / key / f"{sex}_{panel}"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / "participants_BrainAGE_S.csv"
        corrected.to_csv(destination, index=False)
        summary = {**identity, **metadata, **summary, "Participant_File": str(destination.resolve())}
        if summary["Status"] == "corrected":
            plot_cohort(corrected, summary, f"{label}, {sex}, {panel}", color, directory)
        summaries.append(summary)
        manifest.append({**identity, **metadata, "Status": summary["Status"],
                         "Participant_File": str(destination.resolve())})
    pd.DataFrame(summaries).to_csv(output_root / "cohort_summary.csv", index=False)
    (output_root / "run_manifest.json").write_text(json.dumps({
        "Run_Time_UTC": datetime.now(timezone.utc).isoformat(),
        "Method": "OLS with intercept; subtract slope times eligible-cohort centered age",
        "Correction_Gate": "all eligible cohorts regardless of p-value",
        "Input_Root": str(input_root.resolve()), "Cohorts": manifest,
    }, indent=2) + "\n")
    print(f"Corrected {sum(s['Status'] == 'corrected' for s in summaries)} cohorts; "
          f"skipped {sum(s['Status'] == 'skipped' for s in summaries)}; "
          f"missing {sum(s['Status'] == 'missing_input' for s in manifest)} panels.")
    print(f"Outputs: {output_root.resolve()}")
    return pd.DataFrame(summaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "ethnoracial_data_outputs")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.input_root, args.output_root)
