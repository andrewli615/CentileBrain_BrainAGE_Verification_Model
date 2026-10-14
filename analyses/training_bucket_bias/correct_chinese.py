"""Chinese raw BrainAGE minus sex/panel/bucket-specific training age regression."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analyses" / "sample_age_correction"))
from plot_residuals import style, plt

PANELS = style.PANELS
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs"


def assign_buckets(ages, panel, width=10):
    """Left-closed internal bins; final endpoint included; age 40 is younger only."""
    if panel not in ("5-40", "40-90") or width <= 0:
        raise ValueError("Invalid age panel or bucket width")
    age = pd.to_numeric(ages, errors="coerce")
    result = pd.Series(np.nan, index=ages.index)
    valid = np.isfinite(age) & (age.between(5, 40) if panel == "5-40" else ((age > 40) & (age <= 90)))
    for idx, (lo, hi) in enumerate(style.fixed_segments(panel, width)):
        last = hi == style.AGE_LIMITS[panel][1]
        mask = valid & (age >= lo) & ((age <= hi) if last else (age < hi))
        result.loc[mask] = idx
    return result


def fit_training(frame, panel, width=10):
    age = pd.to_numeric(frame.chronological_age, errors="coerce")
    bag = pd.to_numeric(frame.Raw_BrainAGE, errors="coerce")
    buckets = assign_buckets(age, panel, width)
    rows = []
    for idx, (lo, hi) in enumerate(style.fixed_segments(panel, width)):
        valid = buckets.eq(idx) & np.isfinite(bag)
        x, y = age[valid].to_numpy(float), bag[valid].to_numpy(float)
        row = dict(Bucket_Index=idx, Bucket_Start=lo, Bucket_End=hi, N_Training=len(x))
        if len(x) < 3 or len(np.unique(x)) < 2:
            row.update(Status="unfit", Reason="insufficient_training_data")
        else:
            fit = stats.linregress(x, y)
            half = stats.t.ppf(.975, len(x) - 2) * fit.stderr
            row.update(Status="fitted", Reason="", Training_Age_Min=float(x.min()),
                       Training_Age_Max=float(x.max()), Slope_a=float(fit.slope), Intercept_b=float(fit.intercept),
                       Slope_CI95_Lower=float(fit.slope - half), Slope_CI95_Upper=float(fit.slope + half))
        rows.append(row)
    return pd.DataFrame(rows)


def apply_training(frame, fits, panel, width=10):
    result = frame.copy()
    age = pd.to_numeric(frame.chronological_age, errors="coerce")
    raw = pd.to_numeric(frame.Raw_BrainAGE, errors="coerce")
    buckets = assign_buckets(age, panel, width)
    result["Bucket_Index"] = buckets.astype("Int64")
    result["Correction_Status"] = "excluded"
    result["Exclusion_Reason"] = ""
    for column in ("Bucket_Start", "Bucket_End", "Training_Slope_a", "Training_Intercept_b",
                   "Estimated_Training_Bias", "BrainAGE_TrainingBucketAdjusted"):
        result[column] = np.nan
    result["Outside_Training_Age_Support"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    for idx in frame.index:
        reasons = []
        if not np.isfinite(age.loc[idx]): reasons.append("missing_or_nonfinite_age")
        elif pd.isna(buckets.loc[idx]): reasons.append("outside_age_panel")
        if not np.isfinite(raw.loc[idx]): reasons.append("missing_or_nonfinite_brainage")
        result.loc[idx, "Exclusion_Reason"] = ";".join(reasons)
    for fit in fits.to_dict("records"):
        mask = buckets.eq(fit["Bucket_Index"]) & np.isfinite(raw)
        if fit["Status"] != "fitted":
            result.loc[mask, "Correction_Status"] = "skipped"
            result.loc[mask, "Exclusion_Reason"] = fit["Reason"]
            continue
        bias = fit["Slope_a"] * age[mask] + fit["Intercept_b"]
        result.loc[mask, "Bucket_Start"] = fit["Bucket_Start"]
        result.loc[mask, "Bucket_End"] = fit["Bucket_End"]
        result.loc[mask, "Training_Slope_a"] = fit["Slope_a"]
        result.loc[mask, "Training_Intercept_b"] = fit["Intercept_b"]
        result.loc[mask, "Estimated_Training_Bias"] = bias
        result.loc[mask, "BrainAGE_TrainingBucketAdjusted"] = raw[mask] - bias
        result.loc[mask, "Outside_Training_Age_Support"] = ~age[mask].between(fit["Training_Age_Min"], fit["Training_Age_Max"])
        result.loc[mask, "Correction_Status"] = "corrected"
    return result


def plot_frame(frame, column):
    d = frame.loc[frame.Correction_Status.eq("corrected")]
    return pd.DataFrame({"age": pd.to_numeric(d.chronological_age), "residual": pd.to_numeric(d[column])})


def plot_panels(panels, title, output, width, training_measure="raw", chinese_measure="raw"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), constrained_layout=True)
    rows = []
    for i, (ax, panel) in enumerate(zip(axes.ravel(), panels)):
        rows.extend(style.draw_panel(ax, panel, width, show_legend=i == 0))
        for artist in ax.get_children():
            label = artist.get_label() if hasattr(artist, "get_label") else None
            if isinstance(label, str) and label.startswith("Training"):
                artist.set_label(label.replace("Training", f"Training ({training_measure})", 1))
            elif isinstance(label, str) and label.startswith("Chinese"):
                artist.set_label(label.replace("Chinese", f"Chinese ({chinese_measure})", 1))
        if i == 0:
            ax.legend(loc="lower left", frameon=True, framealpha=.88, fontsize=8)
        ax.set_ylabel("BrainAGE (years)")
    fig.suptitle(title, fontsize=16)
    fig.savefig(output, dpi=240)
    plt.close(fig)
    return rows


def source_record(role, sex, panel, path):
    return dict(Role=role, Sex=sex, Age_Panel=panel, Path=str(path.resolve()),
                SHA256=hashlib.sha256(path.read_bytes()).hexdigest())


def run(output_root=DEFAULT_OUTPUT, width=10):
    output_root.mkdir(parents=True, exist_ok=True)
    all_fits, summaries, sources, bucket_summaries = [], [], [], []
    before_panels, after_panels, diagnostic_panels = [], [], []
    for sex, panel in PANELS:
        key = "age1234" if panel == "5-40" else "age56789"
        training_path = ROOT / "training_data" / f"training_{key}_{sex.lower()}.csv"
        training = pd.read_csv(training_path)
        training["source_row"] = np.arange(1, len(training) + 1)
        training["chronological_age"] = pd.to_numeric(training.age, errors="coerce")
        training["Raw_BrainAGE"] = pd.to_numeric(training["predicted brain age"], errors="coerce") - training.chronological_age
        folder = ROOT / "ethnoracial_data_outputs" / "self_chinese" / f"{sex}_{panel}"
        candidates = [p for p in folder.glob("*_MR_predicted_age_*.csv") if "_Adjusted_" not in p.name]
        path = max(candidates, key=lambda p: (p.stat().st_mtime_ns, p.name))
        chinese = pd.read_csv(path, dtype={"SITE": str, "SubjectID": str}, keep_default_na=False)
        calculated = pd.to_numeric(chinese.MR_predicted_age, errors="coerce") - pd.to_numeric(chinese.chronological_age, errors="coerce")
        if "Unadjusted_BrainAGE" in chinese and not np.allclose(pd.to_numeric(chinese.Unadjusted_BrainAGE, errors="coerce"), calculated, equal_nan=True):
            raise ValueError(f"Raw BrainAGE does not match predicted age minus age: {path}")
        chinese["Raw_BrainAGE"] = calculated
        fits = fit_training(training, panel, width)
        corrected_training = apply_training(training, fits, panel, width)
        corrected_chinese = apply_training(chinese, fits, panel, width)
        all_fits.append(fits.assign(Sex=sex, Age_Panel=panel))
        for label, data in (("Training/reference controls", corrected_training), ("Chinese", corrected_chinese)):
            eligible = data.loc[data.Correction_Status.eq("corrected")]
            a = eligible.Raw_BrainAGE.to_numpy(float)
            b = eligible.BrainAGE_TrainingBucketAdjusted.to_numpy(float)
            summaries.append(dict(Group=label, Sex=sex, Age_Panel=panel, N_Input=len(data), N_Corrected=len(eligible),
                                  N_Excluded=int(data.Correction_Status.eq("excluded").sum()),
                                  N_Skipped=int(data.Correction_Status.eq("skipped").sum()),
                                  N_Outside_Training_Support=int(eligible.Outside_Training_Age_Support.sum()),
                                  Mean_Raw=float(a.mean()), Mean_Adjusted=float(b.mean()),
                                  MAE_Raw=float(np.abs(a).mean()), MAE_Adjusted=float(np.abs(b).mean()),
                                  SD_Raw=float(a.std(ddof=1)), SD_Adjusted=float(b.std(ddof=1))))
            for bucket, d in eligible.groupby("Bucket_Index"):
                bucket_summaries.append(dict(Group=label, Sex=sex, Age_Panel=panel, Bucket_Index=int(bucket),
                    Bucket_Start=float(d.Bucket_Start.iloc[0]), Bucket_End=float(d.Bucket_End.iloc[0]), N=len(d),
                    Mean_Raw=float(d.Raw_BrainAGE.mean()), Mean_Adjusted=float(d.BrainAGE_TrainingBucketAdjusted.mean()),
                    MAE_Raw=float(d.Raw_BrainAGE.abs().mean()), MAE_Adjusted=float(d.BrainAGE_TrainingBucketAdjusted.abs().mean())))
            role = "training_controls" if label.startswith("Training") else "self_chinese"
            directory = output_root / role / f"{sex}_{panel}"
            directory.mkdir(parents=True, exist_ok=True)
            data.to_csv(directory / "participants_training_bucket_adjusted.csv", index=False)
        for training_column, chinese_column, panels in (
                ("Raw_BrainAGE", "Raw_BrainAGE", before_panels),
                ("Raw_BrainAGE", "BrainAGE_TrainingBucketAdjusted", after_panels),
                ("BrainAGE_TrainingBucketAdjusted", "BrainAGE_TrainingBucketAdjusted", diagnostic_panels)):
            panels.append(style.PlotData("self_chinese", "Chinese", sex, panel,
                                         plot_frame(corrected_training, training_column), plot_frame(corrected_chinese, chinese_column)))
        sources.extend([source_record("Training", sex, panel, training_path), source_record("Chinese", sex, panel, path)])
    pd.concat(all_fits, ignore_index=True).to_csv(output_root / "training_bucket_coefficients.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_root / "cohort_summary.csv", index=False)
    pd.DataFrame(bucket_summaries).to_csv(output_root / "bucket_summary.csv", index=False)
    for panels, suffix, title, training_measure, chinese_measure in (
            (before_panels, "raw", "Chinese and controls: raw BrainAGE", "raw", "raw"),
            (after_panels, "training_bucket_adjusted", "Adjusted Chinese BrainAGE with raw training reference", "raw", "adjusted"),
            (diagnostic_panels, "adjusted_controls_diagnostic", "Diagnostic: both training controls and Chinese BrainAGE adjusted", "adjusted", "adjusted")):
        rows = plot_panels(panels, f"{title} — {width}-year buckets", output_root / f"Chinese_{suffix}_{width}yr.png",
                           width, training_measure, chinese_measure)
        pd.DataFrame(rows).assign(Training_Measure=training_measure, Chinese_Measure=chinese_measure).to_csv(
            output_root / f"{suffix}_segment_fit_summary.csv", index=False)
    (output_root / "run_manifest.json").write_text(json.dumps(dict(
        Run_Time_UTC=datetime.now(timezone.utc).isoformat(), Bucket_Width=width,
        Formula="Chinese raw BrainAGE - (training slope a * chronological age + training intercept b)",
        Training_Outcome="predicted brain age - chronological age", Sources=sources,
        Main_Plot_Training_Measure="Raw BrainAGE; original training regression used for correction",
        Main_Plot_Chinese_Measure="BrainAGE_TrainingBucketAdjusted",
        Age_Boundaries="5<=age<=40; 40<age<=90; internal buckets left-closed/right-open; final endpoint included",
    ), indent=2) + "\n")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"Outputs: {output_root.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bucket-years", type=int, default=10)
    args = parser.parse_args()
    run(args.output_root, args.bucket_years)
