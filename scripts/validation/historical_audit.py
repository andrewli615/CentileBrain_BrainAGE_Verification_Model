"""Audit training sources, plotted coordinates, and reference-dependent figures."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.plots.sample_residuals import style, plt, draw_corrected_panel
from scripts.plots.compare_controls import corrected_rows, comparison_plot
from scripts.corrections.sample_age import plot_cohort
from scripts.plots import ridges

OUT = ROOT / "results/analyses/plot_audit"
checks = []


def check(name, actual, expected, tolerance=1e-10):
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0, equal_nan=True, err_msg=name)
    checks.append(name)


def reference(sex, panel, measure):
    key = "age1234" if panel == "5-40" else "age56789"
    source = pd.read_csv(ROOT / "training_data" / f"training_{key}_{sex.lower()}.csv")
    prediction = "predicted brain age" if measure == "raw" else "predicted brain age_adjusted"
    frame = pd.DataFrame(dict(age=source.age, residual=source[prediction] - source.age))
    valid = np.isfinite(frame.age) & np.isfinite(frame.residual)
    valid &= frame.age.between(5, 40) if panel == "5-40" else ((frame.age > 40) & (frame.age <= 90))
    return frame[valid].reset_index(drop=True)


def corrected_reference(sex, panel):
    path = ROOT / "results/analyses/sample_age_correction/training_controls" / f"{sex}_{panel}/participants_BrainAGE_S.csv"
    frame = corrected_rows(path)
    original = reference(sex, panel, "adjusted")
    check(f"{sex} {panel}: reference source rows", frame.chronological_age, original.age)
    check(f"{sex} {panel}: original adjusted reference", frame.Adjusted_BrainAGE, original.residual)
    expected = original.residual - stats.linregress(original.age, original.residual).slope * (original.age - original.age.mean())
    check(f"{sex} {panel}: reference BrainAGE_S", frame.BrainAGE_S, expected)
    return pd.DataFrame(dict(age=frame.chronological_age, residual=frame.BrainAGE_S)).reset_index(drop=True)


def verify_fits(table, refs, name):
    for row in table[table.Data.eq("Training")].itertuples():
        frame = refs[(row.Sex, row.Age_Group)]
        if row.Fit_Mode == "independent_segment":
            last = row.Segment_End == style.AGE_LIMITS[row.Age_Group][1]
            frame = frame[(frame.age >= row.Segment_Start) & ((frame.age <= row.Segment_End) if last else (frame.age < row.Segment_End))]
        check(f"{name} {row.Sex} {row.Age_Group} {row.Fit_Mode} {row.Segment_Years} {row.Segment_Start}: N", row.N, len(frame))
        if len(frame) >= 2 and frame.age.nunique() >= 2:
            fit = stats.linregress(frame.age, frame.residual)
            check(f"{name}: slope", row.Slope, fit.slope)
            check(f"{name}: intercept", row.Intercept, fit.intercept)


def verify_panel(panel, width, corrected=False):
    fig, ax = plt.subplots()
    try:
        if corrected:
            draw_corrected_panel(ax, panel, width, True)
        else:
            style.draw_panel(ax, panel, width, show_legend=True)
        check("Residual renderer: training scatter coordinates", ax.collections[0].get_offsets(), panel.training[["age", "residual"]].to_numpy())
        line = next(line for line in ax.lines if line.get_color() == style.TRAINING_LINEAR_COLOR)
        fit = stats.linregress(panel.training.age, panel.training.residual)
        check("Residual renderer: training global fitted line", line.get_ydata(), fit.slope * line.get_xdata() + fit.intercept)
        lines = [line for line in ax.lines if line.get_color() == style.TRAINING_SEGMENTED_COLOR]
        segments = style.independent_segmented_fit(panel.training, panel.age_group, width)["segments"]
        usable = [s for s in segments if s["can_plot"]]
        check("Residual renderer: training segment count", len(lines), len(usable))
        for line, segment in zip(lines, usable):
            check("Residual renderer: segment coordinates", line.get_ydata(), segment["slope"] * line.get_xdata() + segment["intercept"])
    finally:
        plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = []
    for path in sorted((ROOT/'results/analyses').rglob("*.png")):
        if "plot_audit" in path.parts or ".mplconfig" in path.parts:
            continue
        with Image.open(path) as img:
            img.verify()
        relative = path.relative_to(ROOT)
        training = ("residual_history" in relative.parts or "distribution_history" in relative.parts or
                    "brainage_gap" in relative.parts or "training_bucket_bias" in relative.parts or
                    "training_controls" in relative.parts or "control_comparisons" in relative.parts)
        inventory.append(dict(Figure=str(relative), Training_Reference=training))
    refs = {measure: {(sex, panel): reference(sex, panel, measure) for sex, panel in ridges.PANELS}
            for measure in ("raw", "adjusted")}
    refs["BrainAGE_S"] = {(sex, panel): corrected_reference(sex, panel) for sex, panel in ridges.PANELS}
    # Original and BrainAGE_S residual summaries and actual renderer coordinates.
    for key, label, _ in ridges.BRAINAGE_DATASETS:
        for folder, measure in (("unadjusted", "raw"), ("sample_corrected", "BrainAGE_S")):
            directory = ROOT / "results/analyses/residual_history" / label / folder
            summary = next(directory.glob("*segment_fit_summary.csv"))
            table = pd.read_csv(summary)
            verify_fits(table, refs[measure], f"{label} {folder}")
            panels = table[table.Data.eq("Training") & table.Fit_Mode.eq("linear")][["Sex", "Age_Group"]].drop_duplicates()
            for row in panels.itertuples(index=False):
                train = refs[measure][(row.Sex, row.Age_Group)]
                if measure == "raw":
                    observed = style.load_observed(key, row.Sex, row.Age_Group)
                else:
                    path = ROOT / "results/analyses/sample_age_correction" / key / f"{row.Sex}_{row.Age_Group}/participants_BrainAGE_S.csv"
                    observed = corrected_rows(path).rename(columns={"chronological_age": "age", "BrainAGE_S": "residual"})[["age", "residual"]]
                panel = style.PlotData(key, label, row.Sex, row.Age_Group, train, observed)
                for width in (5, 10):
                    verify_panel(panel, width, corrected=measure == "BrainAGE_S")
    # Gaussian training ridges: verify loading and curve coordinates independently.
    plt_module, locator = ridges.setup_matplotlib()
    sample_root = ROOT / "results/analyses/sample_age_correction"
    for measure in ("Adjusted", "Unadjusted", "BrainAGE_S"):
        if measure == "BrainAGE_S":
            data = pd.read_csv(sample_root / "control_comparisons/ridgeline_plot_data.csv")
        else:
            data = ridges.build_plot_data(measure)
        expected_refs = refs["BrainAGE_S" if measure == "BrainAGE_S" else ("adjusted" if measure == "Adjusted" else "raw")]
        for sex, panel in ridges.PANELS:
            values = data[(data.dataset == "Training") & (data.sex == sex) & (data.age_group == panel)].BrainAGE.to_numpy()
            check(f"{measure} ridge training inputs", values, expected_refs[(sex, panel)].residual)
            fig, ax = plt.subplots()
            grid = np.linspace(*ridges.X_LIMITS, 500)
            ridges.draw_panel(ax, data, sex, panel, grid, locator)
            expected = np.exp(-.5 * ((grid - values.mean()) / values.std(ddof=1)) ** 2)
            expected = expected / expected.max() * ridges.RIDGE_HEIGHT + len(ridges.BRAINAGE_ROW_ORDER) - 1
            check(f"{measure} ridge Gaussian curve", ax.lines[0].get_ydata(), expected)
            plt.close(fig)
    # Capture the exact comparison/control diagnostic renderers without rewriting plots.
    from matplotlib.figure import Figure
    original_savefig = Figure.savefig
    captured = []
    Figure.savefig = lambda self, *args, **kwargs: captured.append(self)
    try:
        control_frames = {(sex, panel): corrected_rows(sample_root / "training_controls" / f"{sex}_{panel}/participants_BrainAGE_S.csv")
                          for sex, panel in ridges.PANELS}
        for key, label, color in ridges.BRAINAGE_DATASETS:
            cohorts = {}
            for sex, panel in ridges.PANELS:
                path = sample_root / key / f"{sex}_{panel}/participants_BrainAGE_S.csv"
                if path.exists(): cohorts[(sex, panel)] = corrected_rows(path)
            for column in ("Adjusted_BrainAGE", "BrainAGE_S"):
                comparison_plot(label, color, cohorts, control_frames, column, Path("audit-not-saved.png"))
                fig = captured.pop()
                for ax, pair in zip(fig.axes, ridges.PANELS):
                    if pair not in cohorts: continue
                    frame = control_frames[pair]
                    check("Comparison training scatter", ax.collections[0].get_offsets(), frame[["chronological_age", column]])
                    fit = stats.linregress(frame.chronological_age, frame[column])
                    check("Comparison training fitted line", ax.lines[0].get_ydata(), fit.slope * ax.lines[0].get_xdata() + fit.intercept)
        control_summary = pd.read_csv(sample_root / "training_controls/cohort_summary.csv")
        for row in control_summary.itertuples():
            frame = control_frames[(row.Sex, row.Age_Panel)]
            plot_cohort(frame, row._asdict(), f"Training/reference controls, {row.Sex}, {row.Age_Panel}", "#0072B2", Path("."))
            age_figure, histogram_figure = captured[-2:]
            for ax, column in zip(age_figure.axes, ("Adjusted_BrainAGE", "BrainAGE_S")):
                check("Control diagnostic scatter", ax.collections[0].get_offsets(), frame[["chronological_age", column]])
                fit = stats.linregress(frame.chronological_age, frame[column])
                check("Control diagnostic fitted line", ax.lines[0].get_ydata(), fit.slope * ax.lines[0].get_xdata() + fit.intercept)
            check("Control histogram before total", sum(p.get_height() for p in histogram_figure.axes[0].patches[:len(histogram_figure.axes[0].patches)//2]), len(frame))
            check("Control histogram after total", sum(p.get_height() for p in histogram_figure.axes[0].patches[len(histogram_figure.axes[0].patches)//2:]), len(frame))
            captured.clear()
    finally:
        Figure.savefig = original_savefig
    # Independently reconstruct Chinese corrections and both plotted datasets.
    bucket_root = ROOT / "results/analyses/training_bucket_bias"
    coefficients = pd.read_csv(bucket_root / "training_bucket_coefficients.csv")
    manifest = json.loads((bucket_root / "run_manifest.json").read_text())
    sources = {(s["Role"], s["Sex"], s["Age_Panel"]): Path(s["Path"]) for s in manifest["Sources"]}
    cohort_summary = pd.read_csv(bucket_root / "cohort_summary.csv")
    bucket_observed = {}
    for sex, panel in ridges.PANELS:
        d = pd.read_csv(bucket_root / "training_controls" / f"{sex}_{panel}/participants_training_bucket_adjusted.csv")
        d = d[d.Correction_Status.eq("corrected")]
        check("Bucket raw training source", d.Raw_BrainAGE, refs["raw"][(sex, panel)].residual)
        check("Bucket correction subtraction", d.BrainAGE_TrainingBucketAdjusted, d.Raw_BrainAGE - (d.Training_Slope_a*d.chronological_age+d.Training_Intercept_b))
        source = pd.read_csv(sources[("Chinese", sex, panel)])
        exported = pd.read_csv(bucket_root / "self_chinese" / f"{sex}_{panel}/participants_training_bucket_adjusted.csv")
        check("Chinese raw source prediction minus age", exported.Raw_BrainAGE, source.MR_predicted_age-source.chronological_age)
        valid_age = source.chronological_age.between(5, 40) if panel == "5-40" else ((source.chronological_age > 40) & (source.chronological_age <= 90))
        expected = np.full(len(source), np.nan)
        training = refs["raw"][(sex, panel)]
        for row in coefficients[(coefficients.Sex == sex) & (coefficients.Age_Panel == panel)].itertuples():
            last = row.Bucket_End == style.AGE_LIMITS[panel][1]
            train_mask = (training.age >= row.Bucket_Start) & ((training.age <= row.Bucket_End) if last else (training.age < row.Bucket_End))
            local = training[train_mask]
            check("Bucket coefficient training N", row.N_Training, len(local))
            if row.Status != "fitted":
                continue
            fit = stats.linregress(local.age, local.residual)
            check("Bucket coefficient raw training slope", row.Slope_a, fit.slope)
            check("Bucket coefficient raw training intercept", row.Intercept_b, fit.intercept)
            mask = valid_age & (source.chronological_age >= row.Bucket_Start) & ((source.chronological_age <= row.Bucket_End) if last else (source.chronological_age < row.Bucket_End))
            expected[mask] = (source.MR_predicted_age-source.chronological_age)[mask] - (fit.slope*source.chronological_age[mask]+fit.intercept)
        check("Chinese independently reconstructed adjustment", exported.BrainAGE_TrainingBucketAdjusted, expected)
        check("Chinese corrected row selection", exported.Correction_Status.eq("corrected"), np.isfinite(expected))
        eligible = exported[exported.Correction_Status.eq("corrected")]
        summary = cohort_summary[(cohort_summary.Group == "Chinese") & (cohort_summary.Sex == sex) & (cohort_summary.Age_Panel == panel)].iloc[0]
        check("Chinese corrected N", summary.N_Corrected, len(eligible))
        check("Chinese adjusted MAE", summary.MAE_Adjusted, eligible.BrainAGE_TrainingBucketAdjusted.abs().mean())
        bucket_observed[(sex, panel)] = eligible
    for suffix, expected_refs in (("raw", refs["raw"]), ("training_bucket_adjusted", refs["raw"])):
        table = pd.read_csv(bucket_root / f"{suffix}_segment_fit_summary.csv")
        verify_fits(table, expected_refs, f"Chinese bucket {suffix}")
        assert table.Training_Measure.eq("raw").all()
        column = "Raw_BrainAGE" if suffix == "raw" else "BrainAGE_TrainingBucketAdjusted"
        observed_refs = {pair: pd.DataFrame(dict(age=d.chronological_age, residual=d[column])) for pair, d in bucket_observed.items()}
        observed_table = table[table.Data.eq("Observed")].copy()
        observed_table["Data"] = "Training"
        verify_fits(observed_table, observed_refs, f"Chinese observed {suffix}")
        for sex, panel in ridges.PANELS:
            plot_data = style.PlotData("self_chinese", "Chinese", sex, panel, expected_refs[(sex, panel)], observed_refs[(sex, panel)])
            verify_panel(plot_data, manifest["Bucket_Width"])
            fig, ax = plt.subplots()
            try:
                style.draw_panel(ax, plot_data, manifest["Bucket_Width"], show_legend=True)
                check("Chinese renderer observed scatter", ax.collections[1].get_offsets(), plot_data.observed[["age", "residual"]])
                fit = stats.linregress(plot_data.observed.age, plot_data.observed.residual)
                line = next(line for line in ax.lines if line.get_color() == style.OBSERVED_LINEAR_COLOR)
                check("Chinese renderer observed global line", line.get_ydata(), fit.slope*line.get_xdata()+fit.intercept)
                segments = style.independent_segmented_fit(plot_data.observed, panel, manifest["Bucket_Width"])["segments"]
                usable = [s for s in segments if s["can_plot"]]
                lines = [line for line in ax.lines if line.get_color() == style.OBSERVED_SEGMENTED_COLOR]
                check("Chinese renderer observed segment count", len(lines), len(usable))
                for line, segment in zip(lines, usable):
                    check("Chinese renderer observed segment", line.get_ydata(), segment["slope"]*line.get_xdata()+segment["intercept"])
            finally:
                plt.close(fig)
    # Inferential histogram inputs: these use the original adjusted reference, not BrainAGE_S.
    for kind, delta_column in (("bootstrap", "Bootstrap_Delta"), ("permutation", "Permutation_Delta")):
        directory = ROOT / "results/analyses/brainage_gap" / kind
        summary = pd.read_csv(directory / f"two_sample_{kind}_summary.csv")
        samples = pd.read_csv(directory / f"two_sample_{kind}_samples.csv")
        for row in summary.itertuples():
            ref = refs["adjusted"][(row.Sex, row.Age_Group)].residual.to_numpy()
            if row.Metric == "MAE": ref = np.abs(ref)
            check(f"{kind} reference N", row.N_Ref, len(ref))
            check(f"{kind} reference mean", row.Ref_Mean, ref.mean())
            check(f"{kind} observed delta", row.Observed_Delta, row.Test_Mean-ref.mean())
            draws = samples[(samples.Group==row.Group)&(samples.Sex==row.Sex)&(samples.Age_Group==row.Age_Group)&(samples.Metric==row.Metric)][delta_column].to_numpy()
            check(f"{kind} iteration count", len(draws), 10000)
            ci = np.percentile(draws, [2.5, 97.5])
            check(f"{kind} plotted interval", ci, [row.CI_95_Lower, row.CI_95_Upper] if kind == "bootstrap" else [row.Null_95_Lower, row.Null_95_Upper])
            check(f"{kind} distribution mean", draws.mean(), row.Bootstrap_Mean_Delta if kind=="bootstrap" else row.Null_Mean_Delta)
    # Existing input checksum manifests establish that the saved plots use unchanged sources.
    manifests = [sample_root / "run_manifest.json", sample_root / "training_controls/run_manifest.json", bucket_root / "run_manifest.json"]
    for path in manifests:
        manifest = json.loads(path.read_text())
        for entry in manifest.get("Cohorts", manifest.get("Controls", manifest.get("Sources", []))):
            if entry.get("Status") == "missing_input": continue
            source = Path(entry.get("Source_Path", entry.get("Path")))
            expected = entry.get("Source_SHA256", entry.get("SHA256"))
            assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, source
            checks.append(f"Input checksum: {source.relative_to(ROOT)}")
    result = dict(Figures=len(inventory), Training_Reference_Figures=sum(r['Training_Reference'] for r in inventory),
                  No_Training_Figures=sum(not r['Training_Reference'] for r in inventory), Checks_Passed=len(checks),
                  Notes=["Training ridges use a Gaussian approximation; labels now explicitly say Gaussian fit.",
                         "BrainAGE_S global training slopes are zero by construction; local segments need not be flat.",
                         "Bucket-adjustment plots use RAW training; Chinese plots show raw or training-bucket-adjusted values. Duplicate diagnostic removed.",
                         "Chinese corrections, coefficients, observed regressions, MAEs, and renderer coordinates were checked against source predictions.",
                         "The repository contains exported predictions, not the fitted CentileBrain prediction model; upstream inference cannot be reproduced here.",
                         "Bootstrap/permutation use original adjusted references, not BrainAGE_S.",
                         "Q-Q plots use a theoretical normal reference, not training controls.",
                         "Pixel images do not retain source coordinates; numerical verification uses saved summaries, source checksums, and current renderer artists."])
    (OUT / "training_plot_audit.json").write_text(json.dumps(dict(**result, Figure_Inventory=inventory), indent=2)+"\n")
    (OUT / "training_plot_audit.md").write_text(
        "# Training data plot audit\n\n" +
        f"Verified {result['Figures']} PNG files: {result['Training_Reference_Figures']} display training data or depend on a training reference, and {result['No_Training_Figures']} contain no training data.\n\n"+
        f"Passed {len(checks)} numerical and provenance checks. Image integrity was checked for every PNG.\n\n"+
        "Checks cover sex/age selection, raw and adjusted reference values, BrainAGE_S formula, scatter coordinates, global and segmented regressions, Gaussian ridge curves, reference counts/means, bootstrap and permutation intervals, and source checksums.\n\n"+
        "\n".join(f"- {note}" for note in result['Notes'])+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
