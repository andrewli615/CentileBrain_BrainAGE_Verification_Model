# Sample-specific age correction of BrainAGE

This analysis applies the supplied controls-only method to **already age-adjusted
CentileBrain BrainAGE**. It produces `BrainAGE_S` separately for each active
self-reported ethnoracial group, sex, and age panel. It does not retrain
CentileBrain or change its original adjustment.

## Run

From the project root, with the packages in `requirements.txt` installed:

```bash
python3 analyses/sample_age_correction/correct.py
python3 analyses/sample_age_correction/compare_controls.py
python3 -m unittest discover -s analyses/sample_age_correction -p 'test_*.py'
```

Optional `--input-root` and `--output-root` arguments select alternative input
and output directories. The default output is `analyses/sample_age_correction/outputs/`.
Rerunning replaces outputs for the current cohorts; use a separate output root
to preserve a historical run. Consult the current manifest for the authoritative
cohort list rather than interpreting files left from a previous run.

## Method and eligible participants

The active groups are Black, Chilean, Chinese, Japanese, Mexican, South Asian,
and Turkish. Each Female/Male and 5–40/>40–90 panel is considered independently.
White and genetic ancestry groups are excluded. All included samples are
assumed to contain controls only, as confirmed for this analysis.

Use finite chronological age and finite `Adjusted_BrainAGE`, restricted to
5 ≤ age ≤ 40 or 40 < age ≤ 90. Calculate mean age from exactly the same rows
used to fit the slope. Fit ordinary least squares with an intercept:

```text
age_centered = chronological_age − eligible cohort mean age
Adjusted_BrainAGE = intercept + β × age_centered + residual
residual_age_effect = β × age_centered
BrainAGE_S = Adjusted_BrainAGE − residual_age_effect
```

Subtract the age term only, preserving the cohort mean. Apply the correction
to every eligible cohort, regardless of the original slope p-value. This follows
the supplied controls-only script; no significance or practical-effect gate is
used. At least three eligible participants and two distinct ages are needed.

The source method is `Correcting Persistent Residual Effects of age on brainAGE.docx`
(supplied October 2026). Its patient-control interaction procedure does not apply
to the confirmed controls-only scope. The implementation follows the stated
formula rather than copying the document's R script.

## Outputs

- `run_manifest.json`: run time, input selection rule through recorded selected
  paths, UTC modification times, SHA-256 checksums, and all 28 configured panels.
  Missing inputs are reported explicitly; no synthetic cohorts are created.
- `cohort_summary.csv`: one row per available input, with correction/skipped
  status, input/eligible/excluded counts, distinct ages, age range, mean age,
  slope before correction, its two-sided Student-t 95% interval and p-value,
  slope after correction, and before/after mean, sample SD, and mean absolute
  BrainAGE. Statistics use eligible rows only.
  Group and status columns come first, followed by counts, age coverage, slope
  diagnostics, and paired before/after statistics. Control and ethnoracial
  summaries share this column order. Source timestamps, file paths, and checksums
  are retained only in the separate manifests, not in the summary tables.
- `<dataset>/<sex>_<panel>/participants_BrainAGE_S.csv`: original fields and
  identifiers, plus `eligible`, `exclusion_reason`, `correction_status`,
  `age_centered`, `residual_age_effect`, and `BrainAGE_S`. Excluded rows retain
  original values and have blank correction fields. Eligible rows in an
  unfit cohort have a skipped status and blank correction fields.
- `age_before_after.png`: paired scatterplots with linear fits, common axes,
  cohort size, observed age span, and original slope confidence interval.
- `distribution_before_after.png`: overlaid histograms using common bin edges.

The newest adjusted BrainAGE input is selected by filesystem modification time;
filename order breaks ties deterministically. Required columns are `SITE`,
`SubjectID`, `chronological_age`, and `Adjusted_BrainAGE`. Site and subject IDs
are read as strings to preserve leading zeros. A malformed source file fails
the run explicitly. No site adjustment is applied.

## Interpretation and validation

The correction preserves mean BrainAGE within each eligible cohort; mean
ethnoracial differences therefore remain. It changes individual values,
dispersion, and potentially mean absolute BrainAGE. Mean absolute BrainAGE here
is a descriptive measure of these sample-adjusted gaps, not independent
predictive accuracy. Training-reference files and existing bootstrap/permutation
analyses remain unchanged and do not consume the new values automatically.

The zero post-correction linear slope is guaranteed by the fitted transformation
and checks implementation, not external validity or removal of ethnoracial bias.
Residual nonlinear age patterns may remain. Inspect the scatterplots, especially
for small samples or narrow age spans; confidence intervals display slope
uncertainty without using it to gate correction. Plots retain the original
sample size and age coverage rather than suggesting equally reliable estimates.

Automated tests cover the known-slope formula, mean/identity preservation,
nonzero noise, age boundaries, missing/nonfinite data, insufficient participants,
constant ages, absent folders, and newest-file selection. Runtime checks enforce
mean preservation and a post-correction slope within absolute tolerance 1e-10.

## Training/reference control comparisons

`compare_controls.py` additionally corrects the four training/reference CSVs,
separately for sex and age panel. It starts with `predicted brain age_adjusted − age`
and applies exactly the same controls-only formula, using each reference panel's
own eligible mean age and slope. The ethnoracial correction coefficients are
not applied to the reference controls. Original reference files remain unchanged.

Run `correct.py` first. The comparison script uses the corrected cohorts in its
current manifest. Optional `--output-root` must match that correction output;
`--training-root` defaults to `training_data/`.

Corrected reference participant exports, before/after diagnostics, a summary,
and a checksum manifest are under `outputs/training_controls/`. Reference files
lack subject IDs and site labels: exported `source_row` is the one-based data-row
number, `SubjectID` is explicitly labeled `source_row_<number>`, and `SITE` is
`not_available`. These labels provide row provenance rather than real identities.

`outputs/control_comparisons/` contains one original-adjusted and one sample-corrected
age-scatter figure per ethnoracial group, with four sex/age panels and matched
training/reference controls. Missing group panels are marked explicitly.
The three BrainAGE_S ridgeline figures show all groups and controls together,
combined and separately by sex. They reuse the previous ridgeline layout,
colors, smoothing and −10 to +10 year display limits. As previously, the Training
ridge is a Gaussian approximation from its mean and SD, group ridges use KDE,
and ridge heights are normalized rather than comparable probability densities.
The plotted individual values are saved in `ridgeline_plot_data.csv`.

Both sides of these figures are now sample-specifically corrected. Mean differences
are still preserved, and the samples are not age-matched. Training/reference
controls are labeled explicitly: their independence from model training has not
been established, so these are descriptive comparisons, not held-out validation.
Existing bootstrap and permutation scripts continue using their original inputs.

## Residual plotting style

After correction of cohorts and controls, run:

```bash
python3 analyses/sample_age_correction/plot_residuals.py
```

This reuses the existing residual plot panel renderer, colors, segment-count
annotations, axis styling, and independent 5- and 10-year segment regressions.
The inputs are `BrainAGE_S` for both controls and ethnoracial participants.
Each group gets combined figures and individual available sex/age-panel figures
under `Residual plotting/<group>/sample_corrected/`, with segment-fit summaries
and a plot-input checksum manifest. Raw PAD residual plots remain unchanged.
Only available ethnoracial panels are drawn, matching the previous residual
plot layout. The Training labels refer to corrected training/reference controls.

Optional `--group-key self_chinese` selects one group; `--segment-years 5 10`
sets the diagnostic segment widths. `--input-root` selects a correction output
directory and `--output-root` selects the residual plot destination.
Segmented lines diagnose local or nonlinear age patterns; they do not apply
another correction. The global linear slope is effectively zero by construction,
while slopes inside individual age segments can remain nonzero.
The corrected plot legends label the training global slope explicitly and use
a dashed training global line to distinguish it from the local segmented fits.
An approximately flat global fit is expected for these sample-corrected inputs.

## Mean absolute BrainAGE_S

```bash
python3 analyses/sample_age_correction/summarize_mae.py
```

This writes `outputs/brainage_s_mae_summary.csv`, with one row per corrected
group/sex/age-panel combination, sample size, mean absolute original adjusted
BrainAGE, mean absolute BrainAGE_S, and the change (new minus original). Values
retain full precision. `outputs/brainage_s_mae_table.md` is the same corrected
MAE measure displayed to two decimal places, with separate sex/age columns.
Missing cohorts are blank rather than assigned an error of zero. The four
training/reference panels are included. MAE here is `mean(abs(BrainAGE_S))`;
these corrected in-sample gaps are descriptive, not independent prediction
validation metrics.
