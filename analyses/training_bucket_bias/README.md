# Training regression adjustment of Chinese raw BrainAGE

This experiment fits raw training BrainAGE against chronological age separately
for each sex, CentileBrain age panel, and age bucket. It subtracts the **full
training regression prediction**, including the intercept, from Chinese raw
BrainAGE. It does not use existing adjusted BrainAGE or BrainAGE_S.

```text
Training raw BrainAGE = predicted brain age − chronological age
Training raw BrainAGE = a × chronological age + b + error
Chinese new BrainAGE = Chinese raw BrainAGE − (a × Chinese age + b)
```

The coefficients are estimated only from training/reference controls. No
Chinese slope or intercept is fitted for the correction. Chinese means and
slopes can change, and need not become zero. Corrected training controls are
exported for comparison; within each fitted bucket their means
and age slopes are zero by construction.

## Run

```bash
python3 analyses/training_bucket_bias/correct_chinese.py
python3 -m unittest discover -s analyses/training_bucket_bias -p 'test_*.py'
```

Requires the repository Python dependencies. Default width is 10 years;
`--bucket-years` and `--output-root` are optional overrides. Bucket origins
follow the previous residual plots, rather than decade-aligned calendar ages:
5–15, 15–25, 25–35, and the final shortened 35–40 bucket for the younger model;
40–50, 50–60, 60–70, 70–80, 80–90 for the older model.
Internal boundaries belong to the next bucket. Age 40 belongs only to the
younger model; age 90 is included in the final older bucket.

Finite age and raw BrainAGE are required. At least three valid training rows
and two distinct ages are required per bucket. Unfit buckets are skipped;
out-of-panel and missing values are excluded explicitly, retaining original
rows with blank correction fields. Application beyond the observed training
age span within a bucket is flagged, but still follows the requested formula.

## Outputs and interpretation

`outputs/` contains participant CSVs for four Chinese and four control panels,
a training coefficient table with 95% slope intervals, a cohort summary with
counts and before/after signed means and MAEs, bucket-level means and MAEs,
and raw-versus-adjusted residual plots with 10-year local regressions. A
manifest holds input paths and checksums; data tables contain no source times.
Plots use the previous residual renderer. The main adjusted figure overlays
**raw training points and their original bucket regressions** in blue with
**adjusted Chinese points and fits** in red. The training regressions show the
bias function that was subtracted, matching the earlier raw residual plots.
Legend labels explicitly distinguish raw training from adjusted Chinese.
This main figure compares different processing stages and must not be read as
an adjusted-versus-adjusted control comparison.
All training lines in the plots are fitted to raw training BrainAGE. Segment summary tables
record the training and Chinese processing stages.
All original inputs and BrainAGE_S
outputs remain unchanged.

## File guide

| File or folder in `outputs/` | Purpose |
| --- | --- |
| `Chinese_training_bucket_adjusted_10yr.png` | Main result: adjusted Chinese data with raw training fits. |
| `Chinese_raw_10yr.png` | Before-adjustment comparison, with both datasets raw. |
| `Chinese_final_MAE.xlsx` | Formatted final MAE report for review. |
| `Chinese_final_MAE.csv` | Compact MAE table for scripts or import; same results as the workbook. |
| `self_chinese/` | Participant-level corrected Chinese data for further analysis. |
| `training_controls/` | Reference-control correction data used by the plot audit. |
| `training_bucket_coefficients.csv` | Training slopes and intercepts defining the correction. |
| `cohort_summary.csv` | Counts, means, MAEs, and standard deviations before and after correction. |
| `bucket_summary.csv` | Counts and error summaries within each age bucket. |
| `raw_segment_fit_summary.csv` | Numerical fits behind the raw plot. |
| `training_bucket_adjusted_segment_fit_summary.csv` | Numerical fits behind the adjusted plot. |
| `run_manifest.json` | Source paths, checksums, settings, and correction formula. |

The duplicate `Chinese_adjusted_controls_diagnostic_10yr.png` and
`adjusted_controls_diagnostic_segment_fit_summary.csv` were removed. Once
training plots use raw values, they repeat the main adjusted comparison.
The generator now produces only the raw and main adjusted comparisons.

MAE is mean absolute adjusted gap, in years. Corrections may jump at bucket
boundaries because regressions are fitted independently. These plots are
descriptive calibration diagnostics; reference-control independence from
model training is unverified. A lower corrected MAE is not sufficient to
establish predictive improvement or removal of ethnoracial bias.
