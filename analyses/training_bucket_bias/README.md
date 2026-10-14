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
exported and plotted for comparison; within each fitted bucket their means
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
an adjusted-versus-adjusted control comparison. A separate
`Chinese_adjusted_controls_diagnostic_10yr.png` shows both sides adjusted; its
training bucket fits are flat at zero by construction. Segment summary tables
record the training and Chinese processing stages.
All original inputs and BrainAGE_S
outputs remain unchanged.

MAE is mean absolute adjusted gap, in years. Corrections may jump at bucket
boundaries because regressions are fitted independently. These plots are
descriptive calibration diagnostics; reference-control independence from
model training is unverified. A lower corrected MAE is not sufficient to
establish predictive improvement or removal of ethnoracial bias.
