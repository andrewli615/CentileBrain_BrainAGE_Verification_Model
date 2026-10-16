# CentileBrain BrainAGE

## Start Here

Open **`results/summaries/BrainAGE_method_comparison.xlsx`** for the current
comparison. Its first sheet shows whole-panel MAE for all 22 available
ethnoracial group/sex/age panels, using the same participants across five methods.
The remaining sheets show matched statistics, method-specific available
statistics, training-reference statistics, and methods/coverage.

```bash
python3 run_analysis.py                 # Corrections, workbook, plots, verification
python3 run_analysis.py --verify-only   # Verify saved results without rebuilding
python3 run_analysis.py --skip-plots    # Refresh tables/workbook, retain existing figures
python3 -m unittest discover -s scripts/validation -p 'test_*.py'
```

### Directory Map

| Location | Contents |
|---|---|
| `ethnoracial_data/` | Original raw inputs and existing templates; protected from the comparison pipeline. |
| `ethnoracial_data_outputs/` | Original CentileBrain exports; preserved in place. |
| `training_data/` | Original four training/reference prediction files; preserved in place. |
| `scripts/shared/` | Shared paths, dataset definitions, pinned source loading, pipeline. |
| `scripts/corrections/` | Training-bin and sample-specific correction methods. |
| `scripts/plots/` | Current comparison renderer and preserved plotting tools. |
| `scripts/statistics/` | Summary/workbook generation and explicit inferential analyses. |
| `scripts/preparation/` | Template preparation and one-time migration utilities. |
| `scripts/validation/` | Unit tests, saved-result verification, historical audit. |
| `results/summaries/` | Main workbook and full-precision CSV statistics. |
| `results/participants/` | All five participant values, eligibility, and training-support flags. |
| `results/coefficients/` | Training-bin and sample-specific coefficients. |
| `results/plots/<method>/<group>/` | Current matched-participant comparison figures. |
| `results/analyses/` | Relocated earlier analyses, iteration tables, workbooks, and figures. |
| `results/provenance/` | Source catalog, run settings, validation and migration reports. |
| `archive/pre_reorganization/` | Exact pre-migration code and outputs, including every historical workbook. |

`archive/pre_reorganization/migration_inventory.json` records each original
file's hash and archived/active location. Historical snapshots are not updated
by routine runs. The source data files are checked against their original hashes.
Finder `.DS_Store` changes are reported separately from scientific data.

### Input Selection and Reporting Rules

`results/provenance/source_catalog.json` pins every selected model export and
its SHA-256 hash. Reruns fail if a pinned source changes. Multiple candidate
exports require explicit catalog selection rather than choosing by modification
time. Review and deliberately edit the catalog to select a different export.

Exports join by string `SITE` and `SubjectID`; duplicates and conflicting ages
fail. The comparison pipeline never writes into the three source directories.
Missing sex/age panels are listed in `coverage.csv` and omitted from plots.

Age panels are **5 <= age <= 40** and **40 < age <= 90**. Ten-year bins are
5-15, 15-25, 25-35, 35-40, then 40-50 through 80-90. Five-year bins start at
5 and 40. Internal boundaries belong to the next bin, with final endpoints
included. At least three training rows and two distinct ages are required.

Training corrections start from raw BAG and subtract both the matching bin's
slope times age and intercept. The coefficients use only the matching training
sex and age panel and are reused across ethnoracial groups. Unsupported bins
remain unavailable. Within-bin extrapolation beyond observed training ages is
applied and explicitly flagged, following the historical Chinese method.

BrainAGE_S uses each cohort's own eligible CentileBrain-adjusted values and
preserves that cohort's mean. Corrections are computed on their defined
populations before selecting the intersection for matched comparisons.

MAE is `mean(abs(individual BrainAGE))` over the whole panel. Signed BrainAGE
mean/SD and chronological-age mean/SD/min/max appear in the detail sheets.
SD uses the sample definition (n-1). Missing results are blank, not zero.
CSV and workbook values retain calculation precision; workbook display uses
two decimals. Plots use the same matched participants as the Overview and
always show **raw training points and fits**, with each observed method labelled.
Training and observed segmented fits remain independent, with visible gaps.

These are descriptive corrected gaps. A lower MAE does not establish improved
prediction accuracy. The upstream fitted MRI-to-age model is external to this
repository; its inference cannot be reproduced here.

Current working scope: self-reported ethnoracial datasets. Genetic datasets are still present, but the default scripts are set to ignore them unless you explicitly ask for them.

Install Python packages if needed:

```bash
python3 -m pip install -r requirements.txt
```

## Populate BrainAGE Templates

Dry run one dataset:

```bash
python3 PopulateExcel.py --dataset self_chinese
```

Write the populated Excel templates:

```bash
python3 PopulateExcel.py --dataset self_chinese --write
```

Run all self datasets:

```bash
python3 PopulateExcel.py --all --write
```

`PopulateExcel.py` remains a compatibility entry point. Configuration lives in
`scripts/preparation/populate_excel.py`. The few values you are most likely to edit are:

- `DEFAULT_DATASET`
- `DEFAULT_CSV_SUBFOLDER`
- `DEFAULT_WRITE_FILES`
- `ALL_DATASET_PREFIXES`
- `SORT_INPUT_ROWS_BY_COLUMNS`

Reusable blank templates live here:

- `ethnoracial_data/brainAGE_template_Male_REUSABLE.xlsx`
- `ethnoracial_data/brainAGE_template_Female_REUSABLE.xlsx`

## Plot BrainAGE Distributions

The active BrainAGE groups are Black, Chilean, Chinese, Japanese, Mexican,
South Asian, and Turkish. White is retained in the workspace but excluded
from the adjusted and nonadjusted BrainAGE analysis scope. `self_mexican`
contains the August 2026 replacement Mexican results.

Create the panel-adjusted and raw CentileBrain BAG self-dataset ridgeline plots:

```bash
python3 plot.py
```

This writes:

- Six adjusted/raw ridgeline figures under `results/analyses/distribution_history/`.

The smoothing and axis controls are near the top of `scripts/plots/ridges.py`:

- `X_LIMITS`
- `RIDGE_HEIGHT`
- `TRAINING_BANDWIDTH`
- `DATASET_BANDWIDTH`

Training ridges are Gaussian approximations using the reference mean and SD;
their labels say "Gaussian fit". Ethnoracial ridges use kernel density estimates.

## BrainAGE GAP Analyses

### BrainAGE Measure Names

Use these names consistently in reports, figures, and analysis descriptions:

| Display name | Meaning |
|---|---|
| Raw CentileBrain BAG | CentileBrain predicted brain age minus chronological age, before any adjustment. |
| CentileBrain panel-adjusted BAG | The supplied CentileBrain-adjusted value, using its sex- and age-panel-specific linear adjustment (5–40 and >40–90). |
| Training-reference 10-year bucket-adjusted BAG | Raw CentileBrain BAG minus the sex/panel-specific raw training BAG trend fitted within 10-year age bins. |
| Training-reference 5-year bucket-adjusted BAG | Same training-reference adjustment, using 5-year bins. |
| BrainAGE_S (cohort age-trend adjusted BAG) | CentileBrain panel-adjusted BAG with the eligible cohort's own linear age trend removed while preserving its mean. |

MAE is the mean absolute BAG over the full age panel. The internal column and
source-file names retain their existing identifiers for compatibility; those
identifiers are not the preferred display names.
“Unadjusted” and “raw” mean the same pre-adjustment BAG here; use “Raw CentileBrain BAG”
as the single display name. The before/after MAEs are different values because the
after value follows a named adjustment.

Build the current method-comparison tables and plots with:

```bash
python3 run_analysis.py
```

The independent two-sample analyses remain explicit separate commands:

```bash
python3 -m scripts.statistics.bootstrap
python3 -m scripts.statistics.permutation
```

Their summaries, iteration tables, and cohort plots are kept separately under
`results/analyses/brainage_gap/bootstrap/` and
`results/analyses/brainage_gap/permutation/`. They retain the original
CentileBrain panel-adjusted measure and are not rerun by `run_analysis.py`.

Adjusted-data Q-Q plots are generated separately:

```bash
python3 -m scripts.plots.qq_plots
```

## Notes

Audit training inputs and fitted curves across the plot families:

```bash
python3 -m scripts.validation.historical_audit
```

The historical report is saved under `results/analyses/plot_audit/`.
The new five-method audit is `results/provenance/validation_report.json`.

Sample-specific residual age correction is available separately:

```bash
python3 -m scripts.corrections.sample_age
```

This produces `BrainAGE_S` and before/after diagnostics for each active group,
sex, and age panel. See `archive/pre_reorganization/analyses/sample_age_correction/README.md` for the
method and interpretation. Existing BrainAGE GAP analyses retain their original inputs.

- `plot.py` is a small wrapper so you can keep using the simple command.
- `scripts/plots/ridges.py` contains the distribution plotting logic.
- Generated caches and macOS metadata files are ignored by git.

### Workbook Runtime

The report builder uses the bundled Node.js spreadsheet runtime at
`~/.cache/codex-runtimes/codex-primary-runtime/dependencies/`. If installed
elsewhere, set `BRAINAGE_NODE` to its Node executable and
`BRAINAGE_NODE_MODULES` to the directory containing `@oai/artifact-tool`.
Python requirements remain in `requirements.txt`. Participant corrections and
CSV statistics are saved before workbook generation, so a missing spreadsheet
runtime does not discard those results.
