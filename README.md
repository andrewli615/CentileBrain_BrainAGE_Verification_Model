# CentileBrain BrainAGE

This folder has three main working areas:

- `ethnoracial_data/`: input CSVs and BrainAGE Excel templates.
- `ethnoracial_data_outputs/`: CentileBrain output CSVs.
- `graph_outputs/`: generated summaries and BrainAGE plots.

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

The few values you are most likely to edit are near the top of `PopulateExcel.py`:

- `DEFAULT_DATASET`
- `DEFAULT_CSV_SUBFOLDER`
- `DEFAULT_WRITE_FILES`
- `ALL_DATASET_PREFIXES`
- `SORT_INPUT_ROWS_BY_COLUMNS`

Reusable blank templates live here:

- `ethnoracial_data/brainAGE_template_Male_REUSABLE.xlsx`
- `ethnoracial_data/brainAGE_template_Female_REUSABLE.xlsx`

## Plot BrainAGE Distributions

Create the adjusted and unadjusted self-dataset ridgeline plots:

```bash
python3 plot.py
```

This writes:

- `graph_outputs/self_adjusted_brainage_ridgelines.png`
- `graph_outputs/self_unadjusted_brainage_ridgelines.png`

The smoothing and axis controls are near the top of `plot_brainage_ridges.py`:

- `TRAINING_DENSITY_STYLE`
- `TRAINING_KDE_BANDWIDTH`
- `SELF_KDE_BANDWIDTH`
- `RIDGE_HEIGHT`
- `X_AXIS_TICK_STEP`
- `SAVE_PLOT_DATA`

## Notes

- `plot.py` is a small wrapper so you can keep using the simple command.
- `plot_brainage_ridges.py` contains the actual plotting logic.
- Generated caches and macOS metadata files are ignored by git.
