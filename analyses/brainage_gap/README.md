# BrainAGE GAP Calculations

Run the two analyses separately from the project root:

```bash
python3 analyses/brainage_gap/bootstrap.py
python3 analyses/brainage_gap/permutation.py
```

## Walkthrough

1. Both scripts process the active datasets from `brainage_dataset_config.py` using adjusted BrainAGE.
2. The bootstrap independently resamples each cohort and its full same-sex, same-age-panel training reference at their original sizes.
3. Each bootstrap iteration calculates the signed mean BAG difference and the percentile confidence interval.
4. The permutation test converts BAG to absolute error, pools cohort and reference MAE values, and shuffles group membership.
5. The permutation output reports the one-sided error-increase p-value and a two-sided p-value.
6. Neither analysis performs person-level age matching.
7. Both analyses use 10,000 iterations and seed `2005`.

## Outputs

- `outputs/bootstrap/`: bootstrap CSV/XLSX summary, all iteration values, and cohort plots.
- `outputs/permutation/`: permutation CSV/XLSX summary, all iteration values, and cohort plots.

The reference files do not contain site/scanner labels, so site adjustment is not possible. They must also be held out from model training and bias correction for an independent reference analysis.
