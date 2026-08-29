# Adjusted BrainAGE Q-Q Plots

Run from the repository root:

```bash
python3 analyses/qq_plots/generate_qq_plots.py
```

The script creates one signed adjusted BrainAGE Q-Q plot for every available
active dataset panel. It standardizes adjusted BrainAGE within each panel, uses
identical horizontal and vertical limits, and draws `y = x` in a square plotting
area. The reference line therefore appears at 45 degrees while deviations from
normality remain visible.

Plots and the summary CSV are written to `outputs/`.
