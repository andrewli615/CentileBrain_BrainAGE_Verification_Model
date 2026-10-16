"""Compare each BAG method with the raw CentileBrain training reference."""

import numpy as np
import pandas as pd

from scripts.shared.data import METHODS, age_mask
from scripts.shared.datasets import BRAINAGE_DATASETS
from scripts.shared.paths import RESULTS
from scripts.plots import residual_style as style


def generate(cohorts, training):
    import matplotlib.pyplot as plt

    outputs, rows = [], []
    for column, method in METHODS.items():
        for key, label, _ in BRAINAGE_DATASETS:
            panels = []
            for sex, panel in style.PANELS:
                frame = cohorts.get((key, label, sex, panel))
                if frame is None:
                    continue
                observed = frame.loc[frame.Matched_All_Methods, ["chronological_age", column]].rename(columns={"chronological_age":"age", column:"residual"})
                if observed.empty:
                    continue
                ref = training[(sex, panel)]
                ref = ref.loc[age_mask(ref.chronological_age, panel) & np.isfinite(ref.Raw_BrainAGE)]
                ref = ref[["chronological_age", "Raw_BrainAGE"]].rename(columns={"chronological_age":"age", "Raw_BrainAGE":"residual"})
                panels.append(style.PlotData(key, f"{label} ({method})", sex, panel, ref, observed))
            if not panels:
                continue
            directory = RESULTS / "plots" / column / key
            directory.mkdir(parents=True, exist_ok=True)
            for width in (5, 10):
                nrows, ncols, size = style.combined_plot_layout(len(panels))
                fig, axes = plt.subplots(nrows, ncols, figsize=size, constrained_layout=True)
                axes = np.atleast_1d(axes).ravel()
                for i, (ax, data) in enumerate(zip(axes, panels)):
                    fits = style.draw_panel(ax, data, width, show_legend=i == 0)
                    for fit in fits:
                        fit.update(Dataset=key, Group=label, Method=method, Training_Measure="Raw", Population="Matched across five methods")
                    rows.extend(fits)
                for ax in axes[len(panels):]:
                    ax.set_visible(False)
                fig.suptitle(f"{label}: {method} | Raw CentileBrain training BAG | {width}-year fits", fontsize=15)
                target = directory / f"combined_{width}yr.png"
                fig.savefig(target, dpi=240)
                plt.close(fig)
                outputs.append(target)
    pd.DataFrame(rows).to_csv(RESULTS / "summaries/plot_fit_statistics.csv", index=False)
    return outputs
