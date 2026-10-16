from pathlib import Path
import re
import sys
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.shared.datasets import BRAINAGE_DATASETS, age_panel_rows
from scripts.shared.data import model_exports

OUT = ROOT / "results/analyses/qq_plots"


def cohorts():
    colors={label:color for _,label,color in BRAINAGE_DATASETS}
    for label,sex,panel,path in model_exports('CentileBrain'):
        yield label,colors[label],sex,panel,path


def main():
    rows = []
    for group, color, sex, age_group, path in cohorts():
        values = age_panel_rows(pd.read_csv(path), age_group)["Adjusted_BrainAGE"].dropna().to_numpy()
        standardized = (values - values.mean()) / values.std(ddof=1)
        theoretical, observed = stats.probplot(standardized, dist="norm", fit=False)
        limit = 1.08 * max(abs(np.r_[theoretical, observed]))

        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        ax.scatter(theoretical, observed, s=22, color=color, alpha=.75, edgecolors="none")
        ax.plot([-limit, limit], [-limit, limit], color="#252525", linewidth=1.8, label="Normal reference")
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit),
               xlabel="Theoretical normal quantiles", ylabel="Standardized observed quantiles")
        ax.set_title(f"{group} {sex} {age_group} (n={len(values)})", fontsize=15, pad=10)
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)
        ax.grid(alpha=.2)
        ax.legend(frameon=False, loc="upper left")
        folder = OUT / group.replace(" ", "_")
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / f"{group.replace(' ', '_')}_{sex}_{age_group}_adjusted_qq.png"
        fig.savefig(output, dpi=250)
        plt.close(fig)
        rows.append({"Group": group, "Sex": sex, "Age_Group": age_group, "N": len(values),
                     "Mean_Adjusted_BrainAGE": values.mean(), "SD_Adjusted_BrainAGE": values.std(ddof=1),
                     "Plot_Path": str(output)})

    pd.DataFrame(rows).to_csv(OUT / "adjusted_qq_plot_summary.csv", index=False)
    print(f"Created {len(rows)} adjusted Q-Q plots in {OUT}")


if __name__ == "__main__":
    main()
