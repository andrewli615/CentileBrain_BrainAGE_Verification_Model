from pathlib import Path
import re
import sys
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from brainage_dataset_config import BRAINAGE_DATASETS, age_panel_rows
OUT = Path(__file__).resolve().parent / "outputs" / "bootstrap"
N_REPS, SEED = 10_000, 2005


def reference_bag(sex, age_group):
    age_key = "age1234" if age_group == "5-40" else "age56789"
    df = pd.read_csv(ROOT / "training_data" / f"training_{age_key}_{sex.lower()}.csv")
    return (df["predicted brain age_adjusted"] - df["age"]).dropna().to_numpy()


def cohorts():
    for key, label, _ in BRAINAGE_DATASETS:
        root = ROOT / "ethnoracial_data_outputs" / key
        for folder in sorted(path for path in root.iterdir() if path.is_dir()):
            match = re.fullmatch(r"(Female|Male)_(5-40|40-90)", folder.name)
            if not match:
                continue
            files = list(folder.glob("*_Adjusted_BrainAGE_*.csv"))
            if files:
                yield label, *match.groups(), max(files, key=lambda path: path.stat().st_mtime)


def plot_histogram(label, metric, deltas, observed, ci, contains_zero):
    folder = OUT / "Plots" / label.split("_")[0]
    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.hist(deltas, bins=50, density=True, color="#A1D99B", edgecolor="white", alpha=.8,
            label=f"Bootstrap Distribution (B={N_REPS:,})")
    ax.axvline(0, color="#252525", linestyle=":", linewidth=2, label="Zero Difference (Delta = 0)")
    ax.axvline(ci[0], color="#006D2C", linestyle="--", label=f"95% CI: [{ci[0]:+.2f}, {ci[1]:+.2f}]")
    ax.axvline(ci[1], color="#006D2C", linestyle="--")
    ax.axvline(observed, color="#E6550D", linewidth=2.5, label=f"Observed: {observed:+.2f}")
    result = "No Sig. Difference (CI Contains 0)" if contains_zero else "Sig. Difference (CI Excludes 0)"
    ax.set(title=f"{label}\nTwo-Sample Bootstrap Test (Delta Mean {metric}) | {result}",
           xlabel=f"Delta Mean {metric} (Test - Reference, Years)",
           ylabel="Empirical Bootstrap Density")
    ax.legend(loc="upper right", fontsize=8, framealpha=.9)
    fig.tight_layout()
    output = folder / f"{label}_{metric}_bootstrap_histogram.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def main():
    rng = np.random.default_rng(SEED)
    summaries, samples = [], []
    OUT.mkdir(exist_ok=True)
    for group, sex, age_group, path in cohorts():
        signed_test = age_panel_rows(pd.read_csv(path), age_group)["Adjusted_BrainAGE"].dropna().to_numpy()
        signed_ref = reference_bag(sex, age_group)
        for metric, test, ref in (("BAG", signed_test, signed_ref),
                                  ("MAE", np.abs(signed_test), np.abs(signed_ref))):
            observed = test.mean() - ref.mean()
            deltas = np.empty(N_REPS)
            for i in range(N_REPS):
                deltas[i] = rng.choice(test, len(test), replace=True).mean() - rng.choice(ref, len(ref), replace=True).mean()
            ci = np.percentile(deltas, [2.5, 97.5])
            contains_zero = bool(ci[0] <= 0 <= ci[1])
            p_value = min(1.0, 2 * min(np.mean(deltas >= 0), np.mean(deltas <= 0)))
            label = f"{group}_{sex}_{age_group}"
            plot = plot_histogram(label, metric, deltas, observed, ci, contains_zero)
            summaries.append({"Group": group, "Sex": sex, "Age_Group": age_group, "Metric": metric,
                              "Bracket": "40bi" if age_group == "5-40" else "40a",
                              "N_Test": len(test), "N_Ref": len(ref), "Test_Mean": test.mean(),
                              "Ref_Mean": ref.mean(), "Observed_Delta": observed,
                              "Bootstrap_Mean_Delta": deltas.mean(), "CI_95_Lower": ci[0],
                              "CI_95_Upper": ci[1], "Contains_Zero": contains_zero,
                              "p_value": p_value, "Plot_Path": str(plot)})
            samples.append(pd.DataFrame({"Iteration": np.arange(1, N_REPS + 1), "Group": group,
                                         "Sex": sex, "Age_Group": age_group, "Metric": metric,
                                         "Bootstrap_Delta": deltas}))
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "two_sample_bootstrap_summary.csv", index=False)
    summary.to_excel(OUT / "two_sample_bootstrap_summary.xlsx", index=False)
    pd.concat(samples).to_csv(OUT / "two_sample_bootstrap_samples.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
