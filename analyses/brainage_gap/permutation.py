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
from brainage_dataset_config import BRAINAGE_DATASETS
OUT = Path(__file__).resolve().parent / "outputs" / "permutation"
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


def plot_histogram(label, deltas, observed, null_mean, ci, p_value):
    folder = OUT / "Plots" / label.split("_")[0]
    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(deltas, bins=50, density=True, color="#9ECAE1", edgecolor="white", alpha=.8,
            label=f"Permutation Null (B={N_REPS:,})")
    ax.axvline(null_mean, color="#252525", linestyle=":", linewidth=1.8, label=f"Null Mean: {null_mean:+.3f}")
    ax.axvline(ci[0], color="#08519C", linestyle="--", label=f"95% Null Interval: [{ci[0]:+.2f}, {ci[1]:+.2f}]")
    ax.axvline(ci[1], color="#08519C", linestyle="--")
    ax.axvline(observed, color="#E6550D", linewidth=2.5, label=f"Observed: {observed:+.3f}")
    p_text = f"p = {p_value:.4f}" if p_value >= .0001 else "p < 0.0001"
    ax.set(title=f"{label}\nTwo-Sample Permutation Test (Delta MAE) | {p_text}",
           xlabel="Cohort MAE - reference MAE (years)", ylabel="Density")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output = folder / f"{label}_permutation_histogram.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def main():
    rng = np.random.default_rng(SEED)
    summaries, samples = [], []
    OUT.mkdir(exist_ok=True)
    for group, sex, age_group, path in cohorts():
        test = np.abs(pd.read_csv(path)["Adjusted_BrainAGE"].dropna().to_numpy())
        ref = np.abs(reference_bag(sex, age_group))
        observed = test.mean() - ref.mean()
        pooled = np.concatenate([test, ref])
        deltas = np.empty(N_REPS)
        for i in range(N_REPS):
            shuffled = rng.permutation(pooled)
            deltas[i] = shuffled[:len(test)].mean() - shuffled[len(test):].mean()
        ci = np.percentile(deltas, [2.5, 97.5])
        p_greater = (1 + np.sum(deltas >= observed)) / (N_REPS + 1)
        p_two_sided = (1 + np.sum(np.abs(deltas) >= abs(observed))) / (N_REPS + 1)
        label = f"{group}_{sex}_{age_group}"
        plot = plot_histogram(label, deltas, observed, deltas.mean(), ci, p_greater)
        summaries.append({"Group": group, "Sex": sex, "Age_Group": age_group,
                          "N_Test": len(test), "N_Ref": len(ref), "Test_MAE": test.mean(),
                          "Ref_MAE": ref.mean(), "Observed_Delta_MAE": observed,
                          "Null_Mean_Delta": deltas.mean(), "Null_95_Lower": ci[0],
                          "Null_95_Upper": ci[1], "p_value_error_increase": p_greater,
                          "p_value_two_sided": p_two_sided, "Plot_Path": str(plot)})
        samples.append(pd.DataFrame({"Iteration": np.arange(1, N_REPS + 1), "Group": group,
                                     "Sex": sex, "Age_Group": age_group, "Permutation_Delta": deltas}))
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "two_sample_permutation_summary.csv", index=False)
    summary.to_excel(OUT / "two_sample_permutation_summary.xlsx", index=False)
    pd.concat(samples).to_csv(OUT / "two_sample_permutation_samples.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
