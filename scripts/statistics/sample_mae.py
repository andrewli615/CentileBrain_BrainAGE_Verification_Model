"""Tabulate MAE before and after the cohort age-trend BAG adjustment."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.corrections.sample_age import DEFAULT_OUTPUT
from scripts.plots.compare_controls import corrected_rows


def run(input_root=DEFAULT_OUTPUT):
    cohort_manifest = json.loads((input_root / "run_manifest.json").read_text())
    control_manifest = json.loads((input_root / "training_controls" / "run_manifest.json").read_text())
    entries = [dict(c, Group="Training/reference controls") for c in control_manifest["Controls"]]
    entries += [c for c in cohort_manifest["Cohorts"] if c["Status"] == "corrected"]
    rows = []
    for entry in entries:
        frame = corrected_rows(Path(entry["Participant_File"]))
        before = pd.to_numeric(frame.Adjusted_BrainAGE).to_numpy(float)
        after = pd.to_numeric(frame.BrainAGE_S).to_numpy(float)
        if not len(after) or not np.isfinite(after).all() or not np.isfinite(before).all():
            raise ValueError(f"No finite corrected values: {entry['Participant_File']}")
        panel_adjusted_mae = float(np.abs(before).mean())
        brainage_s_mae = float(np.abs(after).mean())
        rows.append(dict(Group=entry["Group"], Sex=entry["Sex"], Age_Panel=entry["Age_Panel"],
                         N=len(after), MAE_CentileBrain_Panel_Adjusted_BAG=panel_adjusted_mae,
                         MAE_BrainAGE_S_Cohort_Age_Trend_Adjusted=brainage_s_mae,
                         MAE_Change=brainage_s_mae - panel_adjusted_mae))
    summary = pd.DataFrame(rows)
    output = input_root / "brainage_s_mae_summary.csv"
    summary.to_csv(output, index=False)
    # Readable pivot keeps distinct sex/age panels rather than pooling age distributions.
    order = list(dict.fromkeys(summary.Group))
    columns = [("Female", "5-40"), ("Female", "40-90"), ("Male", "5-40"), ("Male", "40-90")]
    pivot = summary.pivot(index="Group", columns=["Sex", "Age_Panel"], values="MAE_BrainAGE_S_Cohort_Age_Trend_Adjusted")
    pivot = pivot.reindex(index=order, columns=pd.MultiIndex.from_tuples(columns))
    text = ["# BrainAGE_S (cohort age-trend adjusted BAG) MAE", "", "Values are in years. MAE = mean absolute BAG; eligible rows only.",
            "These are sample-adjusted descriptive gaps, not independent prediction-validation errors.", "",
            "| Group | Female 5–40 | Female >40–90 | Male 5–40 | Male >40–90 |",
            "|---|---:|---:|---:|---:|"]
    for group, row in pivot.iterrows():
        values = ["—" if pd.isna(value) else f"{value:.2f}" for value in row]
        text.append("| " + " | ".join([group, *values]) + " |")
    (input_root / "brainage_s_mae_table.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))
    print(f"\nFull-precision summary: {output.resolve()}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args().input_root)
