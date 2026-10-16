"""Tabulate mean absolute BrainAGE_S by group, sex, and age panel."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from correct import DEFAULT_OUTPUT
from compare_controls import corrected_rows


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
        original_mae, corrected_mae = float(np.abs(before).mean()), float(np.abs(after).mean())
        rows.append(dict(Group=entry["Group"], Sex=entry["Sex"], Age_Panel=entry["Age_Panel"],
                         N=len(after), MAE_Adjusted_BrainAGE=original_mae, MAE_BrainAGE_S=corrected_mae,
                         MAE_Change=corrected_mae - original_mae))
    summary = pd.DataFrame(rows)
    output = input_root / "brainage_s_mae_summary.csv"
    summary.to_csv(output, index=False)
    # Readable pivot keeps distinct sex/age panels rather than pooling age distributions.
    order = list(dict.fromkeys(summary.Group))
    columns = [("Female", "5-40"), ("Female", "40-90"), ("Male", "5-40"), ("Male", "40-90")]
    pivot = summary.pivot(index="Group", columns=["Sex", "Age_Panel"], values="MAE_BrainAGE_S")
    pivot = pivot.reindex(index=order, columns=pd.MultiIndex.from_tuples(columns))
    text = ["# Mean absolute BrainAGE_S", "", "Values are in years. MAE = mean(abs(BrainAGE_S)); eligible corrected rows only.",
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
