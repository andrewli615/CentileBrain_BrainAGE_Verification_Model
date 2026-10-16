"""Active dataset labels and colors used by the BrainAGE scripts."""

import pandas as pd

BRAINAGE_DATASETS = [
    ("self_black", "Black", "#1F77B4"),
    ("self_chilean", "Chilean", "#F28E2B"),
    ("self_chinese", "Chinese", "#2F8F46"),
    ("self_japanese", "Japanese", "#4C78A8"),
    ("self_mexican", "Mexican", "#E45756"),
    ("self_southasian", "South Asian", "#72B7B2"),
    ("self_turkish", "Turkish", "#B279A2"),
]

STATS_DATASETS = [
    ("self_turkish", "Turkish", "#B279A2"),
]

COLORS = {label: color for _, label, color in BRAINAGE_DATASETS}
COLORS["Training"] = "#585858"

BRAINAGE_ROW_ORDER = ["Training", *(label for _, label, _ in BRAINAGE_DATASETS)]


def age_panel_rows(df, age_group):
    age = pd.to_numeric(df["chronological_age"], errors="coerce")
    return df[age.between(5, 40) if age_group == "5-40" else ((age > 40) & (age <= 90))]
