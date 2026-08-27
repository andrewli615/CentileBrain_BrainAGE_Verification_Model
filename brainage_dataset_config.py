"""Active dataset labels and colors used by the BrainAGE scripts."""

BRAINAGE_DATASETS = [
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
