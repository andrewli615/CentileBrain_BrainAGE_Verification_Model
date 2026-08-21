"""Dataset labels/colors used by the adjusted-only BrainAGE scripts."""

BRAINAGE_DATASETS = [
    ("self_chilean", "Chilean", "#F58518"),
    ("self_chinese", "Chinese", "#2F8F46"),
    ("self_turkish", "Turkish", "#B279A2"),
]

STATS_DATASETS = [
    ("self_chinese", "Chinese", "#2F8F46"),
    ("self_turkish", "Turkish", "#B279A2"),
]

COLORS = {label: color for _, label, color in BRAINAGE_DATASETS}
COLORS["Training"] = "#585858"

BRAINAGE_ROW_ORDER = ["Training", *(label for _, label, _ in BRAINAGE_DATASETS)]
