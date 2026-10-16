"""Project paths shared by analysis entry points."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
PROVENANCE = RESULTS / "provenance"
ARCHIVE = ROOT / "archive/pre_reorganization"
SOURCE_DIRS = ("ethnoracial_data", "ethnoracial_data_outputs", "training_data")
