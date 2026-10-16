"""Pinned model exports and validated participant joins."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.shared.datasets import BRAINAGE_DATASETS
from scripts.shared.paths import ROOT, PROVENANCE

PANELS = [(sex, age) for sex in ("Female", "Male") for age in ("5-40", "40-90")]
KEYS = ["SITE", "SubjectID"]
METHODS = {
    "Raw_BrainAGE": "Raw CentileBrain BAG",
    "CentileBrain_Adjusted": "CentileBrain panel-adjusted BAG",
    "Training_10yr": "Training-reference 10-year bucket-adjusted BAG",
    "Training_5yr": "Training-reference 5-year bucket-adjusted BAG",
    "BrainAGE_S": "BrainAGE_S (cohort age-trend adjusted BAG)",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def age_mask(age, panel):
    return np.isfinite(age) & (age.between(5, 40) if panel == "5-40" else ((age > 40) & (age <= 90)))


def catalog():
    path = PROVENANCE / "source_catalog.json"
    if path.exists():
        entries = json.loads(path.read_text())
    else:
        entries = []
        for sex, panel in PANELS:
            name = "age1234" if panel == "5-40" else "age56789"
            source = ROOT / "training_data" / f"training_{name}_{sex.lower()}.csv"
            entries.append(dict(Role="Training", Dataset="training", Sex=sex, Age_Panel=panel,
                                Path=str(source.relative_to(ROOT)), SHA256=digest(source)))
        for key, label, _ in BRAINAGE_DATASETS:
            for sex, panel in PANELS:
                folder = ROOT / "ethnoracial_data_outputs" / key / f"{sex}_{panel}"
                for role, pattern in (("Raw", "*_MR_predicted_age_*.csv"), ("CentileBrain", "*_Adjusted_BrainAGE_*.csv")):
                    candidates = [p for p in folder.glob(pattern) if role != "Raw" or "_Adjusted_" not in p.name]
                    if len(candidates) > 1:
                        raise ValueError(f"Select a source explicitly in source_catalog.json: {folder} ({role})")
                    source = candidates[0] if candidates else None
                    entries.append(dict(Role=role, Dataset=key, Group=label, Sex=sex, Age_Panel=panel,
                                        Path=str(source.relative_to(ROOT)) if source else None,
                                        SHA256=digest(source) if source else None))
        PROVENANCE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries, indent=2)+"\n")
    for entry in entries:
        if entry["Path"] and digest(ROOT / entry["Path"]) != entry["SHA256"]:
            raise ValueError(f"Pinned source changed: {entry['Path']}; review catalog explicitly")
    return entries


def read_export(path, raw):
    df = pd.read_csv(path, dtype={key: str for key in KEYS}, keep_default_na=False)
    required = set(KEYS + ["chronological_age", "MR_predicted_age" if raw else "Adjusted_BrainAGE"])
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns in {path}: {required-set(df.columns)}")
    if df[KEYS].eq("").any().any() or df.duplicated(KEYS).any():
        raise ValueError(f"Missing or duplicate participant keys in {path}")
    for column in df.columns.difference(KEYS):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if raw:
        calculated = df.MR_predicted_age-df.chronological_age
        if "Unadjusted_BrainAGE" in df and not np.allclose(df.Unadjusted_BrainAGE, calculated, equal_nan=True):
            raise ValueError(f"Raw BrainAGE disagrees with predicted age minus age: {path}")
        df["Raw_BrainAGE"] = calculated
        return df[KEYS+["chronological_age", "MR_predicted_age", "Raw_BrainAGE"]]
    return df[KEYS+["chronological_age", "Adjusted_BrainAGE"]].rename(columns={"Adjusted_BrainAGE":"CentileBrain_Adjusted"})


def join_exports(raw, adjusted):
    """Outer join retains unmatched participants for coverage and exclusions."""
    if raw.duplicated(KEYS).any() or adjusted.duplicated(KEYS).any():
        raise ValueError("Duplicate participant keys")
    merged = raw.merge(adjusted, on=KEYS, how="outer", validate="one_to_one", suffixes=("_raw", "_adjusted"), indicator="Source_Match")
    a = pd.to_numeric(merged.chronological_age_raw,errors="coerce")
    b = pd.to_numeric(merged.chronological_age_adjusted,errors="coerce")
    both = a.notna() & b.notna()
    if not np.isclose(a[both], b[both], atol=1e-10, rtol=0).all():
        raise ValueError("Chronological age mismatch between model exports")
    merged["chronological_age"] = a.combine_first(b)
    return merged.drop(columns=["chronological_age_raw", "chronological_age_adjusted"])


def training_frames(entries):
    result = {}
    for e in entries:
        if e["Role"] != "Training":
            continue
        df = pd.read_csv(ROOT/e["Path"])
        df["chronological_age"] = pd.to_numeric(df.age, errors="coerce")
        df["Raw_BrainAGE"] = pd.to_numeric(df["predicted brain age"], errors="coerce")-df.chronological_age
        df["CentileBrain_Adjusted"] = pd.to_numeric(df["predicted brain age_adjusted"], errors="coerce")-df.chronological_age
        df["source_row"] = np.arange(1, len(df)+1)
        df["SITE"] = "not_available"
        df["SubjectID"] = "source_row_"+df.source_row.astype(str)
        result[(e["Sex"], e["Age_Panel"])] = df
    return result


def model_exports(role):
    """Pinned export iterator for explicit supporting-analysis commands."""
    for entry in catalog():
        if entry["Role"]==role and entry["Path"]:
            yield entry["Group"],entry["Sex"],entry["Age_Panel"],ROOT/entry["Path"]
