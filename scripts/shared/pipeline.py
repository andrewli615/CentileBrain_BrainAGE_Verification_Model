"""Pinned-source correction, descriptive reporting, and figure pipeline."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.shared.paths import ROOT, RESULTS, PROVENANCE
from scripts.shared.data import catalog, training_frames, read_export, join_exports, METHODS, PANELS, KEYS, age_mask
from scripts.shared.datasets import BRAINAGE_DATASETS
from scripts.corrections.compare_methods import fit_references, correct_all
from scripts.statistics.summary_tables import summarize, statistics


def csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.17g")


def build():
    entries = catalog()
    training = training_frames(entries)
    fits = fit_references(training)
    coefficients = []
    for (sex, panel, width), table in fits.items():
        coefficients.append(table.assign(Sex=sex, Age_Panel=panel, Segment_Years=width))
    csv(pd.concat(coefficients, ignore_index=True), RESULTS/"coefficients/training_segments.csv")
    lookup = {(e["Dataset"],e["Sex"],e["Age_Panel"],e["Role"]):e for e in entries}
    cohorts, coverage, sample_coefficients = {}, [], []
    for key, label, _ in BRAINAGE_DATASETS:
        for sex, panel in PANELS:
            ids = dict(Dataset=key, Group=label, Sex=sex, Age_Panel=panel)
            raw = lookup[(key,sex,panel,"Raw")]
            adj = lookup[(key,sex,panel,"CentileBrain")]
            if not raw["Path"] and not adj["Path"]:
                coverage.append(dict(ids, Status="missing_panel", N_Input=0, N_Matched=0))
                continue
            r = read_export(ROOT/raw["Path"], True) if raw["Path"] else pd.DataFrame(columns=KEYS+["chronological_age","MR_predicted_age","Raw_BrainAGE"])
            a = read_export(ROOT/adj["Path"], False) if adj["Path"] else pd.DataFrame(columns=KEYS+["chronological_age","CentileBrain_Adjusted"])
            merged = join_exports(r, a)
            merged.insert(0, "Source_Row", np.arange(1,len(merged)+1))
            frame, sample = correct_all(merged, sex, panel, fits)
            frame = frame.assign(**ids)
            sample_coefficients.append(dict(ids, **sample))
            csv(frame, RESULTS / "participants" / key / f"{sex}_{panel}.csv")
            cohorts[(key,label,sex,panel)] = frame
            row = dict(ids, Status="available", N_Input=len(frame), N_Matched=int(frame.Matched_All_Methods.sum()),
                       N_Invalid_Age=int((~age_mask(frame.chronological_age,panel)).sum()),
                       N_Unmatched_Source=int(frame.Source_Match.ne("both").sum()))
            for col in METHODS:
                row[f"N_{col}"] = int(np.isfinite(frame[col]).sum())
            for width in (5,10):
                row[f"Outside_Training_Support_{width}yr"] = int(frame[f"Training_{width}yr_Outside_Training_Support"].fillna(False).sum())
            coverage.append(row)
    csv(pd.DataFrame(sample_coefficients), RESULTS/"coefficients/sample_age.csv")
    tables = summarize(cohorts)
    training_stats = []
    for (sex,panel), frame in training.items():
        corrected, sample = correct_all(frame, sex, panel, fits)
        csv(corrected, RESULTS/"participants/training"/f"{sex}_{panel}.csv")
        for col, method in METHODS.items():
            training_stats.append(dict(Group="Training/reference", Sex=sex, Age_Panel=panel, Method=method,
                                       **statistics(corrected,col)))
    tables["training_reference"] = pd.DataFrame(training_stats)
    tables["coverage"] = pd.DataFrame(coverage)
    for name, frame in tables.items():
        csv(frame, RESULTS/"summaries"/f"{name}.csv")
    settings = dict(Created_UTC=datetime.now(timezone.utc).isoformat(),
        Methods=METHODS, Population="Common eligible participants across all five methods per group/sex/age panel",
        Raw_CentileBrain_BAG="CentileBrain predicted brain age - chronological age; no adjustment",
        CentileBrain_Panel_Adjustment="Supplied Adjusted_BrainAGE, with CentileBrain's separate sex-by-age-panel linear adjustment",
        Training_Bucket_Adjustment="Raw CentileBrain BAG - (sex/panel training-bin slope * chronological_age + intercept)",
        Formula="Training-reference W-year bucket-adjusted BAG = raw CentileBrain BAG - fitted training raw-BAG trend within the W-year bin",
        BrainAGE_S="CentileBrain panel-adjusted BAG - own eligible cohort slope * (age - own eligible cohort mean age)",
        Age_Panels="5 <= age <= 40; 40 < age <= 90", Segment_Years=[5,10],
        Training_Minimum_N=3, Training_Minimum_Distinct_Ages=2,
        Extrapolation="Apply the fitted regression within its bin; flag ages outside training observed support",
        Unfit_Bin="Leave correction unavailable; retain participant and reason",
        Source_Catalog="results/provenance/source_catalog.json",
        Inference="Descriptive gaps; model training independence and upstream MRI inference are unverified")
    PROVENANCE.mkdir(parents=True,exist_ok=True)
    (PROVENANCE/"run_manifest.json").write_text(json.dumps(settings,indent=2)+"\n")
    payload = {name:frame.astype(object).where(pd.notna(frame),None).to_dict(orient="records") for name,frame in tables.items()}
    payload.update(methods=settings, sources=entries)
    (RESULTS/"summaries/workbook_data.json").write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
    print(f"Computed five methods for {len(cohorts)} panels; {sum(int(d.Matched_All_Methods.sum()) for d in cohorts.values())} matched participants")
    return cohorts, training
