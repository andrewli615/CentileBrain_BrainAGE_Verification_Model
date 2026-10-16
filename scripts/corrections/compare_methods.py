"""Five explicitly distinct BrainAGE measures, preserving method populations."""

import numpy as np
import pandas as pd

from scripts.corrections.training_buckets import fit_training, apply_training
from scripts.corrections.sample_age import correct_frame
from scripts.shared.data import age_mask, METHODS


def fit_references(training):
    fits = {}
    for pair, frame in training.items():
        for width in (10, 5):
            fits[(*pair, width)] = fit_training(frame, pair[1], width)
    return fits


def correct_all(frame, sex, panel, fits):
    result = frame.copy()
    for column in ("chronological_age","Raw_BrainAGE","CentileBrain_Adjusted"):
        result[column] = pd.to_numeric(result[column],errors="coerce")
    valid_age = age_mask(result.chronological_age, panel)
    for column in ("Raw_BrainAGE", "CentileBrain_Adjusted"):
        good = valid_age & np.isfinite(result[column])
        result[column+"_Status"] = np.where(good, "available", np.where(~valid_age, "outside_or_invalid_age", "missing_or_nonfinite_value"))
        result.loc[~good, column] = np.nan
    for width in (10, 5):
        corrected = apply_training(result, fits[(sex, panel, width)], panel, width)
        column = f"Training_{width}yr"
        result[column] = corrected.BrainAGE_TrainingBucketAdjusted
        result[column+"_Status"] = corrected.Correction_Status
        result[column+"_Reason"] = corrected.Exclusion_Reason
        result[column+"_Outside_Training_Support"] = corrected.Outside_Training_Age_Support
        result[column+"_Bin"] = corrected.Bucket_Index
    corrected, summary = correct_frame(result.rename(columns={"CentileBrain_Adjusted":"Adjusted_BrainAGE"}), panel)
    result["BrainAGE_S"] = corrected.BrainAGE_S
    result["BrainAGE_S_Status"] = corrected.correction_status
    result["BrainAGE_S_Reason"] = corrected.exclusion_reason
    result["Matched_All_Methods"] = np.isfinite(result[list(METHODS)]).all(axis=1)
    return result, summary
