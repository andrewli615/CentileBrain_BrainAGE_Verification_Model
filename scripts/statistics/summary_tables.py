"""Whole-panel statistics; never average bucket MAEs."""

import numpy as np
import pandas as pd

from scripts.shared.data import METHODS


def statistics(frame, column):
    d = frame.loc[np.isfinite(frame[column]) & np.isfinite(frame.chronological_age)]
    if d.empty:
        return dict(N=0, Age_Mean=None, Age_SD=None, Age_Min=None, Age_Max=None,
                    BrainAGE_Mean=None, BrainAGE_SD=None, MAE=None)
    return dict(N=len(d), Age_Mean=d.chronological_age.mean(), Age_SD=d.chronological_age.std(ddof=1),
                Age_Min=d.chronological_age.min(), Age_Max=d.chronological_age.max(),
                BrainAGE_Mean=d[column].mean(), BrainAGE_SD=d[column].std(ddof=1), MAE=d[column].abs().mean())


def summarize(cohorts):
    matched_rows, available_rows, overview = [], [], []
    for (key, label, sex, panel), df in cohorts.items():
        ids = dict(Dataset=key, Group=label, Sex=sex, Age_Panel=panel)
        matched = df.loc[df.Matched_All_Methods]
        row = dict(ids, N=len(matched), Age_Mean=matched.chronological_age.mean(), Age_SD=matched.chronological_age.std(ddof=1))
        for column, name in METHODS.items():
            matched_rows.append(dict(ids, Method=name, **statistics(matched, column)))
            available_rows.append(dict(ids, Method=name, **statistics(df, column)))
            row[name] = statistics(matched, column)["MAE"]
        overview.append(row)
    return {"overview": pd.DataFrame(overview), "matched_statistics": pd.DataFrame(matched_rows),
            "available_statistics": pd.DataFrame(available_rows)}
