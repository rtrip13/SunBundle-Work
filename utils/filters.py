"""
Hard filters for geographies before ranking.

Filters used in the app:
- max_median_household_income
- min_poverty_rate
- max_distance_miles
- min_school_count
- optional state_filter
"""

import numpy as np
import pandas as pd


def apply_filters(
    df: pd.DataFrame,
    max_median_household_income: float | None = None,
    min_poverty_rate: float | None = None,
    max_distance_miles: float | None = None,
    min_school_count: float | None = None,
    state_filter: str | None = None,
) -> pd.DataFrame:
    """Apply hard filters. None = do not filter on that criterion."""
    out = df.copy()

    if "median_household_income" in out.columns and max_median_household_income is not None:
        out = out[out["median_household_income"].fillna(np.inf) <= max_median_household_income]

    if "poverty_rate" in out.columns and min_poverty_rate is not None:
        out = out[out["poverty_rate"].fillna(-np.inf) >= min_poverty_rate]

    if "distance_to_ann_arbor_miles" in out.columns and max_distance_miles is not None:
        out = out[out["distance_to_ann_arbor_miles"].fillna(np.inf) <= max_distance_miles]

    if "school_count" in out.columns and min_school_count is not None:
        out = out[out["school_count"].fillna(-np.inf) >= min_school_count]

    if "state" in out.columns and state_filter:
        out = out[out["state"].astype(str).str.strip().str.upper() == str(state_filter).strip().upper()]

    return out.reset_index(drop=True)

