"""
Hard filters for geographies before ranking.

Filters used in the app:
- max_median_household_income
- min_poverty_rate
- max_distance_miles
- min_school_count
- optional state_filter
- athletics_budget_proxy_zip (NCES SLFS proxy; not exact athletics spend)
- booster_exists_zip / latest_booster_revenue_zip (EO BMF–based signals)
"""

import numpy as np
import pandas as pd


def apply_filters(
    df: pd.DataFrame,
    max_median_household_income: float | None = None,
    min_poverty_rate: float | None = None,
    max_distance_miles: float | None = None,
    distance_column: str = "distance_to_ann_arbor_miles",
    min_school_count: float | None = None,
    state_filter: str | None = None,
    athletics_proxy_mode: str | None = None,
    min_athletics_budget_proxy_zip: float | None = None,
    booster_support_mode: str | None = None,
    min_booster_revenue_zip: float | None = None,
) -> pd.DataFrame:
    """Apply hard filters. None = do not filter on that criterion."""
    out = df.copy()

    if "median_household_income" in out.columns and max_median_household_income is not None:
        out = out[out["median_household_income"].fillna(np.inf) <= max_median_household_income]

    if "poverty_rate" in out.columns and min_poverty_rate is not None:
        out = out[out["poverty_rate"].fillna(-np.inf) >= min_poverty_rate]

    if distance_column in out.columns and max_distance_miles is not None:
        out = out[out[distance_column].fillna(np.inf) <= max_distance_miles]

    if "school_count" in out.columns and min_school_count is not None:
        out = out[out["school_count"].fillna(-np.inf) >= min_school_count]

    if "state" in out.columns and state_filter:
        out = out[out["state"].astype(str).str.strip().str.upper() == str(state_filter).strip().upper()]

    # Athletics budget proxy (NCES SLFS — not exact school-level athletics spend)
    if "athletics_budget_proxy_zip" in out.columns and athletics_proxy_mode and athletics_proxy_mode != "any":
        proxy = out["athletics_budget_proxy_zip"]
        if athletics_proxy_mode == "has_proxy":
            out = out[proxy.notna()]
        elif athletics_proxy_mode == "no_proxy":
            out = out[proxy.isna()]

    if "athletics_budget_proxy_zip" in out.columns and min_athletics_budget_proxy_zip is not None:
        if min_athletics_budget_proxy_zip > 0:
            proxy = out["athletics_budget_proxy_zip"].fillna(-np.inf)
            out = out[proxy >= min_athletics_budget_proxy_zip]

    # Booster / support org signals (EO BMF match — not verified as athletics-only)
    if "booster_exists_zip" in out.columns and booster_support_mode and booster_support_mode != "any":
        has_booster = out["booster_exists_zip"].fillna(False).astype(bool)
        if booster_support_mode == "has_booster":
            out = out[has_booster]
        elif booster_support_mode == "no_booster":
            out = out[~has_booster]

    if "latest_booster_revenue_zip" in out.columns and min_booster_revenue_zip is not None:
        if min_booster_revenue_zip > 0:
            rev = out["latest_booster_revenue_zip"]
            out = out[rev.fillna(-np.inf) >= min_booster_revenue_zip]

    return out.reset_index(drop=True)

