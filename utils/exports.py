"""
Export helpers: ranked geographies and school list to CSV.
Uses internal column names (zip_code, population, density, school_count, etc.).
"""

import pandas as pd


def prepare_ranked_export(df: pd.DataFrame) -> pd.DataFrame:
    """Add rank column and sort by total_score for CSV download."""
    out = df.copy()
    if "total_score" in out.columns:
        out = out.sort_values("total_score", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def prepare_schools_export(df: pd.DataFrame) -> pd.DataFrame:
    """Column order for school list CSV (internal names)."""
    cols = [
        "school_name",
        "district_name",
        "address",
        "city",
        "state",
        "zip_code",
        "enrollment",
        "grades",
        "school_type",
        "charter_status",
        "operational_status",
        "ncessch",
        "phone",
        "website",
        "is_public_school",
        "athletics_budget_proxy",
        "athletics_budget_proxy_source",
        "booster_exists",
        "booster_match_confidence",
        "latest_booster_revenue",
        "latest_booster_net_assets",
        "latest_booster_tax_year",
        "matched_org_name",
        "matched_ein",
        "athletics_budget_proxy_zip",
        "booster_exists_zip",
        "booster_match_confidence_zip",
        "latest_booster_revenue_zip",
        "latest_booster_net_assets_zip",
        "latest_booster_tax_year_zip",
    ]
    existing = [c for c in cols if c in df.columns]
    return df[existing] if existing else df
