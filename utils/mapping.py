"""
Map visualization: choropleth by total_score and school markers.
Tooltip shows total, need, feasibility scores and key drivers (poverty, income, schools, distance).
"""

from __future__ import annotations

import copy
from typing import Optional

import pandas as pd

try:
    import folium
    from folium.features import GeoJsonTooltip
except ImportError:  # pragma: no cover - handled at runtime
    folium = None

# Center on US for nationwide ACS data
DEFAULT_CENTER = [39.0, -98.0]
DEFAULT_ZOOM = 4


def _zip_str(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _format_value(col: str, val) -> str:
    """Format values for choropleth tooltip."""
    if pd.isna(val) or val == "":
        return "—"
    if col == "median_household_income":
        return f"${float(val):,.0f}"
    if col == "poverty_rate":
        return f"{float(val):.1f}%"
    if col in ("total_score", "need_score", "feasibility_score", "_score"):
        return f"{float(val):.2f}"
    if col == "distance_to_ann_arbor_miles":
        return f"{float(val):.0f} mi"
    if col == "school_count":
        return f"{int(val)}"
    if col == "population":
        return f"{int(val):,}"
    if col == "density":
        return f"{float(val):.1f}"
    return str(val)


def build_choropleth(
    geojson: dict,
    scored_df: pd.DataFrame,
    score_column: str = "total_score",
    zip_column: str = "zip_code",
    tooltip_columns: Optional[list] = None,
):
    """
    Build folium choropleth.
    Returns None only if folium is not available or there is no scored data.
    """
    if folium is None:
        # Try a late import in case folium was installed after module import
        try:
            import folium as _f  # type: ignore
            from folium.features import GeoJsonTooltip as _G  # type: ignore

            globals()["folium"] = _f
            globals()["GeoJsonTooltip"] = _G
        except Exception:
            # Folium really isn't available; skip map instead of crashing
            return None

    if scored_df is None or len(scored_df) == 0:
        return None

    geojson = copy.deepcopy(geojson)
    scored_df = scored_df.copy()
    scored_df[zip_column] = _zip_str(scored_df[zip_column])
    row_lookup = scored_df.set_index(zip_column)

    # Default tooltip contents if caller doesn't override
    if tooltip_columns is None:
        tooltip_columns = [
            "city",
            "state",
            "need_score",
            "feasibility_score",
            "poverty_rate",
            "median_household_income",
            "school_count",
            "distance_to_ann_arbor_miles",
        ]
    display_cols = [c for c in tooltip_columns if c in scored_df.columns]

    for feat in geojson.get("features", []):
        props = feat.setdefault("properties", {})
        z = props.get("zip_code") or props.get("ZCTA5CE20") or props.get("GEOID")
        z = str(z).strip() if z is not None else None
        if z and z in row_lookup.index:
            row = row_lookup.loc[z]
            total = row.get(score_column, 0.0)
            props["zip_code"] = z
            props["_score"] = float(total)
            props["_score_fmt"] = _format_value("total_score", total)
            for col in display_cols:
                val = row.get(col)
                props[f"{col}_fmt"] = _format_value(col, val)
        else:
            props["zip_code"] = z or "—"
            props["_score"] = None
            props["_score_fmt"] = "—"
            for col in display_cols:
                props[f"{col}_fmt"] = "—"

    tooltip_fields = [
        "zip_code",
        "_score_fmt",
        "need_score_fmt",
        "feasibility_score_fmt",
        "city_fmt",
        "state_fmt",
        "poverty_rate_fmt",
        "median_household_income_fmt",
        "school_count_fmt",
        "distance_to_ann_arbor_miles_fmt",
    ]
    aliases = [
        "ZIP",
        "Total score",
        "Need score",
        "Feasibility score",
        "City",
        "State",
        "Poverty %",
        "Median HH income",
        "Schools",
        "Dist (mi)",
    ]

    m = folium.Map(location=DEFAULT_CENTER, zoom_start=DEFAULT_ZOOM, tiles="CartoDB positron")
    folium.Choropleth(
        geo_data=geojson,
        name="Score",
        data=scored_df,
        columns=[zip_column, score_column],
        key_on="feature.properties.zip_code",
        fill_color="YlOrRd",
        fill_opacity=0.6,
        line_opacity=0.3,
        legend_name="Expansion score (0–1)",
        nan_fill_color="white",
    ).add_to(m)

    tip = GeoJsonTooltip(fields=tooltip_fields, aliases=aliases, localize=True)
    folium.GeoJson(
        geojson,
        style_function=lambda x: {"fillColor": "transparent", "color": "gray", "weight": 0.5},
        tooltip=tip,
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def add_school_markers(map_obj, schools_df: pd.DataFrame):
    """Add circle markers for each school. Skips rows without lat/lng."""
    if folium is None or map_obj is None:
        return map_obj
    for _, row in schools_df.iterrows():
        lat, lon = row.get("latitude"), row.get("longitude")
        if pd.isna(lat) or pd.isna(lon):
            continue
        name = row.get("school_name", "School")
        enrollment = row.get("enrollment", "")
        folium.CircleMarker(
            location=[float(lat), float(lon)],
            radius=5,
            popup=f"{name}<br>Enrollment: {enrollment}",
            color="blue",
            fill=True,
            fillColor="blue",
        ).add_to(map_obj)
    return map_obj


def build_school_map_only(schools_df: pd.DataFrame, center: Optional[list] = None):
    """Build map with school markers. Returns None if no schools have lat/lng."""
    if folium is None:
        raise ImportError("folium is required. pip install folium")
    if schools_df is None or len(schools_df) == 0:
        return None
    has_coords = "latitude" in schools_df.columns and "longitude" in schools_df.columns
    if has_coords and schools_df["latitude"].notna().any() and schools_df["longitude"].notna().any():
        if center is None:
            center = [schools_df["latitude"].mean(), schools_df["longitude"].mean()]
        m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
        add_school_markers(m, schools_df)
        return m
    return None

