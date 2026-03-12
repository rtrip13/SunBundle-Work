"""
Data loading and enrichment for SunBundle Expansion Decision Tool.

Loads:
- data/Geographies/geographies.csv
- data/Schools/schools.csv
- data/Shapes/tl_2025_us_zcta520.shp  (or fallback GeoJSON)
- data/acs/income.csv, data/acs/poverty.csv (median income & poverty rate)
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Base paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GEOGRAPHIES_PATH = DATA_DIR / "Geographies" / "geographies.csv"
SCHOOLS_PATH = DATA_DIR / "Schools" / "schools.csv"
SHAPE_SHP_PATH = DATA_DIR / "Shapes" / "tl_2025_us_zcta520.shp"
ACS_INCOME_PATH = DATA_DIR / "acs" / "income.csv"
ACS_POVERTY_PATH = DATA_DIR / "acs" / "poverty.csv"

# Ann Arbor coordinates for distance calculation
ANN_ARBOR_LAT, ANN_ARBOR_LON = 42.2808, -83.7430


def _zip_to_string(series: pd.Series) -> pd.Series:
    """Convert to string and preserve leading zeros for 5-digit ZIPs."""
    s = series.astype(str).str.strip()
    mask = s.str.match(r"^\d+$") & (s.str.len() <= 5)
    return s.where(~mask, s.str.zfill(5))


def _extract_zip_from_name(name: str) -> str:
    """Extract 5-digit ZIP/ZCTA code from ACS NAME field (e.g. 'ZCTA5 48104')."""
    if not isinstance(name, str):
        return ""
    m = re.search(r"(\d{5})", name)
    return m.group(1) if m else ""


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in miles between two lat/lon points."""
    R = 3959.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return float(R * c)


def _load_acs_income() -> pd.DataFrame | None:
    """Load ACS median household income by ZCTA (B19013_001E)."""
    if not ACS_INCOME_PATH.exists():
        return None
    df = pd.read_csv(ACS_INCOME_PATH, dtype=str)
    if "GEO_ID" not in df.columns or "NAME" not in df.columns:
        return None
    # Drop label row where GEO_ID == 'Geography'
    df = df[df["GEO_ID"] != "Geography"].copy()
    df["zip_code"] = _zip_to_string(df["NAME"].apply(_extract_zip_from_name))

    income_col = None
    for cand in ["B19013_001E"]:
        if cand in df.columns:
            income_col = cand
            break
    if income_col is None:
        for c in df.columns:
            if "B19013" in c and c.endswith("E"):
                income_col = c
                break
    if income_col is None:
        return None

    df["median_household_income"] = pd.to_numeric(df[income_col], errors="coerce")
    return df[["zip_code", "median_household_income"]]


def _load_acs_poverty() -> pd.DataFrame | None:
    """Load ACS poverty percentage by ZCTA (S1701_C03_001E)."""
    if not ACS_POVERTY_PATH.exists():
        return None
    df = pd.read_csv(ACS_POVERTY_PATH, dtype=str)
    if "GEO_ID" not in df.columns or "NAME" not in df.columns:
        return None
    df = df[df["GEO_ID"] != "Geography"].copy()
    df["zip_code"] = _zip_to_string(df["NAME"].apply(_extract_zip_from_name))

    poverty_col = None
    for cand in ["S1701_C03_001E"]:
        if cand in df.columns:
            poverty_col = cand
            break
    if poverty_col is None:
        for c in df.columns:
            if "S1701_C03_001E" in c or (c.startswith("S1701_C03_") and c.endswith("E")):
                poverty_col = c
                break
    if poverty_col is None:
        return None

    df["poverty_rate"] = pd.to_numeric(df[poverty_col], errors="coerce")
    return df[["zip_code", "poverty_rate"]]


def _load_real_geographies() -> pd.DataFrame:
    """Load geographies CSV and map to internal schema; merge ACS income & poverty."""
    df = pd.read_csv(GEOGRAPHIES_PATH, dtype={"zip": str})
    df = df.rename(
        columns={
            "zip": "zip_code",
            "lat": "latitude",
            "lng": "longitude",
            "state_name": "state",
        }
    )
    if "state" not in df.columns and "state_id" in df.columns:
        df["state"] = df["state_id"]
    df["zip_code"] = _zip_to_string(df["zip_code"])

    for col in ("latitude", "longitude", "population", "density"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    income = _load_acs_income()
    if income is not None:
        df = df.merge(income, on="zip_code", how="left")

    poverty = _load_acs_poverty()
    if poverty is not None:
        df = df.merge(poverty, on="zip_code", how="left")

    return df


def _load_real_schools() -> pd.DataFrame:
    """Load schools CSV and map to internal schema. Use MZIP/LZIP for ZIP lookup."""
    df = pd.read_csv(SCHOOLS_PATH, dtype={"MZIP": str, "LZIP": str}, low_memory=False)
    zip_col = df["MZIP"].copy()
    if "LZIP" in df.columns:
        zip_col = zip_col.fillna(df["LZIP"])
    df["zip_code"] = _zip_to_string(zip_col.astype(str))

    df = df.rename(
        columns={
            "SCH_NAME": "school_name",
            "LEA_NAME": "district_name",
            "MCITY": "city",
            "MSTATE": "state",
        }
    )

    df["address"] = df["MSTREET1"].fillna("").astype(str)
    for col in ("MSTREET2", "MSTREET3"):
        if col in df.columns:
            df["address"] = (df["address"] + " " + df[col].fillna("").astype(str)).str.strip()

    if "GSLO" in df.columns and "GSHI" in df.columns:
        df["grades"] = df.apply(
            lambda r: f"{r.get('GSLO', '')}-{r.get('GSHI', '')}" if pd.notna(r.get("GSLO")) and pd.notna(r.get("GSHI")) else r.get("LEVEL", ""),
            axis=1,
        )
    else:
        df["grades"] = df.get("LEVEL", "")

    if "enrollment" not in df.columns:
        df["enrollment"] = ""
    if "latitude" not in df.columns:
        df["latitude"] = np.nan
    if "longitude" not in df.columns:
        df["longitude"] = np.nan
    return df


def _compute_school_count(geo: pd.DataFrame, schools: pd.DataFrame) -> pd.Series:
    counts = schools.groupby("zip_code").size()
    return geo["zip_code"].map(counts).fillna(0).astype(int)


def _compute_distance_to_ann_arbor(geo: pd.DataFrame) -> pd.Series:
    if "latitude" not in geo.columns or "longitude" not in geo.columns:
        return pd.Series(np.nan, index=geo.index)
    return geo.apply(
        lambda r: _haversine_miles(r["latitude"], r["longitude"], ANN_ARBOR_LAT, ANN_ARBOR_LON),
        axis=1,
    )


def _generate_dummy_geographies() -> pd.DataFrame:
    """Generate dummy data when file is missing."""
    np.random.seed(42)
    n = 25
    states = ["MI", "OH", "IN", "IL", "WI"]
    cities = ["Detroit", "Ann Arbor", "Lansing", "Cleveland", "Columbus", "Indianapolis", "Chicago", "Milwaukee"]
    zips = [f"{(45000 + i * 100):05d}" for i in range(n)]
    return pd.DataFrame(
        {
            "zip_code": zips,
            "city": np.random.choice(cities, n),
            "state": np.random.choice(states, n),
            "latitude": 41.5 + np.random.uniform(-2, 2, n),
            "longitude": -83.5 + np.random.uniform(-3, 3, n),
            "population": np.random.randint(5000, 80000, n),
            "density": np.random.uniform(50, 500, n).round(1),
            "distance_to_ann_arbor_miles": np.random.uniform(10, 250, n).round(1),
            "school_count": np.random.randint(2, 45, n),
            "median_household_income": np.random.uniform(30000, 90000, n).round(0),
            "poverty_rate": np.random.uniform(5, 35, n).round(1),
        }
    )


def _generate_dummy_schools(geographies: pd.DataFrame) -> pd.DataFrame:
    zip_codes = geographies["zip_code"].astype(str).unique().tolist()
    rows = []
    for i, z in enumerate(zip_codes[:20]):
        n_schools = min(4, 1 + (i % 4))
        base = geographies[geographies["zip_code"].astype(str) == z].iloc[0]
        for j in range(n_schools):
            rows.append(
                {
                    "school_name": f"Demo School {z}-{j+1}",
                    "district_name": f"{base['city']} District",
                    "address": f"{100 + j*50} Main St",
                    "city": base["city"],
                    "state": base["state"],
                    "zip_code": str(z),
                    "enrollment": int(np.random.randint(200, 800)),
                    "grades": "K-5" if j % 2 == 0 else "6-8",
                    "latitude": base["latitude"] + np.random.uniform(-0.02, 0.02),
                    "longitude": base["longitude"] + np.random.uniform(-0.02, 0.02),
                }
            )
    return pd.DataFrame(rows)


def load_schools(geographies: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load schools with internal column names."""
    if SCHOOLS_PATH.exists():
        df = _load_real_schools()
    else:
        geo = geographies if geographies is not None else _generate_dummy_geographies()
        df = _generate_dummy_schools(geo)
    df["zip_code"] = _zip_to_string(df["zip_code"])
    return df


def load_geographies(schools: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load geographies, enrich with ACS, and compute school_count and distance."""
    if GEOGRAPHIES_PATH.exists():
        df = _load_real_geographies()
    else:
        df = _generate_dummy_geographies()

    if schools is None:
        schools = load_schools(df)
    df["school_count"] = _compute_school_count(df, schools)
    df["distance_to_ann_arbor_miles"] = _compute_distance_to_ann_arbor(df)
    df["zip_code"] = _zip_to_string(df["zip_code"])
    return df


def load_zip_shapes(geographies: pd.DataFrame | None = None) -> dict:
    """
    Return simple square polygons around each ZIP centroid.

    This avoids requiring geopandas at runtime and guarantees we always have
    shapes for the Heatmap, even if the Census shapefile or geo stack is
    missing or heavy.
    """
    geo = geographies if geographies is not None else load_geographies()
    return _generate_dummy_geojson(geo)


def _generate_dummy_geojson(geographies: pd.DataFrame) -> dict:
    features = []
    for _, row in geographies.iterrows():
        z = str(row["zip_code"])
        lon = float(row.get("longitude", -83.5))
        lat = float(row.get("latitude", 41.5))
        delta = 0.05
        coords = [
            [
                [lon - delta, lat - delta],
                [lon + delta, lat - delta],
                [lon + delta, lat + delta],
                [lon - delta, lat + delta],
                [lon - delta, lat - delta],
            ]
        ]
        features.append(
            {
                "type": "Feature",
                "properties": {"zip_code": z},
                "geometry": {"type": "Polygon", "coordinates": coords},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def get_geography_columns() -> list[str]:
    """Expected internal columns for geographies."""
    return [
        "zip_code",
        "city",
        "state",
        "latitude",
        "longitude",
        "population",
        "density",
        "median_household_income",
        "poverty_rate",
        "distance_to_ann_arbor_miles",
        "school_count",
    ]


def get_school_columns() -> list[str]:
    """Expected internal columns for schools."""
    return [
        "school_name",
        "district_name",
        "address",
        "city",
        "state",
        "zip_code",
        "enrollment",
        "grades",
        "latitude",
        "longitude",
    ]

