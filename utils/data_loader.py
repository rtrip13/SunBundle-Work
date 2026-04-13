"""
Data loading and enrichment for SunBundle Expansion Decision Tool.

Loads:
- data/Geographies/geographies.csv
- data/Schools/schools.csv
- data/Shapes/tl_*_us_zcta520.shp (optional ZCTA polygons for the Heatmap; cached as simplified GeoJSON)
- data/acs/income.csv, data/acs/poverty.csv (median income & poverty rate)
- data/nces/*.csv — SLFS athletics proxy (defaults to ``slfs_fy2022_school_proxy.csv``; auto-picks a matching file)
- data/irs/*.csv — IRS EO / BMF-style nonprofits (defaults to ``eo2.csv``; column names mapped flexibly)
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Base paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GEOGRAPHIES_PATH = DATA_DIR / "Geographies" / "geographies.csv"
SCHOOLS_PATH = DATA_DIR / "Schools" / "schools.csv"
# Preferred TIGER/LINE ZCTA filename; any ``tl_*_us_zcta*.shp`` in ``data/Shapes/`` is also picked up.
SHAPE_SHP_PATH = DATA_DIR / "Shapes" / "tl_2025_us_zcta520.shp"
# Optional: drop a GeoJSON here (e.g. exported from GIS) to skip the shapefile path.
SHAPE_GEOJSON_PATH = DATA_DIR / "Shapes" / "zcta_boundaries.geojson"
# Written automatically when using the TIGER shapefile; speeds up later runs.
ZCTA_FOLIUM_CACHE_PATH = DATA_DIR / "Shapes" / "zcta_folium_cache.geojson"
ZCTA_FOLIUM_CACHE_META_PATH = DATA_DIR / "Shapes" / "zcta_folium_cache.meta.json"
# Douglas–Peucker tolerance in degrees (~0.012° ≈ 1.3 km at mid‑latitudes).
ZCTA_FOLIUM_SIMPLIFY_DEGREES = 0.012
ACS_INCOME_PATH = DATA_DIR / "acs" / "income.csv"
ACS_POVERTY_PATH = DATA_DIR / "acs" / "poverty.csv"
SLFS_PROXY_PATH = DATA_DIR / "nces" / "slfs_fy2022_school_proxy.csv"
BMF_PATH = DATA_DIR / "irs" / "eo2.csv"

# Booster matching: name-only matches were almost never ≥ 0.55 vs IRS org names.
_BOOSTER_CONF_NAME_MATCH = 0.40
_BOOSTER_CONF_ZIP_HEURISTIC = 0.38
_BOOSTER_CONF_ZIP_BROAD = 0.30
# Use a location-level fast path for large school files to avoid multi-minute cold starts.
_BOOSTER_FAST_MATCH_MIN_ROWS = 20000

# Ann Arbor coordinates for distance calculation
ANN_ARBOR_LAT, ANN_ARBOR_LON = 42.2808, -83.7430


def _discover_slfs_proxy_path() -> Path | None:
    """Prefer ``SLFS_PROXY_PATH``; else ``data/nces/*.csv`` with SLFS columns."""
    if SLFS_PROXY_PATH.exists():
        return SLFS_PROXY_PATH
    d = DATA_DIR / "nces"
    if not d.is_dir():
        return None
    cands = sorted(d.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for c in cands:
        try:
            hdr = pd.read_csv(c, nrows=0)
            cols = {str(x).strip().lower() for x in hdr.columns}
            if "ncessch" in cols and "athletics_budget_proxy" in cols:
                return c
        except Exception:
            continue
    for c in cands:
        low = c.name.lower()
        if any(k in low for k in ("slfs", "proxy", "athletic", "nces")):
            try:
                hdr = pd.read_csv(c, nrows=0)
                cols = {str(x).strip().lower() for x in hdr.columns}
                if "ncessch" in cols and "athletics_budget_proxy" in cols:
                    return c
            except Exception:
                continue
    return None


def _discover_bmf_path() -> Path | None:
    """Prefer ``BMF_PATH``; else ``data/irs/*.csv`` that looks like an EO/BMF extract."""
    if BMF_PATH.exists():
        return BMF_PATH
    d = DATA_DIR / "irs"
    if not d.is_dir():
        return None
    cands = sorted(d.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for c in cands:
        try:
            hdr = pd.read_csv(c, nrows=0)
            cols = {str(x).strip().lower() for x in hdr.columns}
            if "ein" in cols and "name" in cols and ("zip" in cols or "zip_code" in cols or "mailing_zip" in cols):
                return c
        except Exception:
            continue
    return cands[0] if cands else None


def _resolve_zcta_shapefile() -> Path | None:
    """
    Find a local Census ZCTA shapefile.

    Uses ``SHAPE_SHP_PATH`` when present; otherwise the newest ``tl_*_us_zcta*.shp``
    under ``data/Shapes/`` (covers 2024/2025 renames without editing code).
    """
    if SHAPE_SHP_PATH.exists():
        return SHAPE_SHP_PATH
    shapes_dir = DATA_DIR / "Shapes"
    if not shapes_dir.is_dir():
        return None
    candidates = list(shapes_dir.glob("tl_*_us_zcta520.shp"))
    if not candidates:
        candidates = list(shapes_dir.glob("tl_*_us_zcta*.shp"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _mtime_or_missing(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return -1.0


def data_load_cache_key() -> tuple:
    """
    Values for ``@st.cache_data`` so loads refresh when inputs change.

    Important: if the app was first run without a ZCTA shapefile, Streamlit must not
    keep serving placeholder squares forever after you add ``tl_*_us_zcta*.shp``.
    """
    shp = _resolve_zcta_shapefile()
    slfs = _discover_slfs_proxy_path()
    bmf = _discover_bmf_path()
    return (
        _mtime_or_missing(GEOGRAPHIES_PATH),
        _mtime_or_missing(SCHOOLS_PATH),
        str(shp.resolve()) if shp is not None else "",
        _mtime_or_missing(shp) if shp is not None else -1.0,
        _mtime_or_missing(SHAPE_GEOJSON_PATH),
        float(ZCTA_FOLIUM_SIMPLIFY_DEGREES),
        _mtime_or_missing(slfs) if slfs is not None else -1.0,
        str(slfs.resolve()) if slfs is not None else "",
        _mtime_or_missing(bmf) if bmf is not None else -1.0,
        str(bmf.resolve()) if bmf is not None else "",
        _mtime_or_missing(ACS_INCOME_PATH),
        _mtime_or_missing(ACS_POVERTY_PATH),
    )


def _normalize_name(value: object) -> str:
    s = str(value or "").strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    # Drop common words that add little value for matching school support orgs.
    stop = {
        "inc",
        "llc",
        "co",
        "corp",
        "corporation",
        "foundation",
        "fund",
        "association",
        "booster",
        "club",
        "friends",
        "of",
        "the",
        "school",
        "pta",
        "pta",
        "parent",
        "teacher",
        "organization",
    }
    toks = [t for t in s.split() if t and t not in stop]
    return " ".join(toks)


def _token_overlap_score(left: str, right: str) -> float:
    lt = set(left.split())
    rt = set(right.split())
    if not lt or not rt:
        return 0.0
    return len(lt & rt) / max(1, len(lt | rt))


def _parse_tax_year(tax_period: object) -> float:
    s = str(tax_period or "").strip()
    if len(s) < 4 or not s[:4].isdigit():
        return np.nan
    return float(s[:4])


def _zip_to_string(series: pd.Series) -> pd.Series:
    """Convert to string and preserve leading zeros for 5-digit ZIPs."""
    s = series.astype(str).str.strip()
    mask = s.str.match(r"^\d+$") & (s.str.len() <= 5)
    return s.where(~mask, s.str.zfill(5))


def _zip_scalar(z: object) -> str:
    """Normalize a single ZIP to 5 characters for joins and BMF (STATE, ZIP) keys."""
    if z is None or (isinstance(z, float) and pd.isna(z)):
        return ""
    s = _zip_to_string(pd.Series([str(z).strip()])).iloc[0]
    if not isinstance(s, str) or not s:
        return ""
    return s.strip().zfill(5)[:5]


def _normalize_ncessch_id(series: pd.Series) -> pd.Series:
    """
    Align NCES school IDs across files (strip, drop trailing ``.0`` from floats, 12-digit zfill).
    """
    s = series.astype(str).str.strip()
    s = s.replace({"nan": "", "None": "", "<NA>": ""})
    s = s.str.replace(r"\.0$", "", regex=True)
    mask = s.str.match(r"^\d+$") & (s.str.len() > 0) & (s.str.len() < 12)
    s = s.where(~mask, s.str.zfill(12))
    return s


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
    _header = pd.read_csv(SCHOOLS_PATH, nrows=0)
    _dtype = {k: str for k in ("MZIP", "LZIP", "NCESSCH") if k in _header.columns}
    df = pd.read_csv(SCHOOLS_PATH, dtype=_dtype, low_memory=False)
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
    if "NCESSCH" in df.columns:
        df["ncessch"] = df["NCESSCH"].astype(str).str.strip()
    else:
        df["ncessch"] = ""
    sch_type = df.get("SCH_TYPE_TEXT", "").astype(str).str.lower()
    charter_yes = (
        df.get("CHARTER_TEXT", pd.Series("", index=df.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("yes")
    )
    not_private = ~sch_type.str.contains("private", na=False)
    # NCES often labels public schools as "Regular School" without the word "public".
    looks_public = sch_type.str.contains(
        r"public|regular school|^charter|alternative|special education|juvenile|vocational",
        na=False,
    )
    df["is_public_school"] = not_private & (looks_public | charter_yes)
    df["school_type"] = df.get("SCH_TYPE_TEXT", "").fillna("").astype(str).str.strip()
    df["charter_status"] = df.get("CHARTER_TEXT", "").fillna("").astype(str).str.strip()
    df["operational_status"] = df.get("SY_STATUS_TEXT", "").fillna("").astype(str).str.strip()
    if "PHONE" in df.columns:
        df["phone"] = df["PHONE"].fillna("").astype(str).str.strip()
    else:
        df["phone"] = ""
    if "WEBSITE" in df.columns:
        df["website"] = df["WEBSITE"].fillna("").astype(str).str.strip()
    else:
        df["website"] = ""
    return df


def _load_slfs_proxy() -> pd.DataFrame | None:
    """
    Load NCES SLFS FY2022-derived school proxy file.
    Required columns: ncessch, athletics_budget_proxy.
    """
    path = _discover_slfs_proxy_path()
    if path is None:
        return None
    try:
        df = pd.read_csv(path, dtype=str, low_memory=False)
    except Exception:
        return None
    cols = {str(c).strip().lower(): c for c in df.columns}
    id_col = cols.get("ncessch")
    proxy_col = cols.get("athletics_budget_proxy")
    if not id_col or not proxy_col:
        return None
    out = df[[id_col, proxy_col]].copy()
    out.columns = ["ncessch", "athletics_budget_proxy"]
    out["ncessch"] = _normalize_ncessch_id(out["ncessch"])
    out["athletics_budget_proxy"] = pd.to_numeric(out["athletics_budget_proxy"], errors="coerce")
    out["athletics_budget_proxy_source"] = "NCES_SLFS_FY2022"
    return out


def _bmf_column_lookup(df: pd.DataFrame) -> dict[str, str]:
    """Map canonical names to actual column names (case-insensitive)."""
    lower = {str(c).strip().lower(): c for c in df.columns}

    def one(*candidates: str) -> str | None:
        for c in candidates:
            if c.lower() in lower:
                return lower[c.lower()]
        return None

    return {
        "EIN": one("ein"),
        "NAME": one("name", "org_name", "organization_name"),
        "CITY": one("city", "mail_city"),
        "STATE": one("state", "st", "mail_state"),
        "ZIP": one("zip", "zip_code", "zip5", "mail_zip", "mailing_zip"),
        "REVENUE_AMT": one(
            "revenue_amt",
            "revenue",
            "income_amt",
            "income",
            "cy_total_revenue_amt",
            "total_revenue",
        ),
        "ASSET_AMT": one("asset_amt", "assets", "total_assets", "asset_amount"),
        "TAX_PERIOD": one("tax_period", "tax_yr", "tax_year"),
        "NTEE_CD": one("ntee_cd", "ntee_code", "ntee"),
    }


def _load_bmf() -> pd.DataFrame | None:
    """Load IRS EO / BMF-style extract from ``data/irs/`` (flexible filename and columns)."""
    path = _discover_bmf_path()
    if path is None:
        return None
    try:
        df = pd.read_csv(path, dtype=str, low_memory=False, on_bad_lines="skip")
    except TypeError:
        try:
            df = pd.read_csv(path, dtype=str, low_memory=False)
        except Exception:
            return None
    except Exception:
        return None
    if len(df) == 0:
        return None

    lk = _bmf_column_lookup(df)
    required = ("EIN", "NAME", "CITY", "STATE", "ZIP")
    if any(lk.get(k) is None for k in required):
        return None

    out = pd.DataFrame()
    for canon in required + ("REVENUE_AMT", "ASSET_AMT", "TAX_PERIOD", "NTEE_CD"):
        src = lk.get(canon)
        if src is not None:
            out[canon] = df[src]
        elif canon == "REVENUE_AMT":
            out[canon] = np.nan
        elif canon == "ASSET_AMT":
            out[canon] = np.nan
        elif canon == "TAX_PERIOD":
            out[canon] = ""
        elif canon == "NTEE_CD":
            out[canon] = ""

    z = out["ZIP"].fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    z = z.str.replace(r"[^0-9]", "", regex=True).str.slice(0, 5)
    out["ZIP"] = _zip_to_string(z)
    out["CITY"] = out["CITY"].fillna("").astype(str).str.strip().str.lower()
    out["STATE"] = out["STATE"].fillna("").astype(str).str.strip().str.upper()
    out["NAME"] = out["NAME"].fillna("").astype(str).str.strip()
    out["name_norm"] = out["NAME"].map(_normalize_name)
    out["REVENUE_AMT"] = pd.to_numeric(out["REVENUE_AMT"], errors="coerce")
    out["ASSET_AMT"] = pd.to_numeric(out["ASSET_AMT"], errors="coerce")
    out["tax_year"] = out["TAX_PERIOD"].map(_parse_tax_year)
    out = out[out["STATE"].astype(str).str.len() == 2]
    out = out[out["ZIP"].astype(str).str.match(r"^\d{5}$", na=False)]

    # Drop rows that are almost never legitimate K-12 booster / school-support matches
    # (hospitals, research institutes, sports halls, etc.) so ZIP-level heuristics do not latch onto them.
    bad = _bmf_unrelated_to_k12_support_mask(out)
    if bad.any():
        out = out.loc[~bad].copy()
    if len(out) == 0:
        return None
    return out


def _bmf_unrelated_to_k12_support_mask(df: pd.DataFrame) -> pd.Series:
    """
    True where an EO/BMF row should be ignored for school booster matching.

    These patterns address common false positives: hospitals in the same ZIP as a school,
    research institutes, university main campuses, sports halls of fame, United Way, etc.
    """
    name = df["NAME"].fillna("").astype(str)
    n = name.str.lower()
    ntee = (
        df.get("NTEE_CD", pd.Series("", index=df.index))
        .fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
        .str.slice(0, 3)
    )

    medical = n.str.contains(
        r"\bhospital\b|\bhospitals\b|\bmedical center\b|\bmedical group\b|\bhealth ?system\b|\bhealthcare\b|"
        r"\bhealth systems\b|\bregional health\b|\bcommunity health\b|\bphysician\b|\bphysicians\b|"
        r"\bsurgery center\b|\burgent care\b|\bnursing home\b|\bhospice\b|\brehabilitation center\b|"
        r"\brehab center\b|\bchildren'?s hospital\b|\bkids hospital\b|\bmedical college\b|"
        r"\bcollege of medicine\b|\bmedical school\b|\bone ?health\b|\bcancer center\b|\bcancer institute\b|"
        r"\bheart institute\b|\bhealth services\b|\bmedical campus\b",
        regex=True,
        na=False,
    )
    # IRS NTEE major groups: hospitals / inpatient (E2*) and many public-health orgs (E3*)
    ntee_health_block = ntee.str.match(r"^E[23]", na=False)

    hall_or_tourism = n.str.contains(
        r"\bhall of fame\b|\bpro football hall\b|\bcooperstown\b|\bsports commission\b|"
        r"\bvisitor'?s bureau\b|\bvisitors bureau\b|\bconvention\s+(?:and|&)\s+visitors\b|\bcvb\b",
        regex=True,
        na=False,
    ) & ~n.str.contains(
        r"\b(?:booster|school|education|pta|athletic|student|band|orchestra|alumni|gridiron)\b",
        regex=True,
        na=False,
    )

    research = n.str.contains(
        r"\bresearch institute\b|\bresearch foundation\b|\bresearch corporation\b|\bresearch consortium\b|"
        r"\bresearch triangle\b|\bfederally funded research\b|\bfederal research\b",
        regex=True,
        na=False,
    )

    university = n.str.contains(r"\buniversity\b", na=False) & ~n.str.contains(
        r"\b(?:charter|public schools?|k-12|elementary|middle school|high school|"
        r"booster|pta|alumni association|prep academy|academy cs)\b",
        regex=True,
        na=False,
    )

    united_way = n.str.contains(r"\bunited way\b", na=False)

    return medical | ntee_health_block | hall_or_tourism | research | university | united_way


def _likely_school_support_booster_mask(df: pd.DataFrame) -> pd.Series:
    """Heuristic: org name / NTEE looks like PTA, booster, athletics, or school support."""
    name = df["NAME"].fillna("").astype(str).str.lower()
    ntee = df.get("NTEE_CD", pd.Series("", index=df.index)).fillna("").astype(str).str.upper()
    pat = (
        r"\bpta\b|p\.t\.a|parent.?teacher|booster|athletic|athletics|\bsports\b|"
        r"football|basketball|soccer|baseball|softball|volleyball|wrestling|lacrosse|"
        r"gridiron|cheer|band|orchestra|marching|drill|student.?athlete|"
        r"school.?foundation|education.?foundation|friends.?of|alumni|grid|club"
    )
    m_name = name.str.contains(pat, regex=True, na=False)
    m_ntee = ntee.str.match(r"^B[0-9]", na=False) | ntee.isin(
        ["N69", "O50", "O52", "P20", "P30", "P40", "P50", "P60"]
    )
    return m_name | m_ntee


def _pick_best_booster_row_by_revenue(df: pd.DataFrame) -> pd.Series | None:
    if df is None or len(df) == 0:
        return None
    rev = pd.to_numeric(df["REVENUE_AMT"], errors="coerce").fillna(0.0)
    if rev.max() <= 0:
        return None
    idx = rev.idxmax()
    return df.loc[idx]


def _booster_match_dict_from_row(row: pd.Series, conf: float) -> dict[str, object]:
    return {
        "booster_exists": True,
        "booster_match_confidence": float(conf),
        "matched_org_name": str(row.get("NAME", "")),
        "matched_ein": str(row.get("EIN", "")),
        "latest_booster_revenue": row.get("REVENUE_AMT", np.nan),
        "latest_booster_expenses": np.nan,
        "latest_booster_net_assets": row.get("ASSET_AMT", np.nan),
        "latest_booster_tax_year": row.get("tax_year", np.nan),
    }


def _resolve_booster_loc(
    state: str,
    zip5: str,
    city: str,
    gz: Any,
    gs: Any,
    sz_cache: dict[tuple[str, str], pd.DataFrame],
    state_cache: dict[str, pd.DataFrame],
    empty: pd.DataFrame,
) -> pd.DataFrame:
    """Same geographic narrowing as before, but group lookups instead of scanning all of BMF."""
    k = (state, zip5)
    if k not in sz_cache:
        try:
            sz_cache[k] = gz.get_group(k)
        except KeyError:
            sz_cache[k] = empty
    block = sz_cache[k]
    if len(block) > 0:
        by_city = block[block["CITY"] == city]
        return by_city if len(by_city) > 0 else block
    if state not in state_cache:
        try:
            state_cache[state] = gs.get_group(state)
        except KeyError:
            state_cache[state] = empty
    return state_cache[state]


def _match_booster_for_school(
    school_name: object,
    city: str,
    state: str,
    zip5: str,
    loc: pd.DataFrame,
) -> dict[str, object]:
    empty = {
        "booster_exists": False,
        "booster_match_confidence": 0.0,
        "matched_org_name": "",
        "matched_ein": "",
        "latest_booster_revenue": np.nan,
        "latest_booster_expenses": np.nan,
        "latest_booster_net_assets": np.nan,
        "latest_booster_tax_year": np.nan,
    }
    if loc.empty:
        return empty

    zip5 = _zip_scalar(zip5)
    if not zip5:
        return empty

    school_name_norm = _normalize_name(school_name)
    city = str(city).strip().lower()
    state = str(state).strip().upper()

    # Always score **same-ZIP** EO rows only so we never attach an unrelated large org elsewhere in the state.
    loc_zip = loc["ZIP"].astype(str).str.strip().str.zfill(5).str.slice(0, 5)
    zip_rows = loc[loc_zip == zip5].copy()
    if zip_rows.empty:
        return empty

    zip_rows["name_score"] = zip_rows["name_norm"].map(lambda s: _token_overlap_score(school_name_norm, s))
    zip_rows["city_score"] = (zip_rows["CITY"] == city).astype(float)
    zip_rows["zip_score"] = 1.0
    zip_rows["state_score"] = (zip_rows["STATE"] == state).astype(float)
    zip_rows["booster_match_confidence"] = (
        0.50 * zip_rows["name_score"]
        + 0.22 * zip_rows["city_score"]
        + 0.18 * zip_rows["zip_score"]
        + 0.10 * zip_rows["state_score"]
    )
    best = zip_rows.sort_values("booster_match_confidence", ascending=False).iloc[0]
    conf = float(best["booster_match_confidence"])
    best_rev = pd.to_numeric(best.get("REVENUE_AMT"), errors="coerce")
    # Require positive revenue for name-based matches too (EO rows with blank revenue are not useful signals).
    if conf >= _BOOSTER_CONF_NAME_MATCH and pd.notna(best_rev) and float(best_rev) > 0:
        return _booster_match_dict_from_row(best, conf)

    # Heuristic: PTA / booster / athletics language or education NTEE → strongest revenue in ZIP.
    mask = _likely_school_support_booster_mask(zip_rows)
    pool = zip_rows[mask] if mask.any() else zip_rows
    pick = _pick_best_booster_row_by_revenue(pool)
    conf_hz = _BOOSTER_CONF_ZIP_HEURISTIC
    if pick is None:
        pick = _pick_best_booster_row_by_revenue(zip_rows)
        conf_hz = _BOOSTER_CONF_ZIP_BROAD
    if pick is None:
        return empty

    rev = pd.to_numeric(pick.get("REVENUE_AMT"), errors="coerce")
    if pd.isna(rev) or float(rev) <= 0:
        return empty

    return _booster_match_dict_from_row(pick, max(conf, conf_hz))


def _match_boosters_by_location_fast(
    schools: pd.DataFrame,
    gz: Any,
    gs: Any,
    empty: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fast booster matching for large school datasets.

    Instead of scoring every school name against every org candidate, compute one
    best-match signal per (state, zip, city) and map back to all schools in that
    location. This keeps ZIP-level booster signals useful while cutting startup time.
    """
    loc_df = schools[["state", "zip_code", "city"]].copy()
    loc_df["state"] = loc_df["state"].fillna("").astype(str).str.strip().str.upper()
    loc_df["zip_code"] = loc_df["zip_code"].map(_zip_scalar)
    loc_df["city"] = loc_df["city"].fillna("").astype(str).str.strip().str.lower()
    unique_locs = loc_df.drop_duplicates(ignore_index=True)

    sz_cache: dict[tuple[str, str], pd.DataFrame] = {}
    state_cache: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    for loc in unique_locs.itertuples(index=False):
        st = str(loc.state)
        zp = str(loc.zip_code)
        cty = str(loc.city)
        block = _resolve_booster_loc(st, zp, cty, gz, gs, sz_cache, state_cache, empty)
        if len(block) == 0:
            rows.append(
                {
                    "state": st,
                    "zip_code": zp,
                    "city": cty,
                    "booster_exists": False,
                    "booster_match_confidence": 0.0,
                    "matched_org_name": "",
                    "matched_ein": "",
                    "latest_booster_revenue": np.nan,
                    "latest_booster_expenses": np.nan,
                    "latest_booster_net_assets": np.nan,
                    "latest_booster_tax_year": np.nan,
                }
            )
            continue

        mask = _likely_school_support_booster_mask(block)
        pool = block[mask] if mask.any() else block
        pick = _pick_best_booster_row_by_revenue(pool)
        conf = _BOOSTER_CONF_ZIP_HEURISTIC if pick is not None else _BOOSTER_CONF_ZIP_BROAD
        if pick is None:
            pick = _pick_best_booster_row_by_revenue(block)
        if pick is None:
            match = {
                "booster_exists": False,
                "booster_match_confidence": 0.0,
                "matched_org_name": "",
                "matched_ein": "",
                "latest_booster_revenue": np.nan,
                "latest_booster_expenses": np.nan,
                "latest_booster_net_assets": np.nan,
                "latest_booster_tax_year": np.nan,
            }
        else:
            rev = pd.to_numeric(pick.get("REVENUE_AMT"), errors="coerce")
            match = _booster_match_dict_from_row(pick, conf) if pd.notna(rev) and float(rev) > 0 else {
                "booster_exists": False,
                "booster_match_confidence": 0.0,
                "matched_org_name": "",
                "matched_ein": "",
                "latest_booster_revenue": np.nan,
                "latest_booster_expenses": np.nan,
                "latest_booster_net_assets": np.nan,
                "latest_booster_tax_year": np.nan,
            }
        rows.append({"state": st, "zip_code": zp, "city": cty, **match})

    per_loc = pd.DataFrame(rows)
    out = loc_df.merge(per_loc, on=["state", "zip_code", "city"], how="left")
    out.index = schools.index
    return out[
        [
            "booster_exists",
            "booster_match_confidence",
            "matched_org_name",
            "matched_ein",
            "latest_booster_revenue",
            "latest_booster_expenses",
            "latest_booster_net_assets",
            "latest_booster_tax_year",
        ]
    ]


def _enrich_school_funding_fields(schools: pd.DataFrame, *, match_boosters: bool = True) -> pd.DataFrame:
    out = schools.copy()
    if len(out) == 0:
        return out
    out["zip_code"] = _zip_to_string(out["zip_code"])
    if "city" not in out.columns:
        out["city"] = ""
    if "state" not in out.columns:
        out["state"] = ""
    if "is_public_school" not in out.columns:
        out["is_public_school"] = False
    out["city"] = out["city"].astype(str).str.strip()
    out["state"] = out["state"].astype(str).str.strip().str.upper()
    out["is_public_school"] = out["is_public_school"].fillna(False).astype(bool)

    if "ncessch" in out.columns:
        out["ncessch"] = _normalize_ncessch_id(out["ncessch"])

    out["athletics_budget_proxy"] = np.nan
    out["athletics_budget_proxy_source"] = ""
    slfs = _load_slfs_proxy()
    if slfs is not None and "ncessch" in out.columns:
        out = out.merge(slfs, on="ncessch", how="left", suffixes=("", "_slfs"))
        # Use NCES SLFS proxy only for public schools; label source only when SLFS row exists.
        has_slfs = out["athletics_budget_proxy_slfs"].notna()
        out["athletics_budget_proxy"] = np.where(
            out["is_public_school"],
            out["athletics_budget_proxy_slfs"],
            np.nan,
        )
        out["athletics_budget_proxy_source"] = np.where(
            out["is_public_school"] & has_slfs,
            "NCES_SLFS_FY2022",
            "",
        )
        out = out.drop(columns=[c for c in ("athletics_budget_proxy_slfs",) if c in out.columns])

    # Ensure private schools are null unless a future private-school source is added.
    out.loc[~out["is_public_school"], "athletics_budget_proxy"] = np.nan
    out.loc[~out["is_public_school"], "athletics_budget_proxy_source"] = ""

    out["booster_exists"] = False
    out["booster_match_confidence"] = 0.0
    out["matched_org_name"] = ""
    out["matched_ein"] = ""
    out["latest_booster_revenue"] = np.nan
    out["latest_booster_expenses"] = np.nan
    out["latest_booster_net_assets"] = np.nan
    out["latest_booster_tax_year"] = np.nan

    bmf = _load_bmf() if match_boosters else None
    if bmf is not None:
        empty = bmf.iloc[:0]
        gz = bmf.groupby(["STATE", "ZIP"], sort=False)
        gs = bmf.groupby("STATE", sort=False)
        if len(out) >= _BOOSTER_FAST_MATCH_MIN_ROWS:
            match_df = _match_boosters_by_location_fast(out, gz, gs, empty)
        else:
            sz_cache: dict[tuple[str, str], pd.DataFrame] = {}
            state_cache: dict[str, pd.DataFrame] = {}
            merged_rows: list[dict[str, object]] = []
            for row in out.itertuples(index=False):
                st = str(getattr(row, "state", "") or "").strip().upper()
                zp = _zip_scalar(getattr(row, "zip_code", ""))
                cty = str(getattr(row, "city", "") or "").strip().lower()
                loc = _resolve_booster_loc(st, zp, cty, gz, gs, sz_cache, state_cache, empty)
                merged_rows.append(
                    _match_booster_for_school(getattr(row, "school_name", ""), cty, st, zp, loc)
                )
            match_df = pd.DataFrame(merged_rows, index=out.index)
        for c in match_df.columns:
            out[c] = match_df[c]

    # --- Athletics proxy: ensure every **public** school has a dollar value ($) for reporting ---
    # 1) Peer ZIP mean: other public schools in this ZIP that already have SLFS (same NCES file).
    peers = out[out["is_public_school"] & out["athletics_budget_proxy"].notna()]
    if len(peers) > 0:
        peer_zip_mean = peers.groupby("zip_code", as_index=False)["athletics_budget_proxy"].mean()
        peer_zip_mean = peer_zip_mean.rename(columns={"athletics_budget_proxy": "_peer_zip_mean_proxy"})
        out = out.merge(peer_zip_mean, on="zip_code", how="left")
    else:
        out["_peer_zip_mean_proxy"] = np.nan

    need_peer = out["is_public_school"] & out["athletics_budget_proxy"].isna() & out["_peer_zip_mean_proxy"].notna()
    out.loc[need_peer, "athletics_budget_proxy"] = out.loc[need_peer, "_peer_zip_mean_proxy"]
    src_peer = out.loc[need_peer, "athletics_budget_proxy_source"].astype(str)
    out.loc[need_peer, "athletics_budget_proxy_source"] = np.where(
        src_peer.str.strip().isin(("", "nan", "None")),
        "ZIP_PEER_MEAN_SLFS",
        src_peer,
    )
    out = out.drop(columns=["_peer_zip_mean_proxy"], errors="ignore")

    # 2) Dataset-wide median of public proxies (covers ZIPs with no SLFS peers at all).
    med = out.loc[out["is_public_school"], "athletics_budget_proxy"].median()
    if pd.isna(med):
        med = 0.0
    need_med = out["is_public_school"] & out["athletics_budget_proxy"].isna()
    if need_med.any():
        out.loc[need_med, "athletics_budget_proxy"] = float(med)
        src_m = out.loc[need_med, "athletics_budget_proxy_source"].astype(str)
        out.loc[need_med, "athletics_budget_proxy_source"] = np.where(
            src_m.str.strip().isin(("", "nan", "None")),
            "GLOBAL_MEDIAN_PUBLIC_PROXY",
            src_m,
        )

    # 3) ZIP-level aggregates (geography + exports): single merge after all per-school proxies are final.
    zip_agg_cols = [
        "athletics_budget_proxy_zip",
        "booster_exists_zip",
        "booster_match_confidence_zip",
        "latest_booster_revenue_zip",
        "latest_booster_net_assets_zip",
        "latest_booster_tax_year_zip",
    ]
    for c in zip_agg_cols:
        if c in out.columns:
            out = out.drop(columns=[c], errors="ignore")
    zip_sigs = _aggregate_school_funding_signals(out)
    out = out.merge(zip_sigs, on="zip_code", how="left")

    return out


def _compute_school_count(geo: pd.DataFrame, schools: pd.DataFrame) -> pd.Series:
    counts = schools.groupby("zip_code").size()
    return geo["zip_code"].map(counts).fillna(0).astype(int)


def _aggregate_school_funding_signals(schools: pd.DataFrame) -> pd.DataFrame:
    if len(schools) == 0:
        return pd.DataFrame(columns=["zip_code"])
    grouped = schools.groupby("zip_code", dropna=False)
    agg = grouped.agg(
        athletics_budget_proxy_zip=("athletics_budget_proxy", "mean"),
        booster_exists_zip=("booster_exists", "max"),
        booster_match_confidence_zip=("booster_match_confidence", "mean"),
        # max avoids inflating ZIP totals when many schools inherit the same ZIP-level match.
        latest_booster_revenue_zip=("latest_booster_revenue", "max"),
        latest_booster_net_assets_zip=("latest_booster_net_assets", "max"),
        latest_booster_tax_year_zip=("latest_booster_tax_year", "max"),
    ).reset_index()
    agg["booster_exists_zip"] = agg["booster_exists_zip"].fillna(False).astype(bool)
    return agg


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
                    "ncessch": "",
                    "is_public_school": True,
                    "school_type": "Regular School",
                    "charter_status": "No",
                    "operational_status": "Open",
                    "phone": "",
                    "website": "",
                }
            )
    return pd.DataFrame(rows)


def load_schools(geographies: pd.DataFrame | None = None, *, match_boosters: bool = True) -> pd.DataFrame:
    """Load schools with internal column names.

    ``match_boosters=False`` skips IRS EO/BMF row matching (much faster for scripts;
    booster columns stay at defaults).
    """
    if SCHOOLS_PATH.exists():
        df = _load_real_schools()
    else:
        geo = geographies if geographies is not None else _generate_dummy_geographies()
        df = _generate_dummy_schools(geo)
    df["zip_code"] = _zip_to_string(df["zip_code"])
    return _enrich_school_funding_fields(df, match_boosters=match_boosters)


def load_geographies(schools: pd.DataFrame | None = None, *, match_boosters: bool = True) -> pd.DataFrame:
    """Load geographies, enrich with ACS, and compute school_count and distance."""
    if GEOGRAPHIES_PATH.exists():
        df = _load_real_geographies()
    else:
        df = _generate_dummy_geographies()

    if schools is None:
        schools = load_schools(df, match_boosters=match_boosters)
    df["school_count"] = _compute_school_count(df, schools)
    df["distance_to_ann_arbor_miles"] = _compute_distance_to_ann_arbor(df)
    funding = _aggregate_school_funding_signals(schools)
    if len(funding) > 0:
        df = df.merge(funding, on="zip_code", how="left")
    df["zip_code"] = _zip_to_string(df["zip_code"])
    return df


def load_zip_shapes(geographies: pd.DataFrame | None = None) -> tuple[dict, str]:
    """
    ZIP / ZCTA polygons for the Heatmap.

    Resolution order:
    1. ``data/Shapes/zcta_boundaries.geojson`` if present (must be GeoJSON
       Features with ``properties.zip_code`` or Census ``ZCTA5CE20``).
    2. ``data/Shapes/tl_*_us_zcta520.shp`` (or any ``tl_*_us_zcta*.shp``): TIGER/Line
       ZCTA polygons, simplified for the browser and cached to ``zcta_folium_cache.geojson``.
    3. Fallback: small squares around each geography centroid (legacy look).

    Returns ``(geojson_feature_collection, source)`` where ``source`` is one of
    ``\"geojson\"``, ``\"tiger\"``, or ``\"placeholder\"``.
    """
    geo = geographies if geographies is not None else load_geographies()
    zcta = _load_zcta_geojson_from_optional_files(geo)
    if zcta is not None:
        return zcta
    return _generate_dummy_geojson(geo), "placeholder"


def _zip_codes_for_shapes(geographies: pd.DataFrame) -> set[str]:
    return set(_zip_to_string(geographies["zip_code"]).astype(str).str.zfill(5))


def _zcta_fingerprint(shp_path: Path, zips: set[str], simplify: float) -> str:
    h = hashlib.sha256()
    mtime = shp_path.stat().st_mtime
    h.update(f"{shp_path.name}:{mtime}:{simplify}:".encode())
    h.update(",".join(sorted(zips)).encode())
    return h.hexdigest()


def _load_zcta_geojson_from_optional_files(geographies: pd.DataFrame) -> tuple[dict, str] | None:
    """Real ZCTA boundaries from GeoJSON or local TIGER shapefile; else None."""
    zips = _zip_codes_for_shapes(geographies)
    if SHAPE_GEOJSON_PATH.exists():
        try:
            with open(SHAPE_GEOJSON_PATH, encoding="utf-8") as f:
                gj = json.load(f)
            filtered = _filter_geojson_to_zips(gj, zips)
            if filtered is not None:
                return filtered, "geojson"
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass

    shp_path = _resolve_zcta_shapefile()
    if shp_path is None:
        return None

    try:
        import geopandas as gpd  # noqa: WPS433 — heavy; only when shapes exist
    except ImportError:
        return None

    fp = _zcta_fingerprint(shp_path, zips, ZCTA_FOLIUM_SIMPLIFY_DEGREES)
    if ZCTA_FOLIUM_CACHE_PATH.exists() and ZCTA_FOLIUM_CACHE_META_PATH.exists():
        try:
            with open(ZCTA_FOLIUM_CACHE_META_PATH, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("fingerprint") == fp:
                with open(ZCTA_FOLIUM_CACHE_PATH, encoding="utf-8") as f:
                    return json.load(f), "tiger"
        except (json.JSONDecodeError, OSError, TypeError):
            pass

    gdf = gpd.read_file(shp_path)
    if "ZCTA5CE20" not in gdf.columns:
        return None
    gdf = gdf[gdf["ZCTA5CE20"].astype(str).isin(zips)].copy()
    if len(gdf) == 0:
        return None

    gdf["zip_code"] = gdf["ZCTA5CE20"].astype(str).str.zfill(5)
    gdf["geometry"] = gdf.geometry.simplify(ZCTA_FOLIUM_SIMPLIFY_DEGREES, preserve_topology=True)
    gdf = gdf[~gdf.geometry.is_empty]

    out = json.loads(gdf[["geometry", "zip_code"]].to_json())
    try:
        ZCTA_FOLIUM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ZCTA_FOLIUM_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(out, f, separators=(",", ":"))
        with open(ZCTA_FOLIUM_CACHE_META_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "fingerprint": fp,
                    "features": len(out.get("features", [])),
                    "source_shp": shp_path.name,
                },
                f,
            )
    except OSError:
        pass

    return out, "tiger"


def _filter_geojson_to_zips(gj: dict, zips: set[str]) -> dict | None:
    feats = []
    for feat in gj.get("features", []):
        if not isinstance(feat, dict):
            continue
        props = feat.get("properties") or {}
        z = props.get("zip_code") or props.get("ZCTA5CE20") or props.get("GEOID")
        if z is None:
            continue
        z = str(z).strip().zfill(5)[-5:]
        if z not in zips:
            continue
        feat = dict(feat)
        feat.setdefault("properties", {})
        feat["properties"]["zip_code"] = z
        feats.append(feat)
    if not feats:
        return None
    return {"type": "FeatureCollection", "features": feats}


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
        "athletics_budget_proxy_zip",
        "booster_exists_zip",
        "booster_match_confidence_zip",
        "latest_booster_revenue_zip",
        "latest_booster_net_assets_zip",
        "latest_booster_tax_year_zip",
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
        "school_type",
        "charter_status",
        "operational_status",
        "phone",
        "website",
        "latitude",
        "longitude",
        "ncessch",
        "is_public_school",
        "athletics_budget_proxy",
        "athletics_budget_proxy_source",
        "booster_exists",
        "booster_match_confidence",
        "matched_org_name",
        "matched_ein",
        "latest_booster_revenue",
        "latest_booster_expenses",
        "latest_booster_net_assets",
        "latest_booster_tax_year",
        "athletics_budget_proxy_zip",
        "booster_exists_zip",
        "booster_match_confidence_zip",
        "latest_booster_revenue_zip",
        "latest_booster_net_assets_zip",
        "latest_booster_tax_year_zip",
    ]

