"""
SunBundle Expansion Decision Tool
Interactive dashboard for ranking ZIP codes / geographies for school-based shoe donations.
"""

from __future__ import annotations


import math
import sys
from pathlib import Path

# Ensure project root is on path so "utils" is found when run from any directory
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import data_load_cache_key, load_geographies, load_schools, load_zip_shapes
from utils.filters import apply_filters
from utils.mapping import build_choropleth, build_school_map_only
from utils.school_geocode import ensure_school_coordinates
from utils.scoring import run_scoring
from utils.exports import prepare_ranked_export, prepare_schools_export
from utils.ranked_column_docs import RANKED_COLUMN_HELP

# Must be first Streamlit command
st.set_page_config(
    page_title="SunBundle Expansion Decision Tool",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in miles between two lat/lon points."""
    radius_miles = 3959.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return float(radius_miles * c)


def _resolve_reference_location(geo_df: pd.DataFrame, user_input: str) -> tuple[float, float, str]:
    """
    Resolve a reference location from data already loaded in geographies.
    Supports ZIP code and city/state matching.
    """
    raw = (user_input or "").strip()
    if not raw:
        raw = "Ann Arbor, MI"

    # ZIP lookup first if the input is numeric.
    zip_digits = "".join(ch for ch in raw if ch.isdigit())
    if len(zip_digits) == 5 and "zip_code" in geo_df.columns:
        m = geo_df[geo_df["zip_code"].astype(str) == zip_digits]
        if len(m) > 0:
            r = m.iloc[0]
            if pd.notna(r.get("latitude")) and pd.notna(r.get("longitude")):
                label = f"{zip_digits} — {r.get('city', '')}, {r.get('state', '')}".strip(", ")
                return float(r["latitude"]), float(r["longitude"]), label

    # City/state matching (exact then contains).
    city_q = raw
    state_q = ""
    if "," in raw:
        parts = [p.strip() for p in raw.split(",", 1)]
        city_q = parts[0]
        state_q = parts[1]

    tmp = geo_df.copy()
    if "city" in tmp.columns:
        tmp["_city"] = tmp["city"].fillna("").astype(str).str.strip().str.lower()
    else:
        tmp["_city"] = ""
    if "state" in tmp.columns:
        tmp["_state"] = tmp["state"].fillna("").astype(str).str.strip().str.lower()
    else:
        tmp["_state"] = ""

    city_q_norm = city_q.strip().lower()
    state_q_norm = state_q.strip().lower()
    exact = tmp[tmp["_city"] == city_q_norm]
    if state_q_norm:
        exact = exact[exact["_state"].str.contains(state_q_norm, na=False)]
    if len(exact) == 0 and city_q_norm:
        contains = tmp[tmp["_city"].str.contains(city_q_norm, na=False)]
        if state_q_norm:
            contains = contains[contains["_state"].str.contains(state_q_norm, na=False)]
        exact = contains

    exact = exact.dropna(subset=["latitude", "longitude"]) if len(exact) > 0 else exact
    if len(exact) > 0:
        lat = float(exact["latitude"].astype(float).mean())
        lon = float(exact["longitude"].astype(float).mean())
        city_label = exact.iloc[0].get("city", city_q.title())
        state_label = exact.iloc[0].get("state", state_q.upper())
        label = f"{city_label}, {state_label}".strip(", ")
        return lat, lon, label

    # Fallback to Ann Arbor centroid if not found.
    fallback = tmp[
        tmp["_city"].str.contains("ann arbor", na=False) & tmp["_state"].str.contains("mi", na=False)
    ].dropna(subset=["latitude", "longitude"])
    if len(fallback) > 0:
        r = fallback.iloc[0]
        return float(r["latitude"]), float(r["longitude"]), "Ann Arbor, MI"
    return 42.2808, -83.7430, "Ann Arbor, MI"


def explain_top_zip(
    row: pd.Series, df: pd.DataFrame, distance_column: str, reference_location_label: str
) -> str:
    """Simple rule-based explanation for the top recommended ZIP."""
    reasons: list[str] = []
    if "poverty_rate" in row and pd.notna(row["poverty_rate"]):
        if row["poverty_rate"] >= df["poverty_rate"].median():
            reasons.append("elevated poverty")
    if "median_household_income" in row and pd.notna(row["median_household_income"]):
        if row["median_household_income"] <= df["median_household_income"].median():
            reasons.append("lower household income")
    if "school_count" in row and pd.notna(row["school_count"]):
        if row["school_count"] >= df["school_count"].median():
            reasons.append("a strong concentration of schools")
    if distance_column in row and pd.notna(row[distance_column]):
        if row[distance_column] <= df[distance_column].median():
            reasons.append(f"reasonable distance from {reference_location_label}")

    if not reasons:
        return (
            "This ZIP ranks highly based on the current balance of need and feasibility "
            "weights you selected."
        )

    if len(reasons) == 1:
        core = reasons[0]
    else:
        core = ", ".join(reasons[:-1]) + f", and {reasons[-1]}"
    return (
        "This ZIP ranks highly because it combines "
        f"{core}, while remaining reasonably feasible operationally."
    )


def _sf_is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, (float, np.floating)) and pd.isna(v):
        return True
    if isinstance(v, str) and not v.strip():
        return True
    return False


def _sf_int(v) -> str:
    if _sf_is_missing(v):
        return "—"
    try:
        return f"{int(float(v)):,}"
    except (TypeError, ValueError):
        return "—"


def _sf_float1(v) -> str:
    if _sf_is_missing(v):
        return "—"
    try:
        return f"{float(v):,.1f}"
    except (TypeError, ValueError):
        return "—"


def _sf_money(v) -> str:
    if _sf_is_missing(v):
        return "—"
    try:
        return f"${float(v):,.0f}"
    except (TypeError, ValueError):
        return "—"


def _sf_pct1(v) -> str:
    if _sf_is_missing(v):
        return "—"
    try:
        return f"{float(v):.1f}%"
    except (TypeError, ValueError):
        return "—"


def _sf_score(v) -> str:
    if _sf_is_missing(v):
        return "—"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "—"


def _sf_miles(v) -> str:
    if _sf_is_missing(v):
        return "—"
    try:
        return f"{float(v):,.0f} mi"
    except (TypeError, ValueError):
        return "—"


def _sf_yesno(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "Yes" if v else "No"
    if _sf_is_missing(v):
        return "—"
    s = str(v).strip().lower()
    if s in ("yes", "y", "true", "1"):
        return "Yes"
    if s in ("no", "n", "false", "0"):
        return "No"
    return str(v)


# ----- Sidebar: primary navigation -----
st.sidebar.title("SunBundle Expansion Decision Tool")
page = st.sidebar.radio(
    "Go to",
    ["Matrix Overview", "Ranked Geographies", "Heatmap", "School Finder"],
    key="nav",
)
st.sidebar.markdown("---")

# Default criteria so every tab uses the same keys before first visit to Matrix Overview
_CRITERIA_DEFAULTS = {
    "strategy_preset": "Balanced Expansion",
    "overall_need_pct": 70,
    "need_poverty": 45,
    "need_income": 35,
    "need_schools": 20,
    "feas_density": 50,
    "feas_distance": 50,
    "reference_location_input": "Ann Arbor, MI",
    "max_income": 100000,
    "min_poverty": 0.0,
    "max_distance": 500.0,
    "min_schools": 1,
    "state_filter": "",
    "athletics_proxy_mode": "any",
    "booster_support_mode": "any",
    "min_athletics_proxy_zip": 0.0,
    "min_booster_revenue_zip": 0.0,
}
for _k, _v in _CRITERIA_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Strategy presets
presets = {
    "Balanced Expansion": {
        "overall_need_pct": 70,
        "need_poverty": 45,
        "need_income": 35,
        "need_schools": 20,
        "feas_density": 50,
        "feas_distance": 50,
    },
    "Highest Need": {
        "overall_need_pct": 85,
        "need_poverty": 55,
        "need_income": 30,
        "need_schools": 15,
        "feas_density": 40,
        "feas_distance": 60,
    },
    "Closest Pilot Markets": {
        "overall_need_pct": 50,
        "need_poverty": 40,
        "need_income": 30,
        "need_schools": 30,
        "feas_density": 30,
        "feas_distance": 70,
    },
}

@st.cache_data
def _cached_geo_schools(_cache_key: tuple):
    """ZIP rows + school rows. ZCTA polygons are loaded separately for the Heatmap only."""
    schools = load_schools(None)
    geo = load_geographies(schools)
    return geo, schools


@st.cache_data
def _cached_zip_shapes(_cache_key: tuple):
    """ZCTA / placeholder GeoJSON — only needed on the Heatmap tab (can be large)."""
    geo, _schools = _cached_geo_schools(_cache_key)
    return load_zip_shapes(geo)


_data_key = data_load_cache_key()
geo_raw, schools_df = _cached_geo_schools(_data_key)
if page == "Heatmap":
    zip_geojson, zip_shape_source = _cached_zip_shapes(_data_key)
else:
    zip_geojson = {"type": "FeatureCollection", "features": []}
    zip_shape_source = "skipped"

state_options = [""]
if "state" in geo_raw.columns and len(geo_raw) > 0:
    state_options = [""] + sorted(geo_raw["state"].dropna().astype(str).unique().tolist())

preset_options = list(presets.keys()) + ["Custom"]

if page == "Matrix Overview":
    st.title("SunBundle Expansion Decision Tool")
    st.caption("Set weights and filters below — **Heatmap**, **Ranked Geographies**, and **School Finder** use the same criteria.")
    with st.expander("Methodology & data sources", expanded=False):
        st.markdown(
            """
This tool helps SunBundle decide which ZIP codes are the strongest candidates for expansion by combining
**community need** and **operational feasibility** into one ranked score.

The app starts by loading and joining multiple datasets:
- **Geographies dataset** (`data/Geographies/geographies.csv`) provides ZIP-level baseline attributes such as city/state,
  latitude/longitude, population, and density.
- **Schools dataset** (`data/Schools/schools.csv`) provides school records used to estimate potential reach in each ZIP.
  The NCES file often has no lat/lon columns; **School Finder** fills coordinates on demand with the **U.S. Census**
  oneline geocoder (mailing address + city + state + ZIP) and caches results under `data/.cache/school_geocode.json`.
- **ACS income and poverty files** (`data/acs/income.csv`, `data/acs/poverty.csv`) provide socioeconomic indicators
  (median household income and poverty rate) by ZIP/ZCTA.
- **ZIP / ZCTA shape data** (`data/Shapes/…`): the Heatmap prefers real Census ZCTA polygons from a local shapefile when
  available (cached as simplified GeoJSON for the browser). If those files are missing, the map falls back to small
  squares around each ZIP centroid (faster to set up, but coarser).

After loading the data, the app standardizes ZIP codes, cleans numeric fields, and builds an analysis table where each
row is one ZIP code. It then computes two important derived fields:
1. **School count** per ZIP by aggregating schools into each geography.
2. **Distance from your selected reference location** (for example, Columbus, OH) using latitude/longitude and a
   great-circle distance formula. This lets you test expansion scenarios anchored to different base markets.

Next, the app applies your **hard filters**. These are strict pass/fail requirements, not preferences. A ZIP is removed
if it violates any selected threshold (for example: income too high, poverty too low, too few schools, too far away, or
outside the selected state). Only ZIP codes that pass all active filters move forward.

For the remaining ZIPs, the app computes scores in three stages:
1. **Normalize each metric** to a common 0-1 scale so different units can be compared fairly.
2. Build a **Need score** from poverty rate, inverse median income, and school count.
3. Build a **Feasibility score** from population density and distance to your chosen reference location.
4. Build a **Funding signal score** from public-school athletics budget proxy (NCES SLFS), booster existence,
   booster funding size, and confidence-adjusted booster matching.

Finally, the app calculates a blended total from core scoring + funding signals with a confidence penalty for weak matches.

The sliders control two layers of weighting:
- **Overall split** between Need and Feasibility.
- **Within-component splits** (for example, how much poverty matters inside Need).

What you see afterward:
- **Matrix Overview**: explanation + current weighting structure + top recommendation.
- **Ranked Geographies**: full prioritized ZIP list with exports.
- **Heatmap**: geographic view of high vs low score areas.
- **School Finder**: school-level view for selected ZIPs, including **ZIP / ACS demographics**, **expansion scores**
  for that ZIP, and an **NCES + funding-signal** table (school type, charter status, proxies, booster matches).

Important interpretation note: scores are **relative to the currently filtered candidate set**. If you tighten filters
or change reference location, score distributions and rankings can shift meaningfully.
"""
        )
    with st.expander("What each control means", expanded=False):
        st.write("- **Strategy preset**: quick starting profile for common expansion approaches.")
        st.write("- **Need vs Feasibility**: sets the high-level balance of impact vs execution practicality.")
        st.write("- **Need score weights**: prioritizes poverty, lower income, and school reach.")
        st.write("- **Feasibility score weights**: balances concentrated populations with travel distance.")
        st.write("- **Funding signals**: use proxy public spending + booster signals, not exact school-level athletics spend.")
        st.write(
            "- **Hard filters**: strict cutoffs (income, poverty, distance, school count, state, athletics proxy, booster) "
            "that must all pass."
        )
        st.write(
            "- **Reference location**: city/state or ZIP used for distance calculations "
            "(for example, `Columbus, OH` or `43215`)."
        )
    st.markdown("---")
    st.subheader("Adjust criteria")
    st.caption(
        "Same settings apply across all tabs. On other pages, use the sidebar expanders "
        "**Scoring weights** and **Hard filters** (same keys as here)."
    )

if page == "Matrix Overview":
    selected_preset = st.selectbox(
        "Strategy preset",
        preset_options,
        key="strategy_preset",
        help="Starting mix of weights. You can still fine-tune every slider below.",
    )
else:
    selected_preset = st.sidebar.selectbox(
        "Strategy preset",
        preset_options,
        key="strategy_preset",
        help="Starting mix of weights. Fine-tune with the controls below.",
    )

# Initialize or update preset-driven weights (before sliders so preset changes apply same run)
if "current_preset" not in st.session_state:
    st.session_state["current_preset"] = selected_preset

if selected_preset != st.session_state["current_preset"] and selected_preset in presets:
    p = presets[selected_preset]
    st.session_state["overall_need_pct"] = p["overall_need_pct"]
    st.session_state["need_poverty"] = p["need_poverty"]
    st.session_state["need_income"] = p["need_income"]
    st.session_state["need_schools"] = p["need_schools"]
    st.session_state["feas_density"] = p["feas_density"]
    st.session_state["feas_distance"] = p["feas_distance"]
    st.session_state["current_preset"] = selected_preset

if page == "Matrix Overview":
    tab_weights, tab_filters = st.tabs(["Scoring weights", "Hard filters"])
    with tab_weights:
        st.caption(
            "Tune how much **Need** vs **Feasibility** matters overall, then how each metric contributes inside those scores."
        )
        st.markdown("##### Overall balance")
        need_pct = st.slider(
            "Slidebar displayed is out of 100. The need weight % is in red, while the feasibility is in grey. The slider is the overall balance of need and feasibility.",
            min_value=0,
            max_value=100,
            key="overall_need_pct",
            help="Higher = prioritize poverty, income, and schools over density and distance.",
        )
        feas_pct = 100 - need_pct
        st.caption(f"Feasibility weight: **{feas_pct}%**")
        st.divider()
        st.markdown("##### Need score — relative weights")
        st.caption("How to weight poverty, income, and school reach *within* the Need score.")
        nc1, nc2, nc3 = st.columns(3)
        with nc1:
            need_poverty = st.slider(
                "Poverty",
                0,
                100,
                key="need_poverty",
                help="Higher weight = emphasize ZIPs with higher poverty rates.",
            )
        with nc2:
            need_income = st.slider(
                "Median income",
                0,
                100,
                key="need_income",
                help="Higher weight = emphasize ZIPs with lower median household income.",
            )
        with nc3:
            need_schools = st.slider(
                "School count",
                0,
                100,
                key="need_schools",
                help="Higher weight = emphasize ZIPs with more schools (more reach).",
            )
        st.divider()
        st.markdown("##### Feasibility score — relative weights")
        st.caption("How to weight density vs distance *within* the Feasibility score.")
        fc1, fc2 = st.columns(2)
        with fc1:
            feas_density = st.slider(
                "Population density",
                0,
                100,
                key="feas_density",
                help="Higher weight = prefer more concentrated populations.",
            )
        with fc2:
            feas_distance = st.slider(
                "Distance to reference",
                0,
                100,
                key="feas_distance",
                help="Higher weight = prefer ZIPs closer to your reference location.",
            )

    with tab_filters:
        st.caption(
            "Hard cutoffs: a ZIP must pass **every** rule here before it is scored. "
            "Use this to narrow the map to realistic service areas."
        )
        reference_location_input = st.text_input(
            "Reference location (city, state, or ZIP)",
            key="reference_location_input",
            help="Used for distance-to-reference and the max-distance filter below.",
        )
        r1, r2, r3 = st.columns(3)
        with r1:
            max_income = st.number_input(
                "Max median household income ($)",
                min_value=0,
                step=5000,
                key="max_income",
                help="Exclude ZIPs with median income above this amount.",
            )
        with r2:
            min_poverty = st.number_input(
                "Min poverty rate (%)",
                min_value=0.0,
                step=1.0,
                key="min_poverty",
                help="Exclude ZIPs with poverty below this percentage.",
            )
        with r3:
            max_distance = st.number_input(
                "Max distance from reference (mi)",
                min_value=0.0,
                step=50.0,
                key="max_distance",
                help="Exclude ZIPs farther than this from the reference point.",
            )
        r4, r5 = st.columns(2)
        with r4:
            min_schools = st.number_input(
                "Min school count",
                min_value=0,
                step=1,
                key="min_schools",
                help="Exclude ZIPs with fewer schools than this.",
            )
        with r5:
            state_filter = st.selectbox(
                "State (optional)",
                state_options,
                key="state_filter",
                help="Leave blank for all states, or pick one to restrict candidates.",
            )
        st.markdown("##### Funding & booster (ZIP-level)")
        st.caption(
            "Athletics proxy comes from NCES SLFS (not exact per-school athletics spend). "
            "Booster signals come from IRS EO BMF matching (not verified as athletics-only)."
        )
        fa, fb = st.columns(2)
        with fa:
            athletics_proxy_mode = st.selectbox(
                "Athletics budget proxy",
                options=["any", "has_proxy", "no_proxy"],
                format_func=lambda x: {
                    "any": "No filter",
                    "has_proxy": "Only ZIPs with proxy data",
                    "no_proxy": "Only ZIPs without proxy data",
                }[x],
                key="athletics_proxy_mode",
                help="Filter ZIPs by whether public-school proxy spending is available.",
            )
            min_athletics_proxy_zip = st.number_input(
                "Min mean athletics proxy ($)",
                min_value=0.0,
                step=1000.0,
                key="min_athletics_proxy_zip",
                help="Require ZIP mean proxy ≥ this amount. 0 = no minimum.",
            )
        with fb:
            booster_support_mode = st.selectbox(
                "Booster / support org (EO BMF)",
                options=["any", "has_booster", "no_booster"],
                format_func=lambda x: {
                    "any": "No filter",
                    "has_booster": "Only ZIPs with a matched org",
                    "no_booster": "Only ZIPs with no matched org",
                }[x],
                key="booster_support_mode",
                help="Filter by whether a booster/support nonprofit was matched in this ZIP.",
            )
            min_booster_revenue_zip = st.number_input(
                "Min booster revenue in ZIP ($)",
                min_value=0.0,
                step=5000.0,
                key="min_booster_revenue_zip",
                help="Require summed matched-org revenue ≥ this. 0 = no minimum.",
            )
else:
    st.sidebar.subheader("Scoring weights")
    need_pct = st.sidebar.slider(
        "Slidebar displayed is out of 100. The need weight % is in red, while the feasibility is in grey. The slider is the overall balance of need and feasibility.",
        min_value=0,
        max_value=100,
        key="overall_need_pct",
        help="Higher = prioritize poverty, income, and schools over density and distance.",
    )
    feas_pct = 100 - need_pct
    st.sidebar.caption(f"Feasibility weight: **{feas_pct}%**")
    need_poverty = st.sidebar.slider(
        "Poverty (within Need)",
        0,
        100,
        key="need_poverty",
        help="Higher weight = emphasize ZIPs with higher poverty rates.",
    )
    need_income = st.sidebar.slider(
        "Median income (within Need)",
        0,
        100,
        key="need_income",
        help="Higher weight = emphasize ZIPs with lower median household income.",
    )
    need_schools = st.sidebar.slider(
        "School count (within Need)",
        0,
        100,
        key="need_schools",
        help="Higher weight = emphasize ZIPs with more schools.",
    )
    feas_density = st.sidebar.slider(
        "Population density (within Feasibility)",
        0,
        100,
        key="feas_density",
        help="Higher weight = prefer more concentrated populations.",
    )
    feas_distance = st.sidebar.slider(
        "Distance to reference (within Feasibility)",
        0,
        100,
        key="feas_distance",
        help="Higher weight = prefer ZIPs closer to your reference location.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Hard filters")
    reference_location_input = st.sidebar.text_input(
        "Reference location (city, state, or ZIP)",
        key="reference_location_input",
        help="Used for distance-to-reference and the max-distance filter.",
    )
    max_income = st.sidebar.number_input(
        "Max median household income ($)",
        min_value=0,
        step=5000,
        key="max_income",
        help="Exclude ZIPs with median income above this amount.",
    )
    min_poverty = st.sidebar.number_input(
        "Min poverty rate (%)",
        min_value=0.0,
        step=1.0,
        key="min_poverty",
        help="Exclude ZIPs with poverty below this percentage.",
    )
    max_distance = st.sidebar.number_input(
        "Max distance from reference (mi)",
        min_value=0.0,
        step=50.0,
        key="max_distance",
        help="Exclude ZIPs farther than this from the reference point.",
    )
    min_schools = st.sidebar.number_input(
        "Min school count",
        min_value=0,
        step=1,
        key="min_schools",
        help="Exclude ZIPs with fewer schools than this.",
    )
    state_filter = st.sidebar.selectbox(
        "State (optional)",
        state_options,
        key="state_filter",
        help="Leave blank for all states.",
    )
    st.sidebar.caption("Funding & booster (ZIP-level)")
    athletics_proxy_mode = st.sidebar.selectbox(
        "Athletics budget proxy",
        options=["any", "has_proxy", "no_proxy"],
        format_func=lambda x: {
            "any": "No filter",
            "has_proxy": "Only ZIPs with proxy data",
            "no_proxy": "Only ZIPs without proxy data",
        }[x],
        key="athletics_proxy_mode",
    )
    min_athletics_proxy_zip = st.sidebar.number_input(
        "Min mean athletics proxy ($)",
        min_value=0.0,
        step=1000.0,
        key="min_athletics_proxy_zip",
    )
    booster_support_mode = st.sidebar.selectbox(
        "Booster / support org",
        options=["any", "has_booster", "no_booster"],
        format_func=lambda x: {
            "any": "No filter",
            "has_booster": "Only ZIPs with a matched org",
            "no_booster": "Only ZIPs with no matched org",
        }[x],
        key="booster_support_mode",
    )
    min_booster_revenue_zip = st.sidebar.number_input(
        "Min booster revenue in ZIP ($)",
        min_value=0.0,
        step=5000.0,
        key="min_booster_revenue_zip",
    )

reference_lat, reference_lon, reference_label = _resolve_reference_location(
    geo_raw, reference_location_input
)
if page == "Matrix Overview":
    st.caption(f"Resolved reference: **{reference_label}**")
else:
    st.sidebar.caption(f"Resolved reference: **{reference_label}**")

geo_raw = geo_raw.copy()
geo_raw["distance_to_reference_miles"] = geo_raw.apply(
    lambda r: _haversine_miles(r["latitude"], r["longitude"], reference_lat, reference_lon)
    if pd.notna(r.get("latitude")) and pd.notna(r.get("longitude"))
    else float("nan"),
    axis=1,
)

state_filter = state_filter if state_filter else None

if page == "Matrix Overview":
    st.markdown("---")
else:
    st.sidebar.markdown("---")

# ----- Apply filters and scoring -----
filtered = apply_filters(
    geo_raw,
    max_median_household_income=max_income or None,
    min_poverty_rate=min_poverty or None,
    max_distance_miles=max_distance or None,
    distance_column="distance_to_reference_miles",
    min_school_count=min_schools or None,
    state_filter=state_filter,
    athletics_proxy_mode=athletics_proxy_mode,
    min_athletics_budget_proxy_zip=min_athletics_proxy_zip if min_athletics_proxy_zip > 0 else None,
    booster_support_mode=booster_support_mode,
    min_booster_revenue_zip=min_booster_revenue_zip if min_booster_revenue_zip > 0 else None,
)

need_weights = {
    "poverty_rate": need_poverty,
    "median_household_income": need_income,
    "school_count": need_schools,
}
feas_weights = {
    "density": feas_density,
    "distance_to_reference_miles": feas_distance,
}

overall_need_weight = need_pct / 100.0
overall_feas_weight = feas_pct / 100.0

scored = run_scoring(
    filtered,
    need_weights=need_weights,
    feasibility_weights=feas_weights,
    overall_need_weight=overall_need_weight,
    overall_feasibility_weight=overall_feas_weight,
)
scored = scored.sort_values("total_score", ascending=False).reset_index(drop=True)

if page != "Matrix Overview":
    st.sidebar.markdown("---")
    st.sidebar.caption("**Active criteria** (same sliders as Matrix Overview)")
    st.sidebar.write(f"- Need / Feasibility: **{need_pct}%** / **{feas_pct}%**")
    st.sidebar.write(f"- Reference: **{reference_label}**")
    st.sidebar.write(
        f"- Athletics proxy filter: **{athletics_proxy_mode}**"
        + (f" (min ${min_athletics_proxy_zip:,.0f})" if min_athletics_proxy_zip > 0 else "")
    )
    st.sidebar.write(
        f"- Booster filter: **{booster_support_mode}**"
        + (f" (min rev ${min_booster_revenue_zip:,.0f})" if min_booster_revenue_zip > 0 else "")
    )
    st.sidebar.write(f"- ZIPs after filters: **{len(filtered)}** → ranked: **{len(scored)}**")

if page == "Matrix Overview":
    st.markdown("---")

    st.subheader("Target Geography Identification Matrix")
    st.markdown(
        f"""
| Score (100%) | Weight |
|---|---|
| **Need Score** | **{need_pct}%** |
| **Feasibility Score** | **{feas_pct}%** |
"""
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Need Score")
        st.write(f"- Poverty rate: **{need_poverty}%**")
        st.write(f"- Median household income (inverse): **{need_income}%**")
        st.write(f"- School count: **{need_schools}%**")
    with c2:
        st.markdown("#### Feasibility Score")
        st.write(f"- Population density: **{feas_density}%**")
        st.write(f"- Distance from {reference_label}: **{feas_distance}%**")

    st.info(
        f"Current reference point is **{reference_label}**. "
        "Use the controls above to tune the matrix and hard filters."
    )

    st.markdown("---")

    st.subheader("Top Recommended ZIP")
    if len(scored) == 0:
        st.info("No geographies passed your filters. Loosen hard filters above or on the Matrix Overview page.")
    else:
        top_row = scored.iloc[0]
        st.write(f"**{top_row['zip_code']} — {top_row.get('city', '')}, {top_row.get('state', '')}**")
        st.write(
            f"Total score: **{top_row['total_score']:.2f}**  "
            f"(Need: {top_row['need_score']:.2f}, Feasibility: {top_row['feasibility_score']:.2f})"
        )
        st.caption(
            explain_top_zip(
                top_row,
                scored,
                distance_column="distance_to_reference_miles",
                reference_location_label=reference_label,
            )
        )

    st.markdown("---")
    st.subheader("Quick navigation")
    st.write("- **Ranked Geographies**: full ranked table + CSV export")
    st.write("- **Heatmap**: map view of total scores")
    st.write("- **School Finder**: school lists by selected ZIP")

# ----- Ranked Geographies -----
elif page == "Ranked Geographies":
    st.title("Ranked Geographies")
    st.caption(
        "All geographies passing your filters, ordered by total score (same weights as Matrix Overview). "
        "Search by ZIP or city."
    )
    st.markdown("---")

    if len(scored) == 0:
        st.warning("No geographies passed your filters. Loosen filters on Matrix Overview (or the sidebar).")
    else:
        _help_choice = st.selectbox(
            "Plain-English guide (pick a column or topic)",
            options=list(RANKED_COLUMN_HELP.keys()),
            key="ranked_column_help",
            help="Short, intuitive explanations. Start with the Overview option.",
        )
        st.caption("Tip: read **Overview — how ranking works** first; other rows explain one column at a time.")
        with st.container():
            st.markdown(RANKED_COLUMN_HELP[_help_choice].strip())
        st.markdown("---")

        # Simple text search by ZIP or city
        search = st.text_input("Search ZIP or city", "", key="ranked_search").strip()

        ranked = scored.copy()
        if search:
            s = search.lower()
            zip_match = ranked["zip_code"].astype(str).str.contains(s, case=False, na=False)
            city_match = ranked.get("city", "").astype(str).str.contains(s, case=False, na=False)
            ranked = ranked[zip_match | city_match]

        ranked.insert(0, "rank", range(1, len(ranked) + 1))
        table_cols = [
            "rank",
            "zip_code",
            "city",
            "state",
            "total_score",
            "need_score",
            "feasibility_score",
            "funding_signal_score",
            "confidence_penalty",
            "poverty_rate",
            "median_household_income",
            "school_count",
            "population",
            "density",
            "distance_to_reference_miles",
            "athletics_budget_proxy_zip",
            "booster_exists_zip",
            "booster_match_confidence_zip",
            "latest_booster_revenue_zip",
        ]
        display_df = ranked[[c for c in table_cols if c in ranked.columns]]

        column_config: dict[str, st.column_config.Column] = {}
        if "total_score" in display_df.columns:
            column_config["total_score"] = st.column_config.NumberColumn("Score", format="%.2f")
        if "need_score" in display_df.columns:
            column_config["need_score"] = st.column_config.NumberColumn("Need", format="%.2f")
        if "feasibility_score" in display_df.columns:
            column_config["feasibility_score"] = st.column_config.NumberColumn("Feasibility", format="%.2f")
        if "funding_signal_score" in display_df.columns:
            column_config["funding_signal_score"] = st.column_config.NumberColumn("Funding signal", format="%.2f")
        if "confidence_penalty" in display_df.columns:
            column_config["confidence_penalty"] = st.column_config.NumberColumn("Confidence penalty", format="%.2f")
        if "poverty_rate" in display_df.columns:
            column_config["poverty_rate"] = st.column_config.NumberColumn("Poverty %", format="%.1f")
        if "median_household_income" in display_df.columns:
            column_config["median_household_income"] = st.column_config.NumberColumn(
                "Median HH income ($)", format="$%d"
            )
        if "population" in display_df.columns:
            column_config["population"] = st.column_config.NumberColumn("Population", format="%d")
        if "density" in display_df.columns:
            column_config["density"] = st.column_config.NumberColumn("Density", format="%.1f")
        if "distance_to_reference_miles" in display_df.columns:
            column_config["distance_to_reference_miles"] = st.column_config.NumberColumn(
                "Distance (mi)", format="%.0f"
            )
        if "athletics_budget_proxy_zip" in display_df.columns:
            column_config["athletics_budget_proxy_zip"] = st.column_config.NumberColumn(
                "Athletics budget proxy", format="$%.0f"
            )
        if "booster_exists_zip" in display_df.columns:
            column_config["booster_exists_zip"] = st.column_config.CheckboxColumn("Booster exists")
        if "booster_match_confidence_zip" in display_df.columns:
            column_config["booster_match_confidence_zip"] = st.column_config.NumberColumn(
                "Booster match confidence", format="%.2f"
            )
        if "latest_booster_revenue_zip" in display_df.columns:
            column_config["latest_booster_revenue_zip"] = st.column_config.NumberColumn(
                "Latest booster revenue", format="$%.0f"
            )
        if "school_count" in display_df.columns:
            column_config["school_count"] = st.column_config.NumberColumn("Schools", format="%d")
        if "zip_code" in display_df.columns:
            column_config["zip_code"] = st.column_config.TextColumn("ZIP")

        st.dataframe(display_df, use_container_width=True, hide_index=True, column_config=column_config)

        export_df = prepare_ranked_export(scored)
        st.download_button(
            "Download ranked geographies (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="sunbundle_ranked_geographies.csv",
            mime="text/csv",
            key="dl_geo",
        )

# ----- Heatmap -----
elif page == "Heatmap":
    st.title("Heatmap")
    st.caption(
        "Choropleth by expansion score (same total scores as Matrix Overview). Darker = higher score."
    )
    st.markdown("---")

    if zip_shape_source == "placeholder":
        st.warning(
            "Heatmap is using **placeholder squares** (centroid boxes) because no ZCTA boundary file was "
            "found under `data/Shapes/`. For real ZIP outlines, add a Census TIGER file such as "
            "`tl_2025_us_zcta520.shp` (plus `.dbf`, `.shx`, `.prj`, …) or place `zcta_boundaries.geojson` there, "
            "then use **C → Reload** or restart Streamlit so the data cache refreshes."
        )
    else:
        st.caption(
            f"Boundaries: **{'ZCTA (GeoJSON)' if zip_shape_source == 'geojson' else 'ZCTA (TIGER shapefile)'}** "
            f"— not placeholder squares."
        )

    if len(scored) == 0:
        st.warning("No geographies to show. Loosen filters on Matrix Overview (or the sidebar).")
    else:
        try:
            m = build_choropleth(zip_geojson, scored, score_column="total_score", zip_column="zip_code")
            if m is None:
                st.warning(
                    "Map could not be drawn — **folium** may be missing or ZIP shapes unavailable. "
                    "In the same environment you use for `streamlit run`, run: "
                    "`pip install folium` or `pip install -r requirements.txt`."
                )
            elif hasattr(st, "folium_chart"):
                st.folium_chart(m, use_container_width=True)
            else:
                import streamlit.components.v1 as components

                components.html(m._repr_html_(), height=500, scrolling=False)
        except Exception as e:  # pragma: no cover - runtime safety
            st.warning(f"Map could not be drawn: {e}")

# ----- School Finder -----
elif page == "School Finder":
    st.title("School Finder")
    st.caption("Pick a ZIP to see schools in that area and download the list.")
    st.markdown("---")

    if len(scored) > 0:
        zip_display = scored.apply(
            lambda r: f"{r['zip_code']} — {r.get('city', '')}, {r.get('state', '')}", axis=1
        )
        zip_options = ["— Select a ZIP —"] + zip_display.tolist()
        zip_to_code = {zip_options[i]: scored.iloc[i - 1]["zip_code"] for i in range(1, len(zip_options))}
    else:
        zip_options = ["— Select a ZIP —"]
        zip_to_code = {}

    selected_label = st.selectbox("ZIP code (select from ranked list)", zip_options, key="school_zip")

    if selected_label and selected_label != "— Select a ZIP —":
        selected_zip = zip_to_code.get(
            selected_label,
            selected_label.split("—")[0].strip() if "—" in selected_label else selected_label,
        )
        selected_zip = str(selected_zip).strip()

        znorm = str(selected_zip).strip().zfill(5)
        school_list = schools_df[schools_df["zip_code"].astype(str).str.strip().str.zfill(5) == znorm]
        if len(school_list) == 0:
            st.info(f"No schools found in ZIP **{selected_zip}** in the dataset.")
        else:
            st.subheader(f"Schools in ZIP {znorm}")

            grow = geo_raw[geo_raw["zip_code"].astype(str).str.strip().str.zfill(5) == znorm]
            srow_df = scored[scored["zip_code"].astype(str).str.strip().str.zfill(5) == znorm]

            st.markdown("##### ZIP & community profile")
            st.caption(
                "Demographics are **ZIP / ZCTA** (geographies + ACS). Scores match your **current** "
                "sidebar filters and weights (same pipeline as Ranked Geographies)."
            )
            if len(grow) == 0:
                st.warning("No geography baseline row for this ZIP.")
            else:
                gr = grow.iloc[0]
                sr = srow_df.iloc[0] if len(srow_df) else None
                r1 = st.columns(4)
                r1[0].metric("Population", _sf_int(gr.get("population")))
                r1[1].metric("Density (per sq mi)", _sf_float1(gr.get("density")))
                r1[2].metric("Median household income", _sf_money(gr.get("median_household_income")))
                r1[3].metric("Poverty rate", _sf_pct1(gr.get("poverty_rate")))

                r2 = st.columns(4)
                sch_n = (sr.get("school_count") if sr is not None else None) or gr.get("school_count")
                r2[0].metric("Schools in ZIP (NCES count)", _sf_int(sch_n))
                dist_src = sr if sr is not None else gr
                r2[1].metric("Distance from reference", _sf_miles(dist_src.get("distance_to_reference_miles")))
                if sr is not None:
                    r2[2].metric("Total score (0–1)", _sf_score(sr.get("total_score")))
                    r2[3].metric("Need score", _sf_score(sr.get("need_score")))
                else:
                    r2[2].metric("Total score (0–1)", "—")
                    r2[3].metric("Need score", "—")

                if sr is not None:
                    r3 = st.columns(5)
                    r3[0].metric("Feasibility score", _sf_score(sr.get("feasibility_score")))
                    r3[1].metric("Funding signal score", _sf_score(sr.get("funding_signal_score")))
                    r3[2].metric("Athletics proxy (ZIP mean)", _sf_money(sr.get("athletics_budget_proxy_zip")))
                    r3[3].metric("Booster org in ZIP", _sf_yesno(sr.get("booster_exists_zip")))
                    r3[4].metric("Booster revenue (ZIP)", _sf_money(sr.get("latest_booster_revenue_zip")))
                    if len(srow_df) and "booster_match_confidence_zip" in srow_df.columns:
                        bc = sr.get("booster_match_confidence_zip")
                        if not _sf_is_missing(bc):
                            st.caption(
                                f"Booster ZIP match confidence: **{float(bc):.2f}** "
                                "(1 = strong aggregate match)."
                            )

            st.markdown("##### School directory")
            cols = [
                "school_name",
                "district_name",
                "address",
                "city",
                "state",
                "enrollment",
                "grades",
            ]
            cols = [c for c in cols if c in school_list.columns]
            st.dataframe(school_list[cols], use_container_width=True, hide_index=True)

            detail_cols = [
                "school_name",
                "school_type",
                "charter_status",
                "operational_status",
                "grades",
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
            detail_present = [c for c in detail_cols if c in school_list.columns]
            if detail_present:
                with st.expander("School-level NCES fields & funding signals we matched", expanded=False):
                    st.caption(
                        "School type and charter fields come from NCES. **Per-school** athletics proxy comes from "
                        "NCES SLFS when the school ID matches; if it is missing, we may show the **ZIP mean** "
                        "(same as geography scoring). **`*_zip` columns** are ZIP roll-ups across schools in that ZIP. "
                        "Booster fields are **estimated** IRS EO BMF matches."
                    )
                    dc: dict = {}
                    if "website" in detail_present:
                        dc["website"] = st.column_config.TextColumn("Website")
                    if "booster_match_confidence" in detail_present:
                        dc["booster_match_confidence"] = st.column_config.NumberColumn(
                            "Booster match (0–1)", format="%.2f"
                        )
                    if "latest_booster_revenue" in detail_present:
                        dc["latest_booster_revenue"] = st.column_config.NumberColumn(
                            "Booster revenue ($)", format="$%d"
                        )
                    if "latest_booster_net_assets" in detail_present:
                        dc["latest_booster_net_assets"] = st.column_config.NumberColumn(
                            "Booster net assets ($)", format="$%d"
                        )
                    if "athletics_budget_proxy" in detail_present:
                        dc["athletics_budget_proxy"] = st.column_config.NumberColumn(
                            "Athletics proxy ($)", format="$%d"
                        )
                    if "athletics_budget_proxy_zip" in detail_present:
                        dc["athletics_budget_proxy_zip"] = st.column_config.NumberColumn(
                            "ZIP mean proxy ($)", format="$%d"
                        )
                    if "latest_booster_revenue_zip" in detail_present:
                        dc["latest_booster_revenue_zip"] = st.column_config.NumberColumn(
                            "ZIP max booster rev ($)", format="$%d"
                        )
                    if "latest_booster_net_assets_zip" in detail_present:
                        dc["latest_booster_net_assets_zip"] = st.column_config.NumberColumn(
                            "ZIP max booster assets ($)", format="$%d"
                        )
                    if dc:
                        st.dataframe(
                            school_list[detail_present],
                            use_container_width=True,
                            hide_index=True,
                            column_config=dc,
                        )
                    else:
                        st.dataframe(
                            school_list[detail_present],
                            use_container_width=True,
                            hide_index=True,
                        )

            export_schools = prepare_schools_export(school_list)
            st.download_button(
                "Download school list (CSV)",
                data=export_schools.to_csv(index=False).encode("utf-8"),
                file_name=f"sunbundle_schools_{selected_zip}.csv",
                mime="text/csv",
                key="dl_schools",
            )

            if "latitude" in school_list.columns and "longitude" in school_list.columns:
                st.subheader("Map")
                zip_centroid: tuple[float, float] | None = None
                gmatch = geo_raw[geo_raw["zip_code"].astype(str).str.strip().str.zfill(5) == znorm]
                if (
                    len(gmatch) > 0
                    and pd.notna(gmatch.iloc[0].get("latitude"))
                    and pd.notna(gmatch.iloc[0].get("longitude"))
                ):
                    zip_centroid = (
                        float(gmatch.iloc[0]["latitude"]),
                        float(gmatch.iloc[0]["longitude"]),
                    )

                missing_coords = school_list["latitude"].isna() | school_list["longitude"].isna()
                if missing_coords.any():
                    with st.spinner("Looking up school locations (U.S. Census geocoder)…"):
                        school_list_map = ensure_school_coordinates(
                            school_list.copy(), zip_centroid=zip_centroid
                        )
                else:
                    school_list_map = school_list.copy()

                has_any_coords = school_list_map["latitude"].notna().any() and school_list_map[
                    "longitude"
                ].notna().any()
                school_map = build_school_map_only(school_list_map)
                if school_map is not None:
                    if hasattr(st, "folium_chart"):
                        st.folium_chart(school_map, use_container_width=True)
                    else:
                        import streamlit.components.v1 as components

                        components.html(school_map._repr_html_(), height=500, scrolling=False)
                elif has_any_coords:
                    st.warning(
                        "Map could not be drawn — **folium** is likely missing. "
                        "Run `pip install folium` or `pip install -r requirements.txt` in the same "
                        "environment as `streamlit run`."
                    )
                else:
                    st.info(
                        "Could not resolve map coordinates (geocoder unavailable or address too vague). "
                        "Check your network connection and try again."
                    )
    else:
        st.info("Select a ZIP code above to see schools and download the list.")

