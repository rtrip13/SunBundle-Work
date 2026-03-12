"""
SunBundle Expansion Decision Tool
Interactive dashboard for ranking ZIP codes / geographies for school-based shoe donations.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path so "utils" is found when run from any directory
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from utils.data_loader import load_geographies, load_schools, load_zip_shapes
from utils.filters import apply_filters
from utils.mapping import build_choropleth, build_school_map_only
from utils.scoring import run_scoring
from utils.exports import prepare_ranked_export, prepare_schools_export

# Must be first Streamlit command
st.set_page_config(
    page_title="SunBundle Expansion Decision Tool",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded",
)


def explain_top_zip(row: pd.Series, df: pd.DataFrame) -> str:
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
    if "distance_to_ann_arbor_miles" in row and pd.notna(row["distance_to_ann_arbor_miles"]):
        if row["distance_to_ann_arbor_miles"] <= df["distance_to_ann_arbor_miles"].median():
            reasons.append("reasonable distance from Ann Arbor")

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


# ----- Sidebar: strategy, weights, filters -----
st.sidebar.title("SunBundle Expansion Decision Tool")
st.sidebar.markdown("---")

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

preset_options = list(presets.keys()) + ["Custom"]
selected_preset = st.sidebar.selectbox("Strategy preset", preset_options, key="strategy_preset")

# Initialize or update preset-driven weights
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

st.sidebar.subheader("Need vs Feasibility weighting")
need_pct = st.sidebar.slider(
    "Need weight (%)",
    min_value=0,
    max_value=100,
    value=st.session_state.get("overall_need_pct", 70),
    key="overall_need_pct",
)
feas_pct = 100 - need_pct
st.sidebar.caption(f"Feasibility weight: **{feas_pct}%**")

st.sidebar.subheader("Need score weights")
need_poverty = st.sidebar.slider(
    "Poverty rate (higher = more need)", 0, 100, st.session_state.get("need_poverty", 45), key="need_poverty"
)
need_income = st.sidebar.slider(
    "Median household income (lower = more need)", 0, 100, st.session_state.get("need_income", 35), key="need_income"
)
need_schools = st.sidebar.slider(
    "School count (more schools = more reach)", 0, 100, st.session_state.get("need_schools", 20), key="need_schools"
)

st.sidebar.subheader("Feasibility score weights")
feas_density = st.sidebar.slider(
    "Population density (higher = more concentrated)",
    0,
    100,
    st.session_state.get("feas_density", 50),
    key="feas_density",
)
feas_distance = st.sidebar.slider(
    "Distance to Ann Arbor (closer = better)",
    0,
    100,
    st.session_state.get("feas_distance", 50),
    key="feas_distance",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Hard filters")
st.sidebar.caption("Only geographies that meet ALL of these are included.")

max_income = st.sidebar.number_input(
    "Max median household income ($)", min_value=0, value=100000, step=5000, key="max_income"
)
min_poverty = st.sidebar.number_input(
    "Min poverty rate (%)", min_value=0.0, value=0.0, step=1.0, key="min_poverty"
)
max_distance = st.sidebar.number_input(
    "Max distance to Ann Arbor (miles)", min_value=0.0, value=500.0, step=50.0, key="max_distance"
)
min_schools = st.sidebar.number_input(
    "Min school count", min_value=0, value=1, step=1, key="min_schools"
)


@st.cache_data
def load_all_data():
    """Load core datasets once per session (cached)."""
    schools = load_schools(None)
    geo = load_geographies(schools)
    shapes = load_zip_shapes(geo)
    return geo, schools, shapes


geo_raw, schools_df, zip_geojson = load_all_data()

# State filter options from geographies
state_options = [""]
if "state" in geo_raw.columns and len(geo_raw) > 0:
    state_options = [""] + sorted(geo_raw["state"].dropna().astype(str).unique().tolist())
state_filter = st.sidebar.selectbox("State (optional)", state_options, key="state_filter")
state_filter = state_filter if state_filter else None

st.sidebar.markdown("---")

# ----- Apply filters and scoring -----
filtered = apply_filters(
    geo_raw,
    max_median_household_income=max_income or None,
    min_poverty_rate=min_poverty or None,
    max_distance_miles=max_distance or None,
    min_school_count=min_schools or None,
    state_filter=state_filter,
)

need_weights = {
    "poverty_rate": need_poverty,
    "median_household_income": need_income,
    "school_count": need_schools,
}
feas_weights = {
    "density": feas_density,
    "distance_to_ann_arbor_miles": feas_distance,
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

# ----- Navigation -----
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Criteria Builder", "Ranked Geographies", "Heatmap", "School Finder"],
    key="nav",
)

# Temporary debug so you can see which page Streamlit thinks is active.
# You can safely delete this once navigation feels correct.
st.write("DEBUG page:", page)

# ----- Overview -----
if page == "Overview":
    st.title("SunBundle Expansion Decision Tool")
    st.markdown(
        "Decide **which ZIP codes or cities to expand into** for school-based shoe donations, "
        "balancing socioeconomic need with operational feasibility."
    )
    st.markdown("---")

    # Top recommended ZIP with explanation
    st.subheader("Top Recommended ZIP")
    if len(scored) == 0:
        st.info("No geographies passed your filters. Loosen the filters in the sidebar to see recommendations.")
    else:
        top_row = scored.iloc[0]
        st.write(
            f"**{top_row['zip_code']} — {top_row.get('city', '')}, {top_row.get('state', '')}**"
        )
        st.write(
            f"Total score: **{top_row['total_score']:.2f}**  "
            f"(Need: {top_row['need_score']:.2f}, Feasibility: {top_row['feasibility_score']:.2f})"
        )
        st.caption(explain_top_zip(top_row, scored))

    st.markdown("---")

    st.subheader("How the score works")
    st.info(
        "We build a **Need score** from poverty, household income, and school count, "
        "and a **Feasibility score** from density and distance to Ann Arbor. "
        "You control the weights in the sidebar (e.g. 70% Need / 30% Feasibility)."
    )

    st.subheader("Current weights")
    summary = pd.DataFrame(
        [
            {"Component": "Overall", "Item": "Need", "Weight %": need_pct},
            {"Component": "Overall", "Item": "Feasibility", "Weight %": feas_pct},
            {"Component": "Need", "Item": "Poverty rate", "Weight %": need_poverty},
            {"Component": "Need", "Item": "Median household income (inverse)", "Weight %": need_income},
            {"Component": "Need", "Item": "School count", "Weight %": need_schools},
            {"Component": "Feasibility", "Item": "Density", "Weight %": feas_density},
            {"Component": "Feasibility", "Item": "Distance to Ann Arbor", "Weight %": feas_distance},
        ]
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader("Top 10 ZIP codes")
    if len(scored) > 0:
        top10 = scored.head(10)
        display_cols = [
            "zip_code",
            "city",
            "state",
            "total_score",
            "need_score",
            "feasibility_score",
            "poverty_rate",
            "median_household_income",
            "school_count",
            "population",
            "density",
            "distance_to_ann_arbor_miles",
        ]
        display_cols = [c for c in display_cols if c in top10.columns]
        st.dataframe(top10[display_cols], use_container_width=True, hide_index=True)
        st.caption(
            f"Showing {len(scored)} geographies. Use **Ranked Geographies** for the full table and CSV export."
        )

# ----- Criteria Builder -----
elif page == "Criteria Builder":
    st.title("Criteria Builder")
    st.markdown(
        "Use the sidebar to adjust **Need vs Feasibility**, sub-weights, and hard filters. "
        "This page summarizes what is currently applied."
    )
    st.markdown("---")

    st.subheader("Active filters")
    st.write(f"- Max median household income: **${max_income:,}**")
    st.write(f"- Min poverty rate: **{min_poverty}%**")
    st.write(f"- Max distance to Ann Arbor: **{max_distance} miles**")
    st.write(f"- Min school count: **{min_schools}**")
    st.write(f"- State: **{state_filter or 'All'}**")

    st.markdown("---")
    st.subheader("Scoring direction")
    st.write("| Metric | Direction |")
    st.write("|--------|-----------|")
    st.write("| Poverty rate | Higher → higher need score |")
    st.write("| Median household income | Lower → higher need score |")
    st.write("| School count | Higher → higher need score |")
    st.write("| Density | Higher → higher feasibility score |")
    st.write("| Distance to Ann Arbor | Lower → higher feasibility score |")

# ----- Ranked Geographies -----
elif page == "Ranked Geographies":
    st.title("Ranked Geographies")
    st.caption("All geographies passing your filters, ordered by total score. Sort or search by ZIP / city.")
    st.markdown("---")

    if len(scored) == 0:
        st.warning("No geographies passed your filters. Loosen the filters in the sidebar.")
    else:
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
            "poverty_rate",
            "median_household_income",
            "school_count",
            "population",
            "density",
            "distance_to_ann_arbor_miles",
        ]
        display_df = ranked[[c for c in table_cols if c in ranked.columns]]

        column_config: dict[str, st.column_config.Column] = {}
        if "total_score" in display_df.columns:
            column_config["total_score"] = st.column_config.NumberColumn("Score", format="%.2f")
        if "need_score" in display_df.columns:
            column_config["need_score"] = st.column_config.NumberColumn("Need", format="%.2f")
        if "feasibility_score" in display_df.columns:
            column_config["feasibility_score"] = st.column_config.NumberColumn("Feasibility", format="%.2f")
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
        if "distance_to_ann_arbor_miles" in display_df.columns:
            column_config["distance_to_ann_arbor_miles"] = st.column_config.NumberColumn(
                "Distance (mi)", format="%.0f"
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
    st.caption("Choropleth by expansion score. Darker = higher score (better fit for expansion).")
    st.markdown("---")

    if len(scored) == 0:
        st.warning("No geographies to show. Loosen the filters in the sidebar.")
    else:
        try:
            m = build_choropleth(zip_geojson, scored, score_column="total_score", zip_column="zip_code")
            if m is None:
                st.warning("Map could not be drawn (folium or shapes not available in this environment).")
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

        school_list = schools_df[schools_df["zip_code"].astype(str) == selected_zip]
        if len(school_list) == 0:
            st.info(f"No schools found in ZIP **{selected_zip}** in the dataset.")
        else:
            st.subheader(f"Schools in ZIP {selected_zip}")
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
                school_map = build_school_map_only(school_list)
                if school_map is not None:
                    if hasattr(st, "folium_chart"):
                        st.folium_chart(school_map, use_container_width=True)
                    else:
                        import streamlit.components.v1 as components

                        components.html(school_map._repr_html_(), height=500, scrolling=False)
    else:
        st.info("Select a ZIP code above to see schools and download the list.")

