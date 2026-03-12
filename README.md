# SunBundle Expansion Decision Tool

MVP dashboard for deciding which ZIP codes or cities to expand into for school-based shoe donations. Built for consulting use: no auth, no database, local CSV/GeoJSON inputs.

## Setup

1. **Clone or download** this project.

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```
   Open the URL shown in the terminal (usually http://localhost:8501).

## Data

- **Without real data:** The app generates small dummy datasets automatically so you can run and demo it immediately.
- **With real data:** Place your files in the `data/` folder:
  - `data/geographies.csv` — one row per geography (ZIP/city) with columns such as:  
    `zip_code`, `city`, `state`, `latitude`, `longitude`, `median_income`, `poverty_rate`, `opportunity_score`, `distance_to_ann_arbor_miles`, `school_count`, `population`, `youth_population`
  - `data/schools.csv` — one row per school:  
    `school_name`, `district_name`, `address`, `city`, `state`, `zip_code`, `enrollment`, `grades`, `latitude`, `longitude`
  - `data/zip_shapes.geojson` — GeoJSON `FeatureCollection` where each feature has a property that can be matched to `zip_code` (e.g. `zip_code` or `ZCTA5CE10` for Census ZCTA). The app uses `feature.properties.zip_code` for the choropleth; if your file uses another key (e.g. `GEOID`), you can add a `zip_code` property in the loader or rename in the GeoJSON.

## Where to plug in real data

- **Opportunity Atlas / ACS / NCES:**  
  - **Geographies:** Build `geographies.csv` from Census (ACS) or Opportunity Atlas at the ZCTA or place level. Include the columns above; add or rename columns as needed and update `utils/data_loader.py` and `utils/scoring.py` if you add new criteria.
  - **Schools:** Export from NCES or your source to `schools.csv` with the expected columns. Ensure `zip_code` is string and matches the geography file for the School Finder.
  - **Shapes:** Use Census TIGER/Line ZCTA shapefiles, convert to GeoJSON, and save as `zip_shapes.geojson`. Ensure each feature has a property the app can use as ZIP (see above).

## Customization for your client

1. **Criteria and weights** — Sidebar sliders and the scoring logic in `utils/scoring.py` (including `SCORING_DIRECTION`). Add new metrics by adding columns to `geographies.csv`, new sliders in `app.py`, and entries in `weights` and `SCORING_DIRECTION`.
2. **Hard filters** — Sidebar number inputs and `utils/filters.py` (e.g. add state list, min population).
3. **Copy and labels** — Page titles, sidebar text, and the “How the score works” text in the Overview and Criteria Builder.
4. **Map** — Default center/zoom in `utils/mapping.py` (`DEFAULT_CENTER`, `DEFAULT_ZOOM`). To switch to pydeck, replace the folium calls in `utils/mapping.py` with pydeck and keep the same function signatures so `app.py` stays unchanged.
5. **Exports** — Column sets and file names in `utils/exports.py` and the download buttons in `app.py`.

## Project structure

```
app.py                 # Main Streamlit app
utils/
  data_loader.py       # Load CSVs/GeoJSON; dummy data
  scoring.py           # Weighted scoring and normalization
  filters.py           # Hard filters
  mapping.py           # Folium choropleth and school markers
  exports.py           # CSV export helpers
data/                  # Put geographies.csv, schools.csv, zip_shapes.geojson here
requirements.txt
README.md
```

## License / use

Internal prototype for nonprofit consulting. No warranty.
