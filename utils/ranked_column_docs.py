"""
Plain-language help for Ranked Geographies. Technical behavior matches utils/scoring.py and utils/data_loader.py.
"""

from __future__ import annotations

RANKED_COLUMN_HELP: dict[str, str] = {
    "Overview — how ranking works (start here)": """
### The big picture
Think of each ZIP as getting **three kinds of “grades”**, then blending them:

1. **Need** — How much the community looks like it could use support (poverty, income, how many schools are there to reach kids).
2. **Feasibility** — How practical it might be to work there (how dense the area is, how far it is from *your* chosen reference place).
3. **Funding signal** — A separate bundle of clues about **public-school spending proxy** (not “real” athletics line items) and **booster-style nonprofits** matched from IRS data — with a discount when we’re *unsure* the match is right.

Your **Need vs Feasibility** slider controls how much weight (1) and (2) get.  
Then we mix in a **smaller slice** (about **one fifth**) of (3) to form the **final score**.

### The one rule that confuses people
**Everything is compared only among ZIPs that survived your filters.**  
So “high poverty” here means “high *compared to the other ZIPs still on your list*,” not compared to the whole country. If you change filters, the same ZIP can move up or down.

### What “scaling” means (without equations)
For each number (poverty, income, distance, etc.), we stretch the **lowest** value in your list toward **0** and the **highest** toward **1** (sometimes flipping the direction — e.g. *lower* income counts as *more* need).  
If every ZIP looks the same on that measure, we don’t split hairs — everyone gets a neutral middle value.

### How the final score is mixed (conceptually)
- **Most** of the score comes from **Need + Feasibility** in the proportion you set.
- **A smaller part** comes from the **funding / booster** bundle.
- Inside that funding bundle, weak booster matches **reduce** that part a bit so bad guesses don’t swing the list too hard.

*Technical reference:* `utils/scoring.py` implements min–max scaling, weighted sums, a fixed funding mix, a confidence penalty, then `total ≈ 80% core + 20% funding`.
""",
    "rank": """
### What it is
Your **position in this table right now** — 1 is the top row after sorting by score.

### Why it changes
If you **search** the table, numbering restarts within what’s visible. It’s **not** a permanent “national rank.”
""",
    "zip_code": """
### What it is
The **ZIP code** we use to tie together schools, census-style stats, and map dots.

### Where it comes from
Your **geographies** file, with standard 5-digit formatting.

### Gotcha
ZIP and Census “ZCTA” boundaries don’t always match perfectly — occasional join quirks are normal.
""",
    "city": """
### What it is
The **city label** shown for that row (for reading the table, not for the heavy math).

### Where it comes from
Your **geographies** file.
""",
    "state": """
### What it is
The **state** for that geography row.

### Where it comes from
Your **geographies** file. The optional **state filter** in the sidebar uses this.
""",
    "total_score": """
### What it is
The **main ranking number** — higher = higher on this tool’s combined idea of need, feasibility, and (a bit) funding signals.

### How it’s built (intuitive)
- Start from **Need** and **Feasibility** scores and blend them using your **Need % / Feasibility %** slider.
- Add a **smaller contribution** from a **funding / booster** score (proxy spending + booster clues, with a **penalty** when matches look shaky).

### What it is *not*
Not dollars you can bank on, and not an official “athletics budget” — it’s a **model output** for comparison inside *your* filtered list.
""",
    "need_score": """
### What it is
A **0–1 style** score for “how much need shows up here” using **poverty**, **median income** (lower = more need), and **school count**.

### Intuition
You told the app how important each of those three is with the **Need** sliders. The app turns each raw column into a **0–1** style scale **among the ZIPs you’re looking at**, then mixes them.

### Why the list shifts when you change filters
Need scores are **relative**. Opening or tightening filters changes who’s in the pool, so the same ZIP’s need score can change even if its census numbers didn’t.
""",
    "feasibility_score": """
### What it is
A **0–1 style** score for “how workable this geography might be” using **density** and **distance from your reference point**.

### Intuition
- **Denser** can mean easier to reach more people in one place (depending on your slider weights).
- **Closer** to your reference location is treated as better for feasibility — using **straight-line miles**, not drive time.

### Limitation
Distance uses the **ZIP’s coordinates**, not individual school addresses, and it’s **not** traffic-aware.
""",
    "funding_signal_score": """
### What it is
A **0–1 style** score combining clues about **public-school spending proxy** (NCES SLFS — *not* “true athletics spend”), **whether a booster-like org matched**, **how much revenue showed up in IRS EO data**, and **how confident we are** about the match.

### Intuition
It’s a **“signals from money and organizations”** layer — useful for triage, not proof of booster or athletics line items.

### What reduces this score
If booster matches look **weak** (low confidence), we **shave down** this part so sketchy links don’t dominate.
""",
    "confidence_penalty": """
### What it is
A **0–1** number where **higher = we trust the booster match less** on average in that ZIP.

### Intuition
Think of it as “how fuzzy the nonprofit–school link feels.” When it’s high, the funding layer is **pulled back** a bit.

### How it’s made (simple)
We look at the **average match confidence** in the ZIP. Penalty is basically **“1 minus that confidence”** (capped so it stays sensible).
""",
    "poverty_rate": """
### What it is
**Community poverty rate (%)** for the geography, from Census-style data joined by ZIP.

### In the score
**Higher poverty → more “need” contribution** after scaling within your filtered list.

### Source
`data/acs/poverty.csv` (see `utils/data_loader.py`).
""",
    "median_household_income": """
### What it is
**Median household income** for the geography (Census-style), in dollars.

### In the score
**Lower income → more “need”** (we flip the direction so it lines up with “more need”).

### Source
`data/acs/income.csv`.
""",
    "school_count": """
### What it is
**How many schools** in your NCES extract share this ZIP (mailing/listing ZIP).

### In the score
More schools can mean more reach — **if** your sliders weight it that way.

### Limitation
It’s a **count of rows in your file**, not enrollment-weighted.
""",
    "population": """
### What it is
**Population** as given in your geographies file for that ZIP row.

### In the score
Shown for context; **not** a default driver of Need/Feasibility unless you add it to the model later.

### Source
`data/Geographies/geographies.csv`.
""",
    "density": """
### What it is
**Population density** as defined in your geographies file (whatever unit your file uses — be consistent across rows).

### In the score
**Higher density → higher feasibility contribution** (after scaling), if your Feasibility sliders say so.

### Gotcha
If density is missing or weirdly scaled in the file, rankings can look off — **keep one clean source**.
""",
    "distance_to_reference_miles": """
### What it is
**Straight-line miles** (“as the crow flies”) from this geography’s point to **your reference location** (city/state or ZIP you typed).

### In the score
**Closer is better** for feasibility — after scaling among filtered ZIPs.

### Limitation
Not driving distance, and based on **ZIP-level** coordinates — not each school’s front door.
""",
    "athletics_budget_proxy_zip": """
### What it is
A **ZIP-level average** of a **proxy** for public-school spending from **NCES SLFS** — **not** an official “athletics budget” and **not** exact per-athletics spending.

### Who gets a number
**Public** schools can carry a proxy; **private** schools are left out of this proxy by design.

### Where it comes from
Merged from `data/nces/slfs_fy2022_school_proxy.csv` by school ID, then **averaged** across public schools in the ZIP.

### Plain caveat
ZIPs with **only private** schools may show **no** proxy here — that’s expected, not a bug.
""",
    "booster_exists_zip": """
### What it is
**Yes/No at the ZIP level:** did **any** school in this ZIP get a “good enough” match to a nonprofit in the IRS EO BMF file?

### Intuition
**Yes** doesn’t mean “this is definitely the football booster” — it means **we found a plausible linked org** under our rules.

### How ZIP gets “Yes”
If **any** school in the ZIP crosses the match threshold, the whole ZIP can show **Yes** — so it can feel broad.
""",
    "booster_match_confidence_zip": """
### What it is
**How strong the nonprofit matches look on average** for schools in this ZIP — higher = we’re more comfortable with the link.

### What goes into “confidence” (plain English)
We compare **school name vs org name** (word overlap), and check **city / ZIP / state** lineups. It’s a **heuristic**, not a human verification.

### How we roll up to ZIP
We **average** school-level confidence in the ZIP — big schools and small schools count **equally** in that average today.

### Source
IRS **EO BMF** (`eo2`-style extract).
""",
    "latest_booster_revenue_zip": """
### What it is
**Largest single-school matched revenue** in the ZIP (`REVENUE_AMT` from EO BMF) — we take the **max** across schools so the ZIP figure is **not inflated** when many schools share one booster-style org.

### Intuition
Think “**rough size signal** from tax-exempt org filings,” not audited athletic income.

### Important caveat
The underlying match is still a **heuristic** (name / ZIP / keywords) — use it as a **clue**, not a fact.

### Source
IRS EO BMF–style extract (`data/irs/*.csv`); **not** full Form 990 line-by-line parsing in this MVP.
""",
    "latest_booster_net_assets_zip": """
### What it is
**Largest matched asset amount** in the ZIP (`ASSET_AMT`) — same **max** rollup idea as revenue so one shared org is not summed many times.

### May not appear in the main table
Sometimes kept in exports or future columns; the **logic** is the same as revenue: **helpful signal, not ground truth**.

### Caveat
Matches can still be **wrong or loose** for any given school — treat as triage, not verification.
""",
}
