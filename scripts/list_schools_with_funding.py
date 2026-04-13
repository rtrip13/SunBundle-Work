#!/usr/bin/env python3
"""
List schools that have an athletics budget proxy and/or a booster-style EO BMF match.

Usage (from repo root):
  python scripts/list_schools_with_funding.py
  python scripts/list_schools_with_funding.py --boosters-only
  python scripts/list_schools_with_funding.py --proxy-only --limit 30
  python scripts/list_schools_with_funding.py --zip 44308
  python scripts/list_schools_with_funding.py --with-boosters   # slow on large EO extracts
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from utils.data_loader import (
    BMF_PATH,
    DATA_DIR,
    SLFS_PROXY_PATH,
    _discover_bmf_path,
    _discover_slfs_proxy_path,
    load_geographies,
    load_schools,
)


def _print_data_sources() -> None:
    print("=== Data files (funding signals) ===")
    print(f"Project data dir: {DATA_DIR}")
    nces = DATA_DIR / "nces"
    irs = DATA_DIR / "irs"
    print(f"  data/nces/ exists: {nces.is_dir()}")
    if nces.is_dir():
        csvs = sorted(nces.glob("*.csv"))
        print(f"    CSV files: {[c.name for c in csvs] or '(none)'}")
    print(f"  data/irs/ exists: {irs.is_dir()}")
    if irs.is_dir():
        csvs = sorted(irs.glob("*.csv"))
        print(f"    CSV files: {[c.name for c in csvs] or '(none)'}")
    slfs = _discover_slfs_proxy_path()
    bmf = _discover_bmf_path()
    print(f"  SLFS default path exists: {SLFS_PROXY_PATH.exists()}  →  {SLFS_PROXY_PATH}")
    print(f"  SLFS resolved for load: {slfs}")
    print(f"  BMF default path exists: {BMF_PATH.exists()}  →  {BMF_PATH}")
    print(f"  BMF resolved for load: {bmf}")
    if slfs is None:
        print("\n  → No SLFS file: athletics proxies will stay empty.")
        print("     Add a CSV under data/nces/ with columns: ncessch, athletics_budget_proxy")
    if bmf is None:
        print("\n  → No IRS EO/BMF file: booster matches will stay empty.")
        print("     Add a CSV under data/irs/ with at least: EIN, NAME, CITY, STATE, ZIP, REVENUE_AMT (or INCOME_AMT)")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Show schools with SLFS proxy and/or booster matches.")
    p.add_argument("--limit", type=int, default=40, help="Max rows to print per section")
    p.add_argument("--zip", type=str, default="", help="Filter to one 5-digit ZIP")
    p.add_argument(
        "--proxy-only",
        action="store_true",
        help="Only rows with direct NCES SLFS proxy (source NCES_SLFS_FY2022), not ZIP/global imputes",
    )
    p.add_argument("--boosters-only", action="store_true", help="Only rows with booster_exists or revenue > 0")
    p.add_argument(
        "--with-boosters",
        action="store_true",
        help="Run IRS EO/BMF matching (slow on large files). Default skips it and leaves booster columns empty/false.",
    )
    args = p.parse_args()

    _print_data_sources()

    # One school load; pass into geographies so we do not re-run booster matching inside load_geographies().
    schools = load_schools(None, match_boosters=args.with_boosters)
    geo = load_geographies(schools)
    df = schools.copy()
    df["zip_code"] = df["zip_code"].astype(str).str.strip().str.zfill(5)

    if args.zip:
        z = args.zip.strip().zfill(5)
        df = df[df["zip_code"] == z]
        print(f"Filtered to ZIP {z}: {len(df)} schools\n")

    has_proxy_col = "athletics_budget_proxy" in df.columns
    is_pub = df["is_public_school"].fillna(False).astype(bool) if "is_public_school" in df.columns else pd.Series(False, index=df.index)
    if has_proxy_col and "athletics_budget_proxy_source" in df.columns:
        src = df["athletics_budget_proxy_source"].astype(str).str.strip()
        proxy_num = pd.to_numeric(df["athletics_budget_proxy"], errors="coerce").fillna(0)
        has_slfs_direct = is_pub & (src == "NCES_SLFS_FY2022") & (proxy_num > 0)
    else:
        has_slfs_direct = pd.Series(False, index=df.index)
    has_public_proxy = (is_pub & df["athletics_budget_proxy"].notna()) if has_proxy_col else pd.Series(False, index=df.index)
    has_booster = pd.Series(False, index=df.index)
    if "booster_exists" in df.columns:
        has_booster = has_booster | df["booster_exists"].fillna(False).astype(bool)
    if "latest_booster_revenue" in df.columns:
        has_booster = has_booster | (df["latest_booster_revenue"].fillna(0) > 0)

    cols = [
        "zip_code",
        "city",
        "state",
        "school_name",
        "is_public_school",
        "athletics_budget_proxy",
        "athletics_budget_proxy_source",
        "booster_exists",
        "booster_match_confidence",
        "latest_booster_revenue",
        "matched_org_name",
        "athletics_budget_proxy_zip",
        "latest_booster_revenue_zip",
        "booster_exists_zip",
    ]
    cols = [c for c in cols if c in df.columns]

    print("=== Counts (all loaded schools) ===")
    print("Public schools with any athletics proxy ($, incl. imputes):", int(has_public_proxy.sum()))
    print("Public schools with direct SLFS proxy (> 0):", int(has_slfs_direct.sum()))
    print("With booster signal (exists or revenue > 0):", int(has_booster.sum()))
    print("With either (public proxy row or booster signal):", int((has_public_proxy | has_booster).sum()))
    print("With both:", int((has_public_proxy & has_booster).sum()))
    print()

    sub = df[has_public_proxy | has_booster]
    if args.proxy_only:
        sub = sub[has_slfs_direct.loc[sub.index]]
    if args.boosters_only:
        sub = sub[has_booster.loc[sub.index]]

    sub = sub.sort_values(
        by=[c for c in ("latest_booster_revenue", "athletics_budget_proxy") if c in sub.columns],
        ascending=False,
    )
    print(f"=== Sample (up to {args.limit} rows) ===")
    if len(sub) == 0:
        print("No rows match (see data-file section above).")
        if args.proxy_only:
            print("Tip: drop --proxy-only to list all public schools with a dollar proxy (incl. ZIP/global imputes).")
        return
    out_show = sub[cols].head(args.limit).copy()
    if "athletics_budget_proxy" in out_show.columns:
        out_show["athletics_budget_proxy_$"] = out_show["athletics_budget_proxy"].apply(
            lambda x: f"${float(x):,.0f}" if pd.notna(x) else "—"
        )
    if "athletics_budget_proxy_zip" in out_show.columns:
        out_show["athletics_budget_proxy_zip_$"] = out_show["athletics_budget_proxy_zip"].apply(
            lambda x: f"${float(x):,.0f}" if pd.notna(x) else "—"
        )
    print(out_show.to_string(index=False))


if __name__ == "__main__":
    main()
