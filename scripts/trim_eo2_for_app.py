#!/usr/bin/env python3
"""
Shrink an IRS EO/BMF-style CSV (e.g. eo2.csv) for this repo's booster matcher.

The app only needs a handful of columns (see ``_load_bmf`` in ``utils/data_loader.py``).
Dropping the rest plus optional row filters typically saves tens of MB on full extracts.

Usage (from repo root):
  python scripts/trim_eo2_for_app.py \\
    --input data/irs/eo2.csv --output data/irs/eo2_slim.csv

  # Keep only orgs in ZIPs that appear in your schools file (often much smaller):
  python scripts/trim_eo2_for_app.py --input data/irs/eo2.csv --output data/irs/eo2_slim.csv --school-zip-filter
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from utils.data_loader import (
    SCHOOLS_PATH,
    _bmf_column_lookup,
    _bmf_unrelated_to_k12_support_mask,
    _zip_to_string,
)


def _build_bmf_slim(df: pd.DataFrame) -> pd.DataFrame | None:
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
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Write a slimmer EO/BMF CSV for SunBundle booster matching.")
    p.add_argument("--input", type=Path, default=_ROOT / "data" / "irs" / "eo2.csv")
    p.add_argument("--output", type=Path, default=_ROOT / "data" / "irs" / "eo2_slim.csv")
    p.add_argument(
        "--school-zip-filter",
        action="store_true",
        help=f"Keep only rows whose ZIP appears in {SCHOOLS_PATH.name} (if that file exists).",
    )
    p.add_argument(
        "--no-unrelated-filter",
        action="store_true",
        help="Do not drop hospitals / halls of fame / etc. (not recommended).",
    )
    args = p.parse_args()

    inp = args.input.resolve()
    out_path = args.output.resolve()
    if not inp.is_file():
        sys.exit(f"Input not found: {inp}")

    print(f"Reading {inp} …")
    df = pd.read_csv(inp, dtype=str, low_memory=False, on_bad_lines="skip")
    in_bytes = inp.stat().st_size
    print(f"  rows={len(df):,} cols={len(df.columns)} size={in_bytes / 1e6:.1f} MB")

    slim = _build_bmf_slim(df)
    if slim is None:
        sys.exit("Could not map required BMF columns (EIN, NAME, CITY, STATE, ZIP).")

    slim = slim[slim["STATE"].astype(str).str.len() == 2]
    slim = slim[slim["ZIP"].astype(str).str.match(r"^\d{5}$", na=False)]

    if args.school_zip_filter and SCHOOLS_PATH.exists():
        sch = pd.read_csv(SCHOOLS_PATH, dtype=str, low_memory=False)
        if "zip_code" in sch.columns:
            zset = set(_zip_to_string(sch["zip_code"]).astype(str).str.zfill(5))
            before = len(slim)
            slim = slim[slim["ZIP"].astype(str).str.zfill(5).isin(zset)].copy()
            print(f"  school ZIP filter: {before:,} → {len(slim):,} rows ({len(zset):,} ZIPs)")
    elif args.school_zip_filter:
        print(f"  school ZIP filter skipped (missing {SCHOOLS_PATH})")

    if not args.no_unrelated_filter:
        bad = _bmf_unrelated_to_k12_support_mask(slim)
        before = len(slim)
        slim = slim.loc[~bad].copy()
        print(f"  unrelated-org filter: {before:,} → {len(slim):,} rows (dropped {int(bad.sum()):,})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    slim.to_csv(out_path, index=False)
    out_bytes = out_path.stat().st_size
    print(f"Wrote {out_path}")
    print(f"  rows={len(slim):,} cols={len(slim.columns)} size={out_bytes / 1e6:.1f} MB")
    print(f"  saved ~{(in_bytes - out_bytes) / 1e6:.1f} MB ({100.0 * (1 - out_bytes / in_bytes):.1f}% smaller)")


if __name__ == "__main__":
    main()
