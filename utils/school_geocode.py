"""
Fill missing school latitude/longitude using the U.S. Census geocoder (on-demand).

Results are cached under ``data/.cache/school_geocode.json`` so repeat views are fast
and we avoid hammering the public geocoder.
"""

from __future__ import annotations

import hashlib
import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_CACHE_PATH = _DATA_DIR / ".cache" / "school_geocode.json"
_USER_AGENT = "SunBundleExpansionTool/1.0 (school map geocoding; contact per Census/Nominatim policy)"


def _ssl_context() -> ssl.SSLContext | None:
    """Use certifi CA bundle when the platform store is incomplete (common on macOS Python)."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


def _one_line_address(row: pd.Series) -> str:
    street = str(row.get("address", "") or "").strip()
    city = str(row.get("city", "") or "").strip()
    state = str(row.get("state", "") or "").strip()
    z = str(row.get("zip_code", "") or "").strip()
    parts = [street, city]
    if state and z:
        parts.append(f"{state} {z}")
    elif state:
        parts.append(state)
    elif z:
        parts.append(z)
    return ", ".join(p for p in parts if p)


def _cache_key_for_row(row: pd.Series, address: str) -> str:
    ncessch = str(row.get("ncessch", "") or "").strip()
    if ncessch:
        return ncessch
    return "|addr|" + hashlib.sha256(address.upper().encode()).hexdigest()


def _load_cache() -> dict[str, Any]:
    try:
        with open(_CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError, TypeError):
        return {}


def _save_cache(cache: dict[str, Any]) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, separators=(",", ":"))
    except OSError:
        pass


def _census_geocode_oneline(address: str) -> tuple[float | None, float | None]:
    """Return (latitude, longitude) in WGS84-like decimal degrees from Census oneline geocoder."""
    if len(address.strip()) < 8:
        return None, None
    params = urllib.parse.urlencode(
        {
            "address": address,
            "benchmark": "Public_AR_Current",
            "format": "json",
        }
    )
    url = f"https://geocoding.geo.census.gov/geocoder/locations/onelineaddress?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    ctx = _ssl_context()
    try:
        open_kw = {"timeout": 25}
        if ctx is not None:
            open_kw["context"] = ctx
        with urllib.request.urlopen(req, **open_kw) as resp:
            payload = json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError, OSError):
        return None, None

    matches = (payload.get("result") or {}).get("addressMatches") or []
    if not matches:
        return None, None
    coords = matches[0].get("coordinates") or {}
    try:
        lon = float(coords["x"])
        lat = float(coords["y"])
    except (KeyError, TypeError, ValueError):
        return None, None
    return lat, lon


def ensure_school_coordinates(
    df: pd.DataFrame,
    *,
    zip_centroid: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """
    Return a copy of ``df`` with ``latitude`` / ``longitude`` filled when missing.

    Uses the Census onelineaddress API, then optional ``zip_centroid`` (from your
    geographies table) for rows that still have no coordinates.
    """
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    if "latitude" not in out.columns:
        out["latitude"] = np.nan
    if "longitude" not in out.columns:
        out["longitude"] = np.nan

    need = out["latitude"].isna() | out["longitude"].isna()
    if not need.any():
        return out

    cache = _load_cache()
    cache_dirty = False

    for i in out.loc[need].index:
        row = out.loc[i]
        addr = _one_line_address(row)
        if len(addr) < 10:
            continue
        key = _cache_key_for_row(row, addr)
        if key in cache:
            ent = cache[key]
            try:
                out.loc[i, "latitude"] = float(ent["lat"])
                out.loc[i, "longitude"] = float(ent["lon"])
            except (KeyError, TypeError, ValueError):
                pass
            continue

        lat, lon = _census_geocode_oneline(addr)
        time.sleep(0.06)
        if lat is not None and lon is not None:
            cache[key] = {"lat": lat, "lon": lon}
            out.loc[i, "latitude"] = lat
            out.loc[i, "longitude"] = lon
            cache_dirty = True

    if cache_dirty:
        _save_cache(cache)

    if zip_centroid is not None:
        still = out["latitude"].isna() | out["longitude"].isna()
        if still.any():
            out.loc[still, "latitude"] = zip_centroid[0]
            out.loc[still, "longitude"] = zip_centroid[1]

    return out
