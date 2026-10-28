"""
Basin-averaged hourly rainfall forcing for every gauge in multigauge/gauges.csv.

Why: stage/discharge alone cannot see a flood coming until the water is
already rising, so a gauge-only policy has (almost) no lead-time signal --
which is exactly what the hindsight oracle in hindsight_oracle.py showed.
Rain falling on the upstream basin is the causal precursor.

Pipeline:
  1. USGS NLDI -> upstream drainage-basin polygon for each gauge
  2. sample ERA5-Land 0.1 deg grid cells inside the basin (stratified by
     distance to the gauge for very large basins, capped at
     MAX_CELLS_PER_BASIN); tiny basins fall back to the gauge's own cell
  3. Open-Meteo historical archive -> hourly precipitation per unique cell
     (cells shared by nested basins are fetched once)
  4. per gauge: basin-mean and basin-max hourly precipitation (mm)

Writes multigauge/rain/<LID>.parquet: datetime, rain_basin_mean_mm,
rain_basin_max_mm, n_rain_cells.

Caveat for the paper: ERA5-Land is a reanalysis, not what a forecaster
sees live. Its operational analogue is radar QPE (MRMS), which has ~1h
latency, so hour-t rainfall is treated as observable at hour t. Open-Meteo
timestamps precipitation as the sum over the PRECEDING hour, so there is
no look-ahead in using it at t.

Open-Meteo's free tier is rate-limited (long multi-year requests count as
many calls). Every cell is cached; on HTTP 429 the script stops cleanly
and a rerun resumes where it left off.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests
from matplotlib.path import Path

from fetch_multigauge import START, END, OUT_DIR, CACHE_DIR, _get_json

GRID_DEG = 0.1             # ERA5-Land native resolution
MAX_CELLS_PER_BASIN = 12
BATCH = 8                  # locations per Open-Meteo request
NLDI = "https://api.water.usgs.gov/nldi/linked-data/nwissite"
OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"


class RateLimited(Exception):
    pass


def basin_polygons(usgs_id: str) -> list:
    data = _get_json(f"{NLDI}/USGS-{usgs_id}/basin", os.path.join(CACHE_DIR, "nldi", f"{usgs_id}.json"),
                     params={"simplified": "true", "splitCatchment": "false"}, timeout=120)
    rings = []
    for feat in data.get("features", []):
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        rings += [np.asarray(p[0]) for p in polys]   # outer rings only
    return rings


def basin_cells(rings: list, gauge_lat: float, gauge_lon: float) -> list:
    """ERA5-Land cell centres (lat, lon) inside the basin, stratified-capped."""
    cells = []
    for ring in rings:
        path = Path(ring)
        xmin, ymin = ring.min(axis=0)
        xmax, ymax = ring.max(axis=0)
        lons = np.arange(np.floor(xmin / GRID_DEG), np.ceil(xmax / GRID_DEG) + 1) * GRID_DEG
        lats = np.arange(np.floor(ymin / GRID_DEG), np.ceil(ymax / GRID_DEG) + 1) * GRID_DEG
        pts = np.array([(x, y) for x in lons for y in lats])
        if len(pts):
            cells += [tuple(p) for p in pts[path.contains_points(pts)]]
    snap = lambda v: round(round(v / GRID_DEG) * GRID_DEG, 1)
    cells = sorted({(snap(lat), snap(lon)) for lon, lat in cells})
    if not cells:
        return [(snap(gauge_lat), snap(gauge_lon))]
    if len(cells) > MAX_CELLS_PER_BASIN:
        # evenly spaced through the distance-sorted list: keeps both the
        # near-gauge cells (flash response) and the far headwaters
        d = [np.hypot(lat - gauge_lat, lon - gauge_lon) for lat, lon in cells]
        order = np.argsort(d)
        pick = np.linspace(0, len(cells) - 1, MAX_CELLS_PER_BASIN).round().astype(int)
        cells = [cells[order[i]] for i in pick]
    return cells


def _cell_path(cell):
    return os.path.join(CACHE_DIR, "rain", f"{cell[0]:.1f}_{cell[1]:.1f}.parquet")


def fetch_cells(cells: list):
    todo = [c for c in cells if not os.path.exists(_cell_path(c))]
    print(f"Rain cells: {len(cells)} unique, {len(cells) - len(todo)} cached, {len(todo)} to fetch")
    os.makedirs(os.path.join(CACHE_DIR, "rain"), exist_ok=True)
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        params = {
            "latitude": ",".join(f"{c[0]:.1f}" for c in batch),
            "longitude": ",".join(f"{c[1]:.1f}" for c in batch),
            "start_date": START, "end_date": END,
            "hourly": "precipitation", "models": "era5_land", "timezone": "GMT",
        }
        attempt = 0
        while True:
            r = requests.get(OPEN_METEO, params=params, timeout=300)
            if r.status_code == 429:
                # minutely/hourly limits clear on their own -- wait them out;
                # the daily limit does not, so stop and let a rerun resume
                if "daily" in r.text.lower():
                    raise RateLimited(r.text[:200])
                wait = 65 if "minutely" in r.text.lower() else 600
                print(f"  rate limited ({r.text[:120]}); waiting {wait}s")
                time.sleep(wait)
                continue
            attempt += 1
            if r.ok or attempt >= 3:
                break
            time.sleep(10 * attempt)
        r.raise_for_status()
        results = r.json()
        results = results if isinstance(results, list) else [results]
        for cell, res in zip(batch, results):
            df = pd.DataFrame({
                "datetime": pd.to_datetime(res["hourly"]["time"], utc=True),
                "precip_mm": np.asarray(res["hourly"]["precipitation"], dtype=float),
            })
            df.to_parquet(_cell_path(cell), index=False)
        print(f"  fetched {min(i + BATCH, len(todo))}/{len(todo)} cells")
        time.sleep(2)


if __name__ == "__main__":
    gauges = pd.read_csv(os.path.join(OUT_DIR, "gauges.csv"), dtype={"usgs_id": str})
    os.makedirs(os.path.join(OUT_DIR, "rain"), exist_ok=True)

    cells_by_gauge = {}
    for _, g in gauges.iterrows():
        rings = basin_polygons(g["usgs_id"])
        cells_by_gauge[g["lid"]] = basin_cells(rings, g["lat"], g["lon"])
        print(f"  {g['lid']:6s} basin rings={len(rings)} cells={len(cells_by_gauge[g['lid']])}")

    # --cached-only: skip fetching and write rainfall only for gauges whose
    # basin cells are all already cached (a subset to work with while the
    # rate-limited fetch catches up)
    if "--cached-only" in sys.argv:
        cells_by_gauge = {lid: cells for lid, cells in cells_by_gauge.items()
                          if all(os.path.exists(_cell_path(c)) for c in cells)}
        print(f"--cached-only: {len(cells_by_gauge)}/{len(gauges)} gauges fully covered")
    else:
        unique = sorted({c for cells in cells_by_gauge.values() for c in cells})
        try:
            fetch_cells(unique)
        except RateLimited as e:
            raise SystemExit(f"Open-Meteo rate limit hit ({e}). Progress is cached -- rerun later to resume.")

    for lid, cells in cells_by_gauge.items():
        series = [pd.read_parquet(_cell_path(c)).set_index("datetime")["precip_mm"] for c in cells]
        mat = pd.concat(series, axis=1)
        out = pd.DataFrame({
            "rain_basin_mean_mm": mat.mean(axis=1),
            "rain_basin_max_mm": mat.max(axis=1),
            "n_rain_cells": len(cells),
        }).rename_axis("datetime").reset_index()
        out.to_parquet(os.path.join(OUT_DIR, "rain", f"{lid}.parquet"), index=False)
    with open(os.path.join(OUT_DIR, "rain_cells.json"), "w") as f:
        json.dump({k: v for k, v in cells_by_gauge.items()}, f)
    print(f"Saved basin rainfall for {len(cells_by_gauge)} gauges to {OUT_DIR}/rain/")
