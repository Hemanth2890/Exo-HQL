"""
Multi-gauge version of fetch_flood_dataset.py: every NWS river forecast
point in the Texas Hill Country ("Flash Flood Alley") that has a USGS gauge,
official NWS flood-stage categories, and at least one flood crest since the
USGS IV archive began. Turns the 5-event single-gauge problem into a
many-basin benchmark with held-out-basin evaluation.

Per gauge, writes multigauge/traj/<LID>.parquet on a continuous hourly
grid (gaps left as NaN so lags and rainfall stay time-aligned):
  datetime, gage_height_ft, discharge_cfs, severity_outcome, alert_action

and multigauge/gauges.csv with ids, location, thresholds and up/downstream
links.

Sources:
  NWPS  api.water.noaa.gov/nwps/v1 -- gauge list, flood categories, crests
  USGS  NWIS instantaneous values (gage height 00065, discharge 00060)
  IEM   VTEC point search -- logged NWS alerts (the behavior policy)

All raw responses are cached under multigauge/cache/ so reruns resume.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import dataretrieval.nwis as nwis

REGION_BBOX = (-100.6, 29.3, -97.8, 30.9)  # xmin, ymin, xmax, ymax -- Hill Country / Balcones escarpment
START = "2007-10-01"   # USGS IV archive begins Oct 2007 for most sites
END = "2026-09-16"
OUT_DIR = "multigauge"
CACHE_DIR = os.path.join(OUT_DIR, "cache")
N_WORKERS = 4

NWPS = "https://api.water.noaa.gov/nwps/v1"


def _get_json(url, cache_path, params=None, timeout=60):
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    last_err = None
    for attempt in range(1, 4):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(data, f)
            return data
        except Exception as e:
            last_err = e
            time.sleep(5 * attempt)
    raise RuntimeError(f"GET {url} failed after 3 attempts: {last_err}")


def _stage(categories, name):
    s = (categories.get(name) or {}).get("stage", -9999)
    return float(s) if s is not None and s > -999 else np.nan


def select_gauges() -> pd.DataFrame:
    xmin, ymin, xmax, ymax = REGION_BBOX
    listing = _get_json(
        f"{NWPS}/gauges", os.path.join(CACHE_DIR, "nwps_bbox.json"),
        params={"bbox.xmin": xmin, "bbox.ymin": ymin, "bbox.xmax": xmax,
                "bbox.ymax": ymax, "srid": "EPSG_4326"},
    )
    start_dt = pd.Timestamp(START, tz="UTC")
    rows = []
    for g in listing["gauges"]:
        lid = g["lid"]
        d = _get_json(f"{NWPS}/gauges/{lid}", os.path.join(CACHE_DIR, "nwps", f"{lid}.json"))
        flood = d.get("flood") or {}
        cats = flood.get("categories") or {}
        minor = _stage(cats, "minor")
        crest_lists = flood.get("crests") or {}
        crests = sorted({
            (pd.Timestamp(c["occurredTime"]), float(c["stage"]))
            for key in ("historic", "recent")
            for c in crest_lists.get(key) or []
            if c.get("occurredTime") and c.get("stage") is not None
        })
        recent_floods = [c for c in crests if c[0] >= start_dt and not np.isnan(minor) and c[1] >= minor]
        rows.append({
            "lid": lid, "usgs_id": d.get("usgsId") or "", "name": d.get("name"),
            "lat": d.get("latitude"), "lon": d.get("longitude"),
            "stage_units": flood.get("stageUnits"),
            "action_ft": _stage(cats, "action"), "minor_ft": minor,
            "moderate_ft": _stage(cats, "moderate"), "major_ft": _stage(cats, "major"),
            "upstream_lid": d.get("upstreamLid") or "", "downstream_lid": d.get("downstreamLid") or "",
            "n_floods_since_start": len(recent_floods),
            "crests": json.dumps([(t.isoformat(), s) for t, s in crests if t >= start_dt]),
        })
    gauges = pd.DataFrame(rows)
    keep = (
        (gauges["usgs_id"].str.fullmatch(r"\d{8,15}"))
        & gauges["minor_ft"].notna()
        & (gauges["stage_units"] == "ft")
        & (gauges["n_floods_since_start"] >= 1)
        # reservoir-level points report lake elevation (stages in the
        # hundreds of ft), not river stage -- a different hazard
        & (gauges["minor_ft"] < 200)
    )
    print(f"NWPS: {len(gauges)} gauges in region, {keep.sum()} usable "
          f"(USGS-linked river gauges, flood stages defined, >=1 flood crest since {START})")
    return gauges[keep].reset_index(drop=True)


def _fetch_iv_year(site: str, year: int) -> pd.DataFrame:
    cache_path = os.path.join(CACHE_DIR, "iv", site, f"{year}.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    start = max(pd.Timestamp(START), pd.Timestamp(year=year, month=1, day=1))
    end = min(pd.Timestamp(END), pd.Timestamp(year=year, month=12, day=31))
    last_err = None
    for attempt in range(1, 4):
        try:
            df, _ = nwis.get_iv(sites=site, start=start.strftime("%Y-%m-%d"),
                                end=end.strftime("%Y-%m-%d"), parameterCd=["00065", "00060"])
            df = df.reset_index()
            break
        except Exception as e:
            # dataretrieval raises on a year with no data at all -- that is a
            # valid empty year, not a failure worth retrying
            if "No sites/data found" in str(e) or "404" in str(e):
                df = pd.DataFrame()
                break
            last_err = e
            time.sleep(5 * attempt)
    else:
        raise RuntimeError(f"IV {site} {year} failed after 3 attempts: {last_err}")

    keep = [c for c in df.columns if c == "datetime" or (c[:5] in ("00065", "00060") and not c.endswith("_cd"))]
    df = df[keep] if not df.empty else df
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def fetch_all_iv(sites: list) -> dict:
    years = range(pd.Timestamp(START).year, pd.Timestamp(END).year + 1)
    tasks = [(s, y) for s in sites for y in years]
    frames = {s: [] for s in sites}
    with ThreadPoolExecutor(N_WORKERS) as pool:
        futures = {pool.submit(_fetch_iv_year, s, y): (s, y) for s, y in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            s, y = futures[fut]
            df = fut.result()
            if not df.empty:
                frames[s].append(df)
            if i % 50 == 0 or i == len(tasks):
                print(f"  IV: {i}/{len(tasks)} site-years fetched")
    return frames


def to_hourly(frames: list) -> pd.DataFrame:
    """Native ~15-min IV readings -> continuous hourly grid, max within each hour."""
    grid = pd.date_range(pd.Timestamp(START, tz="UTC"), pd.Timestamp(END, tz="UTC"), freq="h")
    if not frames:
        return pd.DataFrame({"datetime": grid})
    raw = pd.concat(frames, ignore_index=True)
    height_col = next((c for c in raw.columns if c.startswith("00065")), None)
    disch_col = next((c for c in raw.columns if c.startswith("00060")), None)
    raw["datetime"] = pd.to_datetime(raw["datetime"], utc=True)
    raw["hour"] = raw["datetime"].dt.floor("h")
    agg = {}
    if height_col:
        agg["gage_height_ft"] = (height_col, "max")
    if disch_col:
        agg["discharge_cfs"] = (disch_col, "max")
    hourly = raw.groupby("hour").agg(**agg)
    hourly = hourly.reindex(grid).rename_axis("datetime").reset_index()
    for col in ("gage_height_ft", "discharge_cfs"):
        if col not in hourly:
            hourly[col] = np.nan
    return hourly


def fetch_vtec(lid: str, lat: float, lon: float) -> pd.DataFrame:
    data = _get_json("https://mesonet.agron.iastate.edu/json/vtec_events_bypoint.py",
                     os.path.join(CACHE_DIR, "vtec", f"{lid}.json"),
                     params={"lat": lat, "lon": lon})
    df = pd.DataFrame(data.get("events", []))
    if df.empty:
        return df
    df = df[df["phenomena"].isin(["FA", "FF", "FL"])].copy()
    # river Flood Warnings (FL) are issued per forecast point: only this
    # gauge's own FL products count, not FLs for other points in the county
    df = df[(df["phenomena"] != "FL") | (df["hvtec_nwsli"] == lid)]
    df["issue_dt"] = pd.to_datetime(df["issue"], utc=True, errors="coerce")
    df["expire_dt"] = pd.to_datetime(df["expire"], utc=True, errors="coerce")
    start_dt = pd.Timestamp(START, tz="UTC")
    return df[df["issue_dt"] >= start_dt].dropna(subset=["issue_dt", "expire_dt"])


def build_trajectory(hourly: pd.DataFrame, vtec: pd.DataFrame, gauge: pd.Series) -> pd.DataFrame:
    """
    Same construction as fetch_flood_dataset.build_trajectory, generalized to
    per-gauge thresholds. Crest overrides come from the NWPS historic-crest
    list (exact times), which also covers hours where the sensor died at the
    peak -- the failure mode seen at Kerrville in July 2025.
    """
    traj = hourly.copy()
    thresholds = [
        (gauge["major_ft"], 3), (gauge["moderate_ft"], 2), (gauge["minor_ft"], 1),
    ]

    for t_iso, stage in json.loads(gauge["crests"]):
        t = pd.Timestamp(t_iso).tz_convert("UTC").floor("h")
        match = traj["datetime"] == t
        if match.any():
            sensor = traj.loc[match, "gage_height_ft"].iloc[0]
            if pd.isna(sensor) or stage > sensor:
                traj.loc[match, "gage_height_ft"] = stage

    ft = traj["gage_height_ft"].to_numpy()
    severity = np.where(np.isnan(ft), np.nan, 0.0)
    for level_ft, level in reversed(thresholds):
        if not np.isnan(level_ft):
            severity[ft >= level_ft] = level
    traj["severity_outcome"] = severity

    alert = np.zeros(len(traj), dtype=np.int64)
    if not vtec.empty:
        times = pd.DatetimeIndex(traj["datetime"])
        for _, ev in vtec.iterrows():
            lo = times.searchsorted(ev["issue_dt"])
            hi = times.searchsorted(ev["expire_dt"], side="right")
            level = 2 if ev["phenomena"] == "FF" else 1
            alert[lo:hi] = np.maximum(alert[lo:hi], level)
    traj["alert_action"] = alert
    return traj


if __name__ == "__main__":
    os.makedirs(os.path.join(OUT_DIR, "traj"), exist_ok=True)
    gauges = select_gauges()

    frames = fetch_all_iv(gauges["usgs_id"].tolist())

    kept = []
    for _, g in gauges.iterrows():
        hourly = to_hourly(frames[g["usgs_id"]])
        n_obs = hourly["gage_height_ft"].notna().sum()
        if n_obs < 24 * 365 * 3:
            print(f"  skip {g['lid']} ({g['name']}): only {n_obs} hourly stage readings")
            continue
        traj = build_trajectory(hourly, fetch_vtec(g["lid"], g["lat"], g["lon"]), g)
        traj.to_parquet(os.path.join(OUT_DIR, "traj", f"{g['lid']}.parquet"), index=False)
        sev = traj["severity_outcome"].fillna(0)
        n_events = int(((sev >= 1) & (sev.shift(1).fillna(0) < 1)).sum())
        print(f"  {g['lid']:6s} {g['name'][:40]:40s} stage_hours={n_obs:7d} "
              f"events={n_events:3d} alert_hours={(traj['alert_action'] > 0).sum():6d}")
        kept.append({**g.to_dict(), "n_stage_hours": int(n_obs), "n_events": n_events})

    pd.DataFrame(kept).to_csv(os.path.join(OUT_DIR, "gauges.csv"), index=False)
    print(f"Saved {len(kept)} gauge trajectories to {OUT_DIR}/traj/, metadata to {OUT_DIR}/gauges.csv")
