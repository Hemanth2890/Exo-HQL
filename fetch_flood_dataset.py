"""
Fetch offline-RL trajectory data for flood early-warning decision support.
Gauge: Guadalupe River at Kerrville, TX (USGS 08166200 / NWS point KRRT2)

HOURLY VERSION: uses USGS's instantaneous-values (IV) service instead of the
daily-values (DV) service, resampled to hourly (max within each hour). This
gives ~20x more rows than the daily version over the same real 21-year
window, and -- more importantly -- lets flash floods that rise and fall
within a single day actually show up as a multi-hour rise instead of being
invisible at daily resolution.

Requires: pip install dataretrieval requests pandas --break-system-packages
Run locally -- this environment's sandbox cannot reach usgs.gov or
mesonet.agron.iastate.edu directly.
"""

import os
import time

import numpy as np
import pandas as pd
import requests
import dataretrieval.nwis as nwis

IV_CACHE_DIR = "iv_cache"  # per-year raw IV cache -- survives a mid-run network failure

SITE = "08166200"          # Guadalupe Rv at Kerrville, TX
NWS_POINT = "KRRT2"        # NWS river forecast point ID
START = "2005-01-01"       # IEM VTEC archive has no known holes since Nov 2005
END = "2026-09-16"

# NWS-published thresholds for this gauge (from IEM RVF product / NWS EWX)
THRESHOLDS_FT = {
    "action": 7.0,
    "minor_flood": 9.0,
    "moderate_flood": 12.0,
    "major_flood": 20.0,
}


def fetch_gauge_data():
    """
    Hourly gage height (ft) and discharge (cfs) from USGS NWIS instantaneous
    values (IV), resampled from native ~15-min readings to hourly by taking
    the max within each hour (preserves flash-flood peaks instead of
    averaging or sub-sampling them away).

    Fetched year-by-year rather than as one 21-year request -- IV queries
    return far more rows than DV queries and large single requests are more
    likely to be rejected or time out server-side. Each year's raw result is
    cached to disk (iv_cache/<year>.csv) so a transient network failure
    (TLS handshake timeout, connection reset, etc. -- USGS's service does
    this occasionally under load) only costs re-fetching that one year on
    the next run, not restarting from 2005. Each year gets up to 3 retries
    with a backoff before giving up.
    """
    os.makedirs(IV_CACHE_DIR, exist_ok=True)
    start_dt = pd.to_datetime(START)
    end_dt = pd.to_datetime(END)
    frames = []
    year = start_dt.year
    while year <= end_dt.year:
        cache_path = os.path.join(IV_CACHE_DIR, f"{year}.csv")
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            print(f"  {year}: {len(df)} raw IV readings (from cache)")
            if not df.empty:
                frames.append(df)
            year += 1
            continue

        year_start = max(start_dt, pd.Timestamp(year=year, month=1, day=1))
        year_end = min(end_dt, pd.Timestamp(year=year, month=12, day=31))

        df = None
        last_err = None
        for attempt in range(1, 4):
            try:
                df, meta = nwis.get_iv(
                    sites=SITE,
                    start=year_start.strftime("%Y-%m-%d"),
                    end=year_end.strftime("%Y-%m-%d"),
                    parameterCd=["00065", "00060"],  # 00065=gage height ft, 00060=discharge cfs
                )
                df = df.reset_index()
                break
            except Exception as e:
                last_err = e
                wait = 5 * attempt
                print(f"  {year}: attempt {attempt}/3 failed ({e}); retrying in {wait}s")
                time.sleep(wait)

        if df is None:
            raise RuntimeError(
                f"Failed to fetch {year} after 3 attempts: {last_err}\n"
                f"Years fetched so far are cached in {IV_CACHE_DIR}/ -- just "
                f"rerun this script and it will resume from {year} instead "
                f"of re-fetching everything."
            )

        df.to_csv(cache_path, index=False)
        print(f"  {year}: {len(df)} raw IV readings")
        if not df.empty:
            frames.append(df)
        year += 1

    if not frames:
        raise RuntimeError("No IV data fetched for any year -- check SITE/START/END.")
    raw = pd.concat(frames, ignore_index=True)

    # IV columns are named by parameter code directly (no statCd suffix --
    # these are raw instantaneous readings, not a daily statistic).
    height_col = next((c for c in raw.columns if c.startswith("00065") and not c.endswith("_cd")), None)
    disch_col = next((c for c in raw.columns if c.startswith("00060") and not c.endswith("_cd")), None)
    if height_col is None or disch_col is None:
        raise RuntimeError(
            f"Could not find gage height / discharge columns in IV data. "
            f"Available columns: {list(raw.columns)}"
        )

    raw["datetime"] = pd.to_datetime(raw["datetime"], utc=True)
    raw["hour"] = raw["datetime"].dt.floor("h")
    hourly = raw.groupby("hour").agg(
        gage_height_ft=(height_col, "max"),
        discharge_cfs=(disch_col, "max"),
    ).reset_index().rename(columns={"hour": "datetime"})

    print(f"NWIS IV: {len(raw)} raw readings -> {len(hourly)} hourly rows for site {SITE}")
    return hourly


def fetch_official_peaks():
    """
    USGS's curated annual peak-streamflow record for this site. During the
    most extreme events (e.g. July 2025), the routine sensor feed can fail
    or go silent exactly at the crest -- the July 4, 2025 Kerrville flood
    peaked at 34.29 ft per post-event field survey, while the routine
    sensor record understated it because the gauge itself stopped reporting
    reliably during the surge. Used to patch/override the hourly trajectory
    on known peak dates/times.
    """
    df, meta = nwis.get_discharge_peaks(sites=SITE, start=START, end=END)
    df = df.reset_index()
    print(f"NWIS: {len(df)} official peak-streamflow records fetched for site {SITE}")
    return df


def fetch_vtec_warnings():
    """
    NWS VTEC flood-related products (FA.A/FA.W/FF.W = Flood Watch/Warning/
    Flash Flood Warning) intersecting the Kerrville forecast point, via IEM's
    point-search endpoint. Issue/expire times are kept as full timestamps
    (not truncated to date) so an hourly trajectory can mark exactly which
    hours had an active warning.
    """
    lat, lon = 30.047, -99.140

    url = "https://mesonet.agron.iastate.edu/json/vtec_events_bypoint.py"
    params = {"lat": lat, "lon": lon}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    events = r.json().get("events", [])
    df = pd.DataFrame(events)

    df = df[df["phenomena"].isin(["FA", "FF", "FL"])]  # Flood/FlashFlood/RiverFlood
    df["issue_dt"] = pd.to_datetime(df["issue"], utc=True, errors="coerce")
    df["expire_dt"] = pd.to_datetime(df["expire"], utc=True, errors="coerce")

    start_dt = pd.to_datetime(START, utc=True)
    end_dt = pd.to_datetime(END, utc=True) + pd.Timedelta(days=1)
    df = df[(df["issue_dt"] >= start_dt) & (df["issue_dt"] <= end_dt)]

    # FL (river Flood Warning) events carry hvtec_nwsli = exact NWS gauge ID
    df["gauge_specific"] = df["hvtec_nwsli"] == NWS_POINT
    print(f"IEM VTEC: {len(df)} flood-related warning/watch events fetched")
    return df


def build_trajectory(gauge_df, vtec_df, peaks_df=None):
    """
    Merge into an hourly state-action-outcome table:
      state  = gage height, discharge (+ engineered lags/deltas downstream)
      action = NWS alert level in effect that hour (derived from VTEC events)
      outcome = whether stage crossed each threshold that hour
    """
    daily = gauge_df.sort_values("datetime").reset_index(drop=True)
    daily["datetime"] = pd.to_datetime(daily["datetime"], utc=True)

    def severity(ft):
        if pd.isna(ft):
            return None
        if ft >= THRESHOLDS_FT["major_flood"]:
            return 3
        if ft >= THRESHOLDS_FT["moderate_flood"]:
            return 2
        if ft >= THRESHOLDS_FT["minor_flood"]:
            return 1
        return 0

    daily["severity_outcome"] = daily["gage_height_ft"].apply(severity)

    # Override with official peak record on known peak dates -- the routine
    # sensor feed can fail/go silent exactly at the crest of the most
    # extreme events (confirmed for July 2025 at this site). Uses peak_tm
    # (time of day) when available; otherwise overrides that day's
    # currently-highest recorded hour as the best available proxy for the
    # crest time.
    if peaks_df is not None and not peaks_df.empty:
        peaks_df = peaks_df.copy()
        peaks_df["peak_date"] = pd.to_datetime(peaks_df["datetime"], errors="coerce", utc=True).dt.date
        peaks_df = peaks_df.dropna(subset=["peak_date", "gage_ht"])

        n_overridden = 0
        for _, pk in peaks_df.iterrows():
            target_ts = None
            peak_tm = pk.get("peak_tm")
            if isinstance(peak_tm, str) and ":" in peak_tm:
                try:
                    h = int(peak_tm.strip().split(":")[0])
                    target_ts = pd.Timestamp(pk["peak_date"], tz="UTC") + pd.Timedelta(hours=h)
                except Exception:
                    target_ts = None

            if target_ts is not None and (daily["datetime"] == target_ts).any():
                match = daily["datetime"] == target_ts
            else:
                day_mask = daily["datetime"].dt.date == pk["peak_date"]
                if not day_mask.any():
                    continue
                best_idx = daily.loc[day_mask, "gage_height_ft"].idxmax()
                match = daily.index == best_idx

            official_ft = pk["gage_ht"]
            sensor_ft = daily.loc[match, "gage_height_ft"].iloc[0]
            if pd.isna(sensor_ft) or official_ft > sensor_ft:
                daily.loc[match, "gage_height_ft"] = official_ft
                daily.loc[match, "severity_outcome"] = severity(official_ft)
                n_overridden += 1
        print(f"Overrode {n_overridden} hour(s) with official peak-streamflow values "
              f"(sensor record understated the true crest)")

    # Action: mark each hour covered by an active VTEC flood event
    daily["alert_action"] = 0
    if vtec_df is not None and not vtec_df.empty:
        for _, ev in vtec_df.iterrows():
            mask = (daily["datetime"] >= ev["issue_dt"]) & (daily["datetime"] <= ev["expire_dt"])
            phenomena_severity = 2 if ev.get("phenomena") == "FF" else 1
            daily.loc[mask, "alert_action"] = daily.loc[mask, "alert_action"].clip(lower=phenomena_severity)

    return daily.sort_values("datetime").reset_index(drop=True)


if __name__ == "__main__":
    gauge_df = fetch_gauge_data()
    vtec_df = fetch_vtec_warnings()
    peaks_df = fetch_official_peaks()
    trajectory = build_trajectory(gauge_df, vtec_df, peaks_df)
    trajectory.to_csv("kerrville_flood_trajectory.csv", index=False)
    print(f"Saved {len(trajectory)} rows to kerrville_flood_trajectory.csv")

    # Trim rows with no gage-height reading (sensor gaps / startup period)
    n_before = len(trajectory)
    clean = trajectory.dropna(subset=["gage_height_ft"]).reset_index(drop=True)
    n_after = len(clean)
    print(f"Trimmed {n_before - n_after} row(s) missing gage height "
          f"({n_after} rows after trimming)")
    clean.to_csv("kerrville_flood_trajectory_clean.csv", index=False)
    print(f"Saved {n_after} rows to kerrville_flood_trajectory_clean.csv")
    print(clean.head())