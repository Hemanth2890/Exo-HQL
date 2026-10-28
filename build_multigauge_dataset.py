"""
Multi-gauge MDP dataset: per-gauge trajectories (fetch_multigauge.py) +
basin rainfall (fetch_rainfall.py) -> one table, multigauge_mdp_dataset.parquet.

Differences from build_mdp_dataset.py (single gauge):
  * stage features are NORMALIZED per gauge so one policy can transfer
    across basins: stage_norm = (stage - base) / (minor_flood - base),
    where base is the gauge's median stage over the TRAIN period only.
    stage_norm = 1 is minor flood stage at every gauge.
  * rainfall forcing: basin-mean accumulations over 1-72h, basin-max
    intensity, and an antecedent precipitation index (soil wetness proxy)
  * upstream gauge: normalized stage + rise rate of the NWPS-linked
    upstream forecast point, when that point is in the gauge set
  * split column: time split (TRAIN_END_DATE, as in train_baselines.py)
    crossed with a held-out river basin group (unseen-basin transfer)

Reward / lead-time / effective_severity columns reuse
build_mdp_dataset.add_lead_time_and_reward unchanged, so the exo-MDP
evaluator and every existing trainer read this table the same way.
"""

import os

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from build_mdp_dataset import add_lead_time_and_reward
from train_baselines import TRAIN_END_DATE
from fetch_multigauge import OUT_DIR

HELD_OUT_BASIN_GROUP = "nueces"
API_DECAY_PER_HOUR = 0.9 ** (1 / 24)  # classic daily k=0.9 antecedent precipitation index, hourly
EVENT_MIN_QUIET_HOURS = 24  # threshold re-crossings within a day are one storm, not separate events

MULTI_STATE_COLS = [
    # stage (normalized) and its dynamics
    "stage_norm", "stage_norm_lag1h", "stage_norm_lag3h", "stage_norm_lag6h",
    "stage_norm_delta_1h", "stage_norm_delta_6h", "stage_norm_delta_24h",
    "stage_norm_roll24h_max", "stage_norm_roll72h_max",
    "dist_to_moderate_norm", "dist_to_major_norm",
    "log_discharge", "log_discharge_delta_1h", "has_discharge",
    # rainfall forcing
    "rain_mean_1h", "rain_mean_3h", "rain_mean_6h", "rain_mean_12h", "rain_mean_24h", "rain_mean_72h",
    "rain_max_1h", "rain_max_3h", "rain_max_6h", "rain_api",
    # upstream gauge
    "up_stage_norm", "up_stage_norm_delta_1h", "up_stage_norm_delta_6h", "has_upstream",
    # time
    "hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos",
]


def basin_group(usgs_id: str) -> str:
    """Major river basin from the USGS downstream-order site number."""
    n = int(str(usgs_id)[:5])
    if n < 8160:
        return "colorado"
    if n < 8177:
        return "guadalupe"
    if n < 8190:
        return "san_antonio"
    return "nueces"


def stage_features(traj: pd.DataFrame, g: pd.Series) -> pd.DataFrame:
    train = traj["datetime"] < pd.Timestamp(TRAIN_END_DATE, tz="UTC")
    base = traj.loc[train, "gage_height_ft"].median()
    if np.isnan(base):
        base = traj["gage_height_ft"].median()
    scale = max(g["minor_ft"] - base, 1.0)

    f = pd.DataFrame(index=traj.index)
    s = (traj["gage_height_ft"] - base) / scale
    f["stage_norm"] = s
    for lag in (1, 3, 6):
        f[f"stage_norm_lag{lag}h"] = s.shift(lag)
    for d in (1, 6, 24):
        f[f"stage_norm_delta_{d}h"] = s.diff(d)
    f["stage_norm_roll24h_max"] = s.shift(1).rolling(24, min_periods=12).max()
    f["stage_norm_roll72h_max"] = s.shift(1).rolling(72, min_periods=36).max()
    for name in ("moderate", "major"):
        level = g[f"{name}_ft"]
        f[f"dist_to_{name}_norm"] = ((level - traj["gage_height_ft"]) / scale) if not np.isnan(level) else 10.0

    q = traj["discharge_cfs"]
    has_q = q.notna()
    logq = np.log1p(q.clip(lower=0))
    f["log_discharge"] = logq.fillna(0.0)
    f["log_discharge_delta_1h"] = logq.diff(1).fillna(0.0)
    f["has_discharge"] = has_q.astype(float)
    return f


def rain_features(traj: pd.DataFrame, rain: pd.DataFrame) -> pd.DataFrame:
    r = traj[["datetime"]].merge(rain, on="datetime", how="left")
    mean = r["rain_basin_mean_mm"].fillna(0.0)
    mx = r["rain_basin_max_mm"].fillna(0.0)
    f = pd.DataFrame(index=traj.index)
    for w in (1, 3, 6, 12, 24, 72):
        f[f"rain_mean_{w}h"] = mean.rolling(w, min_periods=1).sum().to_numpy()
    for w in (1, 3, 6):
        f[f"rain_max_{w}h"] = mx.rolling(w, min_periods=1).sum().to_numpy()
    # api[t] = k * api[t-1] + rain[t], as a first-order IIR filter
    f["rain_api"] = lfilter([1.0], [1.0, -API_DECAY_PER_HOUR], mean.to_numpy())
    return f


def build_gauge(g: pd.Series, stage_norm_by_lid: dict) -> pd.DataFrame:
    traj = pd.read_parquet(os.path.join(OUT_DIR, "traj", f"{g['lid']}.parquet"))
    traj = traj.sort_values("datetime").reset_index(drop=True)
    rain = pd.read_parquet(os.path.join(OUT_DIR, "rain", f"{g['lid']}.parquet"))

    df = pd.concat([traj, stage_features(traj, g), rain_features(traj, rain)], axis=1)

    up = g["upstream_lid"] if isinstance(g["upstream_lid"], str) else ""
    if up in stage_norm_by_lid:
        up_s = traj[["datetime"]].merge(stage_norm_by_lid[up], on="datetime", how="left")["stage_norm"]
        df["up_stage_norm"] = up_s.fillna(0.0).to_numpy()
        df["up_stage_norm_delta_1h"] = up_s.diff(1).fillna(0.0).to_numpy()
        df["up_stage_norm_delta_6h"] = up_s.diff(6).fillna(0.0).to_numpy()
        df["has_upstream"] = 1.0
    else:
        df[["up_stage_norm", "up_stage_norm_delta_1h", "up_stage_norm_delta_6h", "has_upstream"]] = 0.0

    dt = df["datetime"]
    df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    df["day_of_year_sin"] = np.sin(2 * np.pi * dt.dt.dayofyear / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear / 365.25)

    # reward/lead time on the continuous grid (sensor gaps -> severity 0),
    # THEN drop rows whose state is undefined
    df = add_lead_time_and_reward(df, min_quiet_hours=EVENT_MIN_QUIET_HOURS)
    df = df.dropna(subset=MULTI_STATE_COLS).reset_index(drop=True)

    df.insert(0, "lid", g["lid"])
    df["basin_group"] = basin_group(g["usgs_id"])
    is_eval_time = df["datetime"] >= pd.Timestamp(TRAIN_END_DATE, tz="UTC")
    is_eval_basin = df["basin_group"] == HELD_OUT_BASIN_GROUP
    df["split"] = np.select(
        [~is_eval_time & ~is_eval_basin, is_eval_time & ~is_eval_basin, ~is_eval_time & is_eval_basin],
        ["train", "eval_time", "eval_basin"], default="eval_both",
    )
    return df


if __name__ == "__main__":
    all_gauges = pd.read_csv(os.path.join(OUT_DIR, "gauges.csv"), dtype={"usgs_id": str})
    has_rain = all_gauges["lid"].map(lambda lid: os.path.exists(os.path.join(OUT_DIR, "rain", f"{lid}.parquet")))
    if not has_rain.all():
        print(f"Skipping {(~has_rain).sum()} gauge(s) without basin rainfall yet: "
              f"{', '.join(all_gauges.loc[~has_rain, 'lid'])}")
    gauges = all_gauges[has_rain].reset_index(drop=True)

    # first pass: normalized stage per gauge (all gauges, including ones
    # skipped above -- they can still be another gauge's upstream point)
    stage_norm_by_lid = {}
    for _, g in all_gauges.iterrows():
        traj = pd.read_parquet(os.path.join(OUT_DIR, "traj", f"{g['lid']}.parquet"))
        stage_norm_by_lid[g["lid"]] = pd.DataFrame(
            {"datetime": traj["datetime"], "stage_norm": stage_features(traj, g)["stage_norm"]})

    frames = []
    for _, g in gauges.iterrows():
        df = build_gauge(g, stage_norm_by_lid)
        frames.append(df)
        print(f"  {g['lid']:6s} rows={len(df):7d} events={int(df['event_start'].sum()):3d} "
              f"split={df['split'].iloc[0] if len(df) else '-'}..")
    data = pd.concat(frames, ignore_index=True)
    data.to_parquet("multigauge_mdp_dataset.parquet", index=False)

    print(f"\nSaved multigauge_mdp_dataset.parquet: {len(data)} rows, {data['lid'].nunique()} gauges")
    summary = data.groupby("split").agg(rows=("lid", "size"), gauges=("lid", "nunique"),
                                        events=("event_start", "sum"))
    print(summary)

    # sanity check that rainfall actually carries lead-time signal
    onsets = data[data["event_start"] == True]
    print(f"\nBasin rain in the 24h before onset (mm): median={onsets['rain_mean_24h'].median():.1f} "
          f"vs all-hours 99th pct={data['rain_mean_24h'].quantile(0.99):.1f}")
