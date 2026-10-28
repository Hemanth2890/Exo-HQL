"""
Build the MDP-ready dataset from kerrville_flood_trajectory_clean.csv:
state features, action space, reward/cost structure, episode/step indexing.

HOURLY VERSION: each row is now one hour, not one day. Lags, rolling
windows and the lead-time lookback are all expressed in hours; see
LOOKBACK_HOURS below for the "how far back do we search for an already-
active alert" cap (kept at the same real-world ~2 weeks as the daily
version, just expressed in hours now).

Input columns expected: datetime, gage_height_ft, discharge_cfs,
                         severity_outcome, alert_action
"""

import numpy as np
import pandas as pd

THRESHOLDS_FT = {
    "action": 7.0,
    "minor_flood": 9.0,
    "moderate_flood": 12.0,
    "major_flood": 20.0,
}

# Ordinal severity cost matrix: rows = issued alert (a), cols = actual outcome (y)
# C[a][y] = cost of issuing alert level a when true severity was y
# 0=None, 1=Watch, 2=Warning, 3=Major/Severe
# Missing a severe event (a=0, y=3) is catastrophically worse than a false
# severe alarm (a=3, y=0) -- encodes the domain asymmetry discussed earlier.
COST_MATRIX = np.array([
    [0,   2,   8,  50],   # issued: None
    [1,   0,   3,  20],   # issued: Watch
    [3,   1,   0,   8],   # issued: Warning
    [10,  4,   1,   0],   # issued: Major/Severe
])

LOOKBACK_HOURS = 24 * 14  # ~2 weeks -- same real-world cap as the old 14-day version

# Lead-time credit: the static COST_MATRIX above only scores (action, SAME-HOUR
# outcome) -- it has no way to reward an alert for being early vs. just-in-time,
# since a correct alert costs 0 whichever hour it fires. That means earlier
# warnings only ever add false-alarm cost during the precursor hours, with no
# offsetting credit -- so a policy trained on this reward alone has no reason
# to warn any earlier than the last possible hour. LOOKAHEAD_HOURS/LEAD_BONUS
# fix that: any hour within LOOKAHEAD_HOURS of a real event, where the action
# already matches or exceeds the event's eventual severity, gets a small
# reward bonus instead of being charged as a plain false alarm. The bonus is
# deliberately small relative to COST_MATRIX (a fully sustained 72h early
# alert accumulates at most 72*0.15=10.8, well under the cost of missing a
# severe event) so the model is nudged toward earlier correct warnings without
# being able to profit from alerting constantly regardless of outcome.
LOOKAHEAD_HOURS = 72     # matches cost_calibrated_iql's default lookahead window
LEAD_BONUS_PER_HOUR = 0.15


def add_state_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("datetime").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # lags (hours ago)
    for lag in (1, 3, 6):
        df[f"gage_height_lag{lag}h"] = df["gage_height_ft"].shift(lag)
        df[f"discharge_lag{lag}h"] = df["discharge_cfs"].shift(lag)

    # rate of change at multiple time scales
    df["gage_height_delta_1h"] = df["gage_height_ft"].diff(1)
    df["gage_height_delta_6h"] = df["gage_height_ft"].diff(6)
    df["gage_height_delta_24h"] = df["gage_height_ft"].diff(24)
    df["discharge_delta_1h"] = df["discharge_cfs"].diff(1)

    # rolling stats (trailing, no leakage -- shift(1) before rolling)
    df["gage_height_roll6h_mean"] = df["gage_height_ft"].shift(1).rolling(6).mean()
    df["gage_height_roll24h_max"] = df["gage_height_ft"].shift(1).rolling(24).max()
    df["gage_height_roll72h_max"] = df["gage_height_ft"].shift(1).rolling(72).max()
    df["discharge_roll24h_mean"] = df["discharge_cfs"].shift(1).rolling(24).mean()

    # normalized distance to next threshold (state signal a rule-based system uses too)
    df["dist_to_action_stage"] = THRESHOLDS_FT["action"] - df["gage_height_ft"]
    df["dist_to_minor_flood"] = THRESHOLDS_FT["minor_flood"] - df["gage_height_ft"]
    df["dist_to_major_flood"] = THRESHOLDS_FT["major_flood"] - df["gage_height_ft"]

    # seasonality + diurnal encoding (now meaningful at hourly resolution)
    df["hour_of_day"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["datetime"].dt.dayofyear / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["datetime"].dt.dayofyear / 365.25)

    return df


def add_lead_time_and_reward(df: pd.DataFrame, min_quiet_hours: int = 1) -> pd.DataFrame:
    """
    Lead time: for each hour a severity>=1 event begins, how many HOURS did
    the alert_action precede it? Used both as an evaluation metric and to
    weight the cost matrix (earlier correct alerts should score better than
    late ones).

    min_quiet_hours: an onset only counts if severity was 0 for at least
    this many preceding hours, so a flashy stage re-crossing the threshold
    within one storm is one event, not several. Default 1 = original
    single-gauge behavior.
    """
    df = df.copy()
    flooding = (df["severity_outcome"].fillna(0) >= 1).astype(int)
    recent_flooding = flooding.shift(1).fillna(0).rolling(min_quiet_hours, min_periods=1).max()
    df["event_start"] = (flooding == 1) & (recent_flooding < 1)

    lead_times = np.full(len(df), np.nan)
    missed = np.zeros(len(df), dtype=bool)
    event_idxs = df.index[df["event_start"]].tolist()
    for idx in event_idxs:
        if df.loc[idx, "alert_action"] == 0:
            # no alert active at all when the event began -- a miss, not a
            # negative lead time. Leave lead_times[idx] as NaN.
            missed[idx] = True
            continue
        look_back_limit = max(0, idx - LOOKBACK_HOURS)
        j = idx - 1
        while j >= look_back_limit and df.loc[j, "alert_action"] != 0:
            j -= 1
        lead_times[idx] = idx - (j + 1)  # in HOURS, not days

    df["lead_time_hours"] = lead_times
    df["missed_event"] = missed

    # effective severity: max severity_outcome over the next LOOKAHEAD_HOURS
    # (inclusive of the current hour) -- same forward-looking max used for
    # the cost-calibrated tau, reused here so the reward itself can tell
    # "false alarm ahead of a real event" apart from "false alarm with
    # nothing coming."
    raw_severity = df["severity_outcome"].fillna(0).to_numpy(dtype=np.int64)
    n = len(raw_severity)
    effective_severity = raw_severity.copy()
    for i in range(n):
        window_end = min(n, i + LOOKAHEAD_HOURS + 1)
        effective_severity[i] = raw_severity[i:window_end].max()
    df["effective_severity"] = effective_severity

    # static per-row cost using the ordinal matrix (action vs SAME-HOUR
    # outcome only) -- kept as its own column since it's still the right
    # number for "how bad was this specific hour's decision," independent
    # of any lead-time framing.
    def lookup_cost(row):
        a = int(row["alert_action"]) if not pd.isna(row["alert_action"]) else 0
        y = int(row["severity_outcome"]) if not pd.isna(row["severity_outcome"]) else 0
        return COST_MATRIX[a, y]

    df["step_cost"] = df.apply(lookup_cost, axis=1)

    # lead-credited cost: same static cost, minus a small bonus on hours
    # where the action is already correctly pre-alerting for a real event
    # that's coming within LOOKAHEAD_HOURS. This is what reward/training
    # actually uses -- see the LOOKAHEAD_HOURS/LEAD_BONUS_PER_HOUR comment
    # above for why the static cost alone can't drive early warnings.
    def lookup_cost_leadcredit(row):
        a = int(row["alert_action"]) if not pd.isna(row["alert_action"]) else 0
        y = int(row["severity_outcome"]) if not pd.isna(row["severity_outcome"]) else 0
        y_eff = int(row["effective_severity"]) if not pd.isna(row["effective_severity"]) else 0
        base = COST_MATRIX[a, y]
        if y == 0 and y_eff >= 1 and a >= y_eff:
            return base - LEAD_BONUS_PER_HOUR
        return base

    df["step_cost_leadcredit"] = df.apply(lookup_cost_leadcredit, axis=1)
    df["reward"] = -df["step_cost_leadcredit"]  # RL convention: maximize reward = minimize cost

    return df


def build_mdp_dataset(path_in: str, path_out: str):
    df = pd.read_csv(path_in)
    df = add_state_features(df)
    df = add_lead_time_and_reward(df)

    # drop rows where lag/rolling features are NaN (first ~3 days of record)
    feature_cols = [c for c in df.columns if "lag" in c or "roll" in c or "delta" in c]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    df.to_csv(path_out, index=False)
    print(f"MDP dataset: {len(df)} rows -> {path_out}")
    print(f"Action distribution:\n{df['alert_action'].value_counts()}")
    print(f"Outcome severity distribution:\n{df['severity_outcome'].value_counts()}")
    print(f"Flood events (severity>=1 onsets): {df['event_start'].sum()}")
    print(f"Missed events (no alert active at onset): {df['missed_event'].sum()}")
    caught = df["lead_time_hours"].dropna()
    if len(caught):
        print(f"Mean lead time on caught events (hours): {caught.mean():.2f}")
    else:
        print("Mean lead time on caught events (hours): n/a (no caught events)")
    return df


if __name__ == "__main__":
    build_mdp_dataset(
        "kerrville_flood_trajectory_clean.csv",
        "kerrville_mdp_dataset.csv",
    )