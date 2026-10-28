"""
Episode construction, time-respecting train/eval split, and offline RL
training (DiscreteCQL, DiscreteBC baseline) on kerrville_mdp_dataset.csv.

HOURLY VERSION: rows are hours now, so EPISODE_MAX_LEN is expressed in hours
(720h = 30 days, same real-world cap as before).

Requires: pip install d3rlpy torch --break-system-packages
"""

import numpy as np
import pandas as pd
import d3rlpy
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteCQLConfig, DiscreteBCConfig

STATE_COLS = [
    "gage_height_ft", "discharge_cfs",
    "gage_height_lag1h", "gage_height_lag3h", "gage_height_lag6h",
    "discharge_lag1h", "discharge_lag3h", "discharge_lag6h",
    "gage_height_delta_1h", "gage_height_delta_6h", "gage_height_delta_24h",
    "discharge_delta_1h",
    "gage_height_roll6h_mean", "gage_height_roll24h_max", "gage_height_roll72h_max",
    "discharge_roll24h_mean",
    "dist_to_action_stage", "dist_to_minor_flood", "dist_to_major_flood",
    "hour_of_day", "month", "day_of_year_sin", "day_of_year_cos",
]

EPISODE_MAX_LEN = 24 * 30  # hours; also split on event boundaries (see build_episodes)
TRAIN_END_DATE = "2022-12-31"  # time-respecting split: train < this, eval >= this


def build_episodes(df: pd.DataFrame):
    """
    Split the continuous hourly record into episodes. An episode ends when:
      (a) a flood event resolves (severity returns to 0 after being >=1), or
      (b) EPISODE_MAX_LEN hours pass with no event, whichever comes first.
    This keeps quiet-period episodes bounded while keeping each flood event's
    full rise-and-recede arc inside a single episode.
    """
    df = df.sort_values("datetime").reset_index(drop=True)
    if len(df) == 0:
        return np.zeros(0, dtype=np.float32)

    terminals = np.zeros(len(df), dtype=np.float32)

    was_flooding = False
    steps_since_episode_start = 0
    for i in range(len(df)):
        is_flooding = df.loc[i, "severity_outcome"] >= 1
        steps_since_episode_start += 1

        episode_ends_here = False
        if was_flooding and not is_flooding:
            episode_ends_here = True  # event just resolved
        elif steps_since_episode_start >= EPISODE_MAX_LEN:
            episode_ends_here = True  # quiet-period cap

        if episode_ends_here:
            terminals[i] = 1.0
            steps_since_episode_start = 0

        was_flooding = is_flooding

    terminals[-1] = 1.0  # force-close the final episode
    return terminals


def build_mdp_dataset(df: pd.DataFrame) -> MDPDataset:
    df = df.sort_values("datetime").reset_index(drop=True)
    observations = df[STATE_COLS].to_numpy(dtype=np.float32)
    actions = df["alert_action"].to_numpy(dtype=np.int64)
    rewards = df["reward"].to_numpy(dtype=np.float32)
    terminals = build_episodes(df)

    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        action_space=d3rlpy.ActionSpace.DISCRETE,
    )
    return dataset


def time_split(df: pd.DataFrame):
    df = df.sort_values("datetime").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    train_df = df[df["datetime"] < TRAIN_END_DATE].reset_index(drop=True)
    eval_df = df[df["datetime"] >= TRAIN_END_DATE].reset_index(drop=True)
    print(f"Train: {len(train_df)} rows ({train_df['datetime'].min()} to "
          f"{train_df['datetime'].max()})")
    print(f"Eval:  {len(eval_df)} rows ({eval_df['datetime'].min()} to "
          f"{eval_df['datetime'].max()})")
    print(f"Train flood events: {train_df['event_start'].sum()}, "
          f"Eval flood events: {eval_df['event_start'].sum()}")
    return train_df, eval_df


def train_baselines(train_dataset: MDPDataset, n_steps: int = 10_000):
    bc = DiscreteBCConfig().create(device=None)  # None = auto-detect CPU/GPU
    bc.fit(train_dataset, n_steps=n_steps, n_steps_per_epoch=1000)

    cql = DiscreteCQLConfig().create(device=None)
    cql.fit(train_dataset, n_steps=n_steps, n_steps_per_epoch=1000)

    return {"behavior_cloning": bc, "discrete_cql": cql}


if __name__ == "__main__":
    df = pd.read_csv("kerrville_mdp_dataset.csv")

    # sanity check: no NaNs in state columns going into the model
    missing = df[STATE_COLS].isna().sum()
    if missing.sum() > 0:
        print("WARNING -- NaNs found in state columns:")
        print(missing[missing > 0])
        df = df.dropna(subset=STATE_COLS).reset_index(drop=True)
        print(f"Dropped to {len(df)} rows after removing NaNs")

    train_df, eval_df = time_split(df)

    train_dataset = build_mdp_dataset(train_df)
    eval_dataset = build_mdp_dataset(eval_df)

    print(f"Train episodes: {train_dataset.size()}")
    print(f"Eval episodes: {eval_dataset.size()}")

    models = train_baselines(train_dataset, n_steps=10_000)

    for name, model in models.items():
        model.save_model(f"{name}.d3")
        print(f"Saved {name}.d3")