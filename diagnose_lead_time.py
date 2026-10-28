"""
Per-event lead-time breakdown for all 5 models -- to check whether the
0-day mean lead time for cost_calibrated_iql (from evaluate_models.py) is
a systematic pattern across all 3 events, or one event dragging the mean.
"""

import numpy as np
import pandas as pd
import d3rlpy
from d3rlpy.algos import DiscreteBCConfig, DiscreteCQLConfig

from train_baselines import STATE_COLS, time_split
from discrete_iql import DiscreteIQL
from evaluate_models import load_d3rlpy_model, rule_based_policy


def per_event_lead_times(df: pd.DataFrame, predicted_actions: np.ndarray, model_name: str):
    df = df.copy().reset_index(drop=True)
    df["pred_action"] = predicted_actions
    event_idxs = df.index[df["event_start"] == True].tolist()

    LOOKBACK_HOURS = 24 * 14

    print(f"\n--- {model_name} ---")
    for idx in event_idxs:
        event_date = df.loc[idx, "datetime"]
        if df.loc[idx, "pred_action"] == 0:
            print(f"  {event_date}: MISSED (no alert active at onset)")
            continue
        look_back_limit = max(0, idx - LOOKBACK_HOURS)
        j = idx - 1
        while j >= look_back_limit and df.loc[j, "pred_action"] != 0:
            j -= 1
        lead = idx - (j + 1)
        print(f"  {event_date}: caught, lead_time={lead} hours")


if __name__ == "__main__":
    df = pd.read_csv("kerrville_mdp_dataset.csv")
    df = df.dropna(subset=STATE_COLS).reset_index(drop=True)
    train_df, eval_df = time_split(df)

    obs_dim = len(STATE_COLS)
    action_size = int(df["alert_action"].max()) + 1
    eval_states = eval_df[STATE_COLS].to_numpy(dtype=np.float32)

    rule_actions = rule_based_policy(eval_df)
    per_event_lead_times(eval_df, rule_actions, "rule_based_threshold")

    bc = load_d3rlpy_model(DiscreteBCConfig, "behavior_cloning.d3", obs_dim, action_size)
    per_event_lead_times(eval_df, bc.predict(eval_states), "behavior_cloning")

    cql = load_d3rlpy_model(DiscreteCQLConfig, "discrete_cql.d3", obs_dim, action_size)
    per_event_lead_times(eval_df, cql.predict(eval_states), "discrete_cql")

    iql = DiscreteIQL(obs_dim, action_size, device="cpu")
    iql.load("discrete_iql.pt")
    per_event_lead_times(eval_df, iql.predict(eval_states), "discrete_iql")

    cc_iql = DiscreteIQL(obs_dim, action_size, device="cpu")
    cc_iql.load("cost_calibrated_iql.pt")
    per_event_lead_times(eval_df, cc_iql.predict(eval_states), "cost_calibrated_iql")