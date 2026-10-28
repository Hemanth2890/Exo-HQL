"""
Evaluate all trained baselines (rule-based, BC, CQL, IQL, cost-calibrated
IQL) on the held-out eval split (2023 onward), using the operational
metrics from your plan: missed-event rate, false-alarm rate, warning
lead time -- not raw reward.

Requires: behavior_cloning.d3, discrete_cql.d3, discrete_iql.pt,
cost_calibrated_iql.pt, kerrville_mdp_dataset.csv all in the same folder.
"""

import numpy as np
import pandas as pd
import d3rlpy
from d3rlpy.algos import DiscreteBCConfig, DiscreteCQLConfig
from d3rlpy.dataset import MDPDataset

from train_baselines import STATE_COLS, build_episodes, time_split
from discrete_iql import DiscreteIQL, build_transitions
from build_mdp_dataset import COST_MATRIX, LEAD_BONUS_PER_HOUR


def load_d3rlpy_model(config_cls, path, obs_dim, action_size):
    """d3rlpy's save_model/load_model requires the architecture to be
    built first (same pattern used when we verified the uploaded .d3
    files earlier)."""
    dummy_obs = np.random.randn(10, obs_dim).astype(np.float32)
    dummy_actions = np.random.randint(0, action_size, size=10).astype(np.int64)
    dummy_rewards = np.random.randn(10).astype(np.float32)
    dummy_terminals = np.zeros(10, dtype=np.float32)
    dummy_terminals[-1] = 1.0
    dummy_ds = MDPDataset(
        observations=dummy_obs, actions=dummy_actions, rewards=dummy_rewards,
        terminals=dummy_terminals, action_space=d3rlpy.ActionSpace.DISCRETE,
    )
    model = config_cls().create(device=None)
    model.build_with_dataset(dummy_ds)
    model.load_model(path)
    return model


def rule_based_policy(states_df: pd.DataFrame) -> np.ndarray:
    """
    The existing NWS threshold system as a baseline: alert level purely
    from current gage height vs the official thresholds, ignoring trend,
    lags, and everything else your learned models can see.
    """
    ft = states_df["gage_height_ft"].to_numpy()
    actions = np.zeros(len(ft), dtype=np.int64)
    actions[ft >= 9.0] = 1   # minor flood -> Watch
    actions[ft >= 12.0] = 2  # moderate flood -> Warning
    actions[ft >= 20.0] = 2  # action space here tops out at 2 (see note below)
    return actions


def compute_metrics(df: pd.DataFrame, predicted_actions: np.ndarray, model_name: str) -> dict:
    """
    missed-event rate: fraction of flood-event onsets where the predicted
      action was 0 (no alert) at the moment the event began
    false-alarm rate: fraction of quiet days (severity 0) where the
      predicted action was > 0
    lead time: for caught events, how many HOURS before onset the
      predicted action first went active (capped at 336-hour / 14-day
      lookback, same convention as build_mdp_dataset.py)
    total/mean expected cost (static): COST_MATRIX[predicted_action,
      actual_severity] summed/averaged over every eval-set hour -- scores
      each hour's decision on its own, with no credit for how early an
      alert fired.
    total/mean expected cost (lead-credited): same, but a correct
      pre-alert within LOOKAHEAD_HOURS of a real event gets a small bonus
      instead of being charged as a plain false alarm -- this is the
      number that actually reflects what the reward function (and
      therefore training) optimizes, so it's the fair way to check
      whether a lead-time gain was worth its false-alarm cost.
    """
    df = df.copy().reset_index(drop=True)
    df["pred_action"] = predicted_actions

    event_idxs = df.index[df["event_start"] == True].tolist()
    n_events = len(event_idxs)
    n_missed = 0
    lead_times = []

    LOOKBACK_HOURS = 24 * 14

    for idx in event_idxs:
        if df.loc[idx, "pred_action"] == 0:
            n_missed += 1
            continue
        look_back_limit = max(0, idx - LOOKBACK_HOURS)
        j = idx - 1
        while j >= look_back_limit and df.loc[j, "pred_action"] != 0:
            j -= 1
        lead_times.append(idx - (j + 1))

    missed_rate = n_missed / n_events if n_events > 0 else float("nan")

    quiet_mask = df["severity_outcome"] == 0
    n_quiet = quiet_mask.sum()
    n_false_alarms = ((df["pred_action"] > 0) & quiet_mask).sum()
    false_alarm_rate = n_false_alarms / n_quiet if n_quiet > 0 else float("nan")

    mean_lead_time = np.mean(lead_times) if lead_times else float("nan")

    actual = df["severity_outcome"].fillna(0).to_numpy(dtype=np.int64)
    predicted = df["pred_action"].fillna(0).to_numpy(dtype=np.int64)
    step_costs = COST_MATRIX[predicted, actual]
    total_cost = float(step_costs.sum())
    mean_cost = float(step_costs.mean())

    # lead-credited version -- needs effective_severity, which
    # build_mdp_dataset.py now writes into kerrville_mdp_dataset.csv
    if "effective_severity" in df.columns:
        eff = df["effective_severity"].fillna(0).to_numpy(dtype=np.int64)
        lead_costs = step_costs.astype(float).copy()
        bonus_mask = (actual == 0) & (eff >= 1) & (predicted >= eff)
        lead_costs[bonus_mask] -= LEAD_BONUS_PER_HOUR
        total_cost_lead = float(lead_costs.sum())
        mean_cost_lead = float(lead_costs.mean())
    else:
        total_cost_lead = float("nan")
        mean_cost_lead = float("nan")

    return {
        "model": model_name,
        "n_events": n_events,
        "n_missed": n_missed,
        "missed_event_rate": missed_rate,
        "false_alarm_rate": false_alarm_rate,
        "mean_lead_time_hours": mean_lead_time,
        "n_caught": len(lead_times),
        "total_expected_cost_static": total_cost,
        "mean_expected_cost_per_hour_static": mean_cost,
        "total_expected_cost_leadcredit": total_cost_lead,
        "mean_expected_cost_per_hour_leadcredit": mean_cost_lead,
    }


if __name__ == "__main__":
    df = pd.read_csv("kerrville_mdp_dataset.csv")
    df = df.dropna(subset=STATE_COLS).reset_index(drop=True)
    train_df, eval_df = time_split(df)

    if len(eval_df) == 0:
        raise SystemExit("Eval split is empty -- check TRAIN_END_DATE in train_baselines.py "
                          "against your data's date range.")

    obs_dim = len(STATE_COLS)
    action_size = int(df["alert_action"].max()) + 1
    eval_states = eval_df[STATE_COLS].to_numpy(dtype=np.float32)

    results = []

    # --- rule-based (existing system) baseline ---
    rule_actions = rule_based_policy(eval_df)
    results.append(compute_metrics(eval_df, rule_actions, "rule_based_threshold"))

    # --- behavior cloning ---
    bc = load_d3rlpy_model(DiscreteBCConfig, "behavior_cloning.d3", obs_dim, action_size)
    bc_actions = bc.predict(eval_states)
    results.append(compute_metrics(eval_df, bc_actions, "behavior_cloning"))

    # --- discrete CQL ---
    cql = load_d3rlpy_model(DiscreteCQLConfig, "discrete_cql.d3", obs_dim, action_size)
    cql_actions = cql.predict(eval_states)
    results.append(compute_metrics(eval_df, cql_actions, "discrete_cql"))

    # --- discrete IQL ---
    iql = DiscreteIQL(obs_dim, action_size, device="cpu")
    iql.load("discrete_iql.pt")
    iql_actions = iql.predict(eval_states)
    results.append(compute_metrics(eval_df, iql_actions, "discrete_iql"))

    # --- cost-calibrated discrete IQL ---
    cc_iql = DiscreteIQL(obs_dim, action_size, device="cpu")
    cc_iql.load("cost_calibrated_iql.pt")
    cc_actions = cc_iql.predict(eval_states)
    results.append(compute_metrics(eval_df, cc_actions, "cost_calibrated_iql"))

    results_df = pd.DataFrame(results)
    pd.set_option("display.width", 120)
    print(results_df.to_string(index=False))
    results_df.to_csv("eval_results.csv", index=False)
    print("\nSaved eval_results.csv")