"""
Exo-MDP evaluation of every policy on the held-out eval split:
exact counterfactual replay cost, cost breakdown, per-event lead time, and
regret to the hindsight-optimal oracle. Also scores the LOGGED NWS alerts
(the behavior policy) the same way.

Ends with a sensitivity sweep over the two most arbitrary cost-model knobs
(onset-cost scale, switching cost) to check whether the policy ranking is
robust to them -- the exo cost model is a design choice and should be
reported that way.

Requires the same model files as evaluate_models.py.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
from d3rlpy.algos import DiscreteBCConfig, DiscreteCQLConfig

from train_baselines import STATE_COLS, time_split
from discrete_iql import DiscreteIQL
from evaluate_models import load_d3rlpy_model, rule_based_policy
from exo_mdp import ExoConfig, COST_KEYS, build_exo_trace, replay
from hindsight_oracle import solve_oracle


def rule_based_4level(states_df: pd.DataFrame) -> np.ndarray:
    """Same thresholds as rule_based_policy, but uses the Major/Severe level."""
    actions = rule_based_policy(states_df)
    actions[states_df["gage_height_ft"].to_numpy() >= 20.0] = 3
    return actions


def rule_action_stage(states_df: pd.DataFrame) -> np.ndarray:
    """Stronger rule: Watch already at NWS action stage (7 ft), then 4-level."""
    actions = rule_based_4level(states_df)
    ft = states_df["gage_height_ft"].to_numpy()
    actions[(ft >= 7.0) & (actions == 0)] = 1
    return actions


def collect_policies(eval_df: pd.DataFrame, action_size: int) -> dict:
    obs_dim = len(STATE_COLS)
    eval_states = eval_df[STATE_COLS].to_numpy(dtype=np.float32)
    policies = {
        "never_alert": np.zeros(len(eval_df), dtype=np.int64),
        "logged_nws": eval_df["alert_action"].to_numpy(dtype=np.int64),
        "rule_based_threshold": rule_based_policy(eval_df),
        "rule_based_4level": rule_based_4level(eval_df),
        "rule_action_stage": rule_action_stage(eval_df),
    }
    bc = load_d3rlpy_model(DiscreteBCConfig, "behavior_cloning.d3", obs_dim, action_size)
    policies["behavior_cloning"] = bc.predict(eval_states)
    cql = load_d3rlpy_model(DiscreteCQLConfig, "discrete_cql.d3", obs_dim, action_size)
    policies["discrete_cql"] = cql.predict(eval_states)
    for name, path in [("discrete_iql", "discrete_iql.pt"),
                       ("cost_calibrated_iql", "cost_calibrated_iql.pt")]:
        model = DiscreteIQL(obs_dim, action_size, device="cpu")
        model.load(path)
        policies[name] = model.predict(eval_states)
    return policies


def evaluate_all(policies: dict, trace: dict, cfg: ExoConfig):
    oracle_actions, oracle_cost = solve_oracle(trace, cfg)
    oracle_replay = replay(oracle_actions, trace, cfg)
    # the DP value and an independent forward replay must agree exactly
    assert np.isclose(oracle_replay["total"], oracle_cost), (oracle_replay["total"], oracle_cost)

    results = {"hindsight_oracle": oracle_replay}
    for name, actions in policies.items():
        results[name] = replay(actions, trace, cfg)
    return results, oracle_actions


def results_table(results: dict) -> pd.DataFrame:
    oracle_total = results["hindsight_oracle"]["total"]
    rows = []
    for name, r in results.items():
        leads = [e["lead_hours"] for e in r["events"]]
        rows.append({
            "model": name,
            "exo_cost": r["total"],
            "regret_to_oracle": r["total"] - oracle_total,
            **{key: r[key] for key in COST_KEYS},
            "mean_lead_hours": float(np.mean(leads)) if leads else float("nan"),
            "per_event_lead_hours": "/".join(str(l) for l in leads),
            "final_debt": r["final_debt"],
        })
    return pd.DataFrame(rows).sort_values("exo_cost").reset_index(drop=True)


def sensitivity_sweep(policies: dict, trace: dict, base: ExoConfig,
                      onset_scales=(0.5, 1.0, 2.0), switch_costs=(0.0, 0.5, 2.0)) -> pd.DataFrame:
    rows = []
    for scale in onset_scales:
        for sw in switch_costs:
            cfg = replace(base, onset_cost=tuple(c * scale for c in base.onset_cost), switch_cost=sw)
            results, _ = evaluate_all(policies, trace, cfg)
            oracle_total = results["hindsight_oracle"]["total"]
            row = {"onset_scale": scale, "switch_cost": sw, "oracle_cost": oracle_total}
            for name in policies:
                row[name] = results[name]["total"] - oracle_total
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv("kerrville_mdp_dataset.csv")
    df = df.dropna(subset=STATE_COLS).reset_index(drop=True)
    train_df, eval_df = time_split(df)
    action_size = int(df["alert_action"].max()) + 1

    cfg = ExoConfig()
    trace = build_exo_trace(eval_df, cfg)
    policies = collect_policies(eval_df, action_size)

    results, oracle_actions = evaluate_all(policies, trace, cfg)
    table = results_table(results)
    pd.set_option("display.width", 200)
    print("\n=== Exo-MDP replay on eval split (lower cost is better) ===")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    table.to_csv("exo_eval_results.csv", index=False)

    print("\n=== Per-event detail: logged NWS vs hindsight oracle ===")
    for name in ("logged_nws", "hindsight_oracle"):
        for e in results[name]["events"]:
            print(f"  {name:18s} {e['datetime']}  peak={e['peak_severity']}  "
                  f"lead={e['lead_hours']}h  trust={e['trust']:.2f}  onset_cost={e['onset_cost']:.1f}")

    oracle_alert_hours = int((oracle_actions >= 1).sum())
    print(f"\nOracle alert hours: {oracle_alert_hours} / {len(oracle_actions)}")
    pd.DataFrame({"datetime": eval_df["datetime"], "oracle_action": oracle_actions}).to_csv(
        "oracle_actions.csv", index=False)

    print("\n=== Sensitivity: regret to oracle under alternative cost models ===")
    sweep = sensitivity_sweep(policies, trace, cfg)
    print(sweep.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    sweep.to_csv("exo_sensitivity.csv", index=False)
    print("\nSaved exo_eval_results.csv, exo_sensitivity.csv, oracle_actions.csv")
