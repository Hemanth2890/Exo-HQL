"""
Exogenous-MDP view of flood alerting, plus an exact counterfactual replay
evaluator.

Key structural fact: the river does not respond to the alert. The hourly
hydrograph (gage height, severity_outcome, ...) is an EXOGENOUS trace that
is identical under every alerting policy. Only a small ENDOGENOUS state is
actually driven by the actions:

  a_prev  alert level issued last hour            {0..n_actions-1}
  h       consecutive hours an alert (>=1) has    {0..max_alert_hours}
          been active -- i.e. current lead time
  k       credibility debt: accumulated wasted    {0..max_debt}
          alert hours ("cry-wolf" effect), which
          lowers public compliance with the next
          real warning

Because the exogenous part is action-independent, ANY candidate action
sequence can be scored exactly by replaying it against the logged trace --
no off-policy estimation, no importance weights. The same structure makes
the hindsight-optimal policy computable by exact DP over the small
endogenous grid (see hindsight_oracle.py).

All dynamics and costs live in `step()`, which is fully numpy-broadcastable
so the oracle (vectorized over the whole grid) and the replay evaluator
(scalar, one trajectory) share a single source of truth.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from build_mdp_dataset import COST_MATRIX, LOOKAHEAD_HOURS


@dataclass(frozen=True)
class ExoConfig:
    n_actions: int = 4                 # 0=None 1=Watch 2=Warning 3=Major/Severe (matches COST_MATRIX)
    switch_cost: float = 0.5           # per level changed per hour -- penalizes alert flapping
    onset_cost: tuple = (0.0, 10.0, 40.0, 200.0)  # one-off cost at event onset, by event PEAK severity,
                                                  # reduced by how prepared the public was (lead time x trust)
    lead_target_hours: int = 12        # lead time at which preparedness saturates
    max_alert_hours: int = 24          # cap on tracked h (must be >= lead_target_hours)
    max_debt: int = 24                 # cap on credibility debt k
    min_trust: float = 0.3             # compliance when debt is maxed out
    recovery_hours: int = 96           # debt decays by 1 every this many hours (clock-driven, so exact DP stays Markov)


def trust_from_debt(k, cfg: ExoConfig):
    return 1.0 - (1.0 - cfg.min_trust) * k / cfg.max_debt


def build_exo_trace(df: pd.DataFrame, cfg: ExoConfig) -> dict:
    """
    Extract the action-independent exogenous signals from an hourly MDP
    table (one contiguous split, e.g. eval_df):
      y      severity_outcome this hour
      eff    max severity over the next LOOKAHEAD_HOURS (from build_mdp_dataset)
      onset_peak  peak severity of the event starting this hour, 0 if no onset
      tick   1 on hours where credibility debt recovers by one unit
    """
    df = df.sort_values("datetime").reset_index(drop=True)
    y = df["severity_outcome"].fillna(0).to_numpy(dtype=np.int64)
    eff = df["effective_severity"].fillna(0).to_numpy(dtype=np.int64)
    onset = df["event_start"].fillna(False).to_numpy(dtype=bool)

    # peak severity of each event, scanning forward until severity returns to 0
    onset_peak = np.zeros(len(y), dtype=np.int64)
    for i in np.flatnonzero(onset):
        j = i
        while j < len(y) and y[j] >= 1:
            j += 1
        onset_peak[i] = y[i:j].max()

    tick = (np.arange(len(y)) % cfg.recovery_hours == cfg.recovery_hours - 1).astype(np.int64)
    return {"y": y, "eff": eff, "onset_peak": onset_peak, "tick": tick,
            "datetime": df["datetime"].to_numpy()}


def step(a_prev, h, k, a, y, eff, onset_peak, tick, cfg: ExoConfig):
    """
    One hour of the exo-MDP. Endogenous args (a_prev, h, k, a) may be numpy
    arrays (broadcast together); exogenous args are scalars for that hour.
    Returns (cost_components: dict of arrays, (a_prev', h', k')).
    """
    C = COST_MATRIX
    a = np.asarray(a)
    alerting = a >= 1
    trust = trust_from_debt(k, cfg)

    # alert hour with no real event now or within LOOKAHEAD_HOURS
    wasted = alerting & (y == 0) & (eff == 0)
    false_alarm = np.where(wasted, C[a, 0], 0.0)

    # precursor hour (event coming): free to warn, but over-stating the
    # eventual severity still costs
    over_severity = np.where((y == 0) & (eff >= 1) & (a > eff), C[a, eff], 0.0)

    # during the event: an issued alert only protects the fraction of
    # people who still comply (trust); the rest behave as if unwarned
    under_alert = np.where(y >= 1, trust * C[a, y] + (1.0 - trust) * C[0, y], 0.0)

    # one-off onset cost, discounted by preparedness = lead time x trust
    prep = trust * np.minimum(h, cfg.lead_target_hours) / cfg.lead_target_hours
    onset = cfg.onset_cost[onset_peak] * (1.0 - prep) if onset_peak > 0 else np.zeros_like(prep, dtype=float)

    switching = cfg.switch_cost * np.abs(a - a_prev)

    h_next = np.where(alerting, np.minimum(h + 1, cfg.max_alert_hours), 0)
    k_next = np.minimum(k + wasted.astype(np.int64), cfg.max_debt)
    if tick:
        k_next = np.maximum(k_next - 1, 0)

    costs = {
        "false_alarm": false_alarm, "over_severity": over_severity,
        "under_alert": under_alert, "onset": onset, "switching": switching,
    }
    return costs, (np.broadcast_to(a, h_next.shape), h_next, k_next)


COST_KEYS = ("false_alarm", "over_severity", "under_alert", "onset", "switching")


def replay(actions, trace: dict, cfg: ExoConfig) -> dict:
    """
    Exact counterfactual evaluation from a calm initial state. `actions` is
    either an open-loop sequence (one action per trace hour -- valid for any
    policy that conditions only on exogenous observations), or a callable
    act(t, a_prev, h, k) -> action for closed-loop policies that also read
    the endogenous alert state.
    """
    if callable(actions):
        act = actions
    else:
        seq = np.asarray(actions, dtype=np.int64)
        assert len(seq) == len(trace["y"]), "need exactly one action per trace hour"
        act = lambda t, a_prev, h, k: seq[t]

    a_prev, h, k = 0, 0, 0
    totals = {key: 0.0 for key in COST_KEYS}
    events = []
    for t in range(len(trace["y"])):
        a = int(np.clip(act(t, a_prev, h, k), 0, cfg.n_actions - 1))
        y, eff, peak, tick = (int(trace["y"][t]), int(trace["eff"][t]),
                              int(trace["onset_peak"][t]), int(trace["tick"][t]))
        costs, (a_prev, h_new, k_new) = step(a_prev, h, k, a, y, eff, peak, tick, cfg)
        for key in COST_KEYS:
            totals[key] += float(costs[key])
        if peak > 0:
            events.append({"datetime": trace["datetime"][t], "peak_severity": peak,
                           "lead_hours": int(h), "trust": float(trust_from_debt(k, cfg)),
                           "action_at_onset": int(a), "onset_cost": float(costs["onset"])})
        a_prev, h, k = int(a_prev), int(h_new), int(k_new)

    totals["total"] = sum(totals[key] for key in COST_KEYS)
    totals["final_debt"] = k
    totals["events"] = events
    return totals
