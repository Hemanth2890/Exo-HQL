"""
Cost-calibrated discrete IQL.

Plain IQL (discrete_iql.py) uses a single fixed expectile tau for every
transition. This variant replaces that constant with a per-transition tau
derived from the ordinal severity cost matrix (COST_MATRIX, defined in
build_mdp_dataset.py): states where the true outcome was severe get a
higher tau (V leans further toward the safer/higher-value action
estimates, penalizing under-alerting harder), while states where the cost
of over- vs under-alerting is closer to symmetric get tau near 0.5
(ordinary expectile regression).

Everything else -- Q update, target networks, AWR policy extraction -- is
identical to discrete_iql.py. This is the concrete implementation of the
tau(s,a) hook that discrete_iql.py's update() was built to support.
"""

import numpy as np
import pandas as pd
import torch

from discrete_iql import DiscreteIQL, build_transitions as _build_transitions_base
from build_mdp_dataset import COST_MATRIX
from train_baselines import STATE_COLS, time_split

TAU_MIN = 0.5   # symmetric expectile floor (ordinary regression)
TAU_MAX = 0.95  # near-max conservatism ceiling (standard IQL practical upper bound)


def compute_tau_by_severity() -> dict:
    """
    For each true severity level y, compare the average cost of
    under-alerting (issuing a < y) vs over-alerting (issuing a > y) using
    the ordinal cost matrix. A larger under-cost/over-cost ratio -> higher
    tau. Returns {severity_level: tau}.
    """
    n_levels = COST_MATRIX.shape[0]
    tau_lookup = {}
    for y in range(n_levels):
        under_costs = [COST_MATRIX[a, y] for a in range(n_levels) if a < y]
        over_costs = [COST_MATRIX[a, y] for a in range(n_levels) if a > y]
        under_cost = float(np.mean(under_costs)) if under_costs else 0.0
        over_cost = float(np.mean(over_costs)) if over_costs else 0.0
        total = under_cost + over_cost
        asymmetry = under_cost / total if total > 0 else 0.5  # no info -> neutral
        tau_lookup[y] = TAU_MIN + (TAU_MAX - TAU_MIN) * asymmetry
    return tau_lookup


def build_transitions_with_severity(df: pd.DataFrame, state_cols: list, lookahead_hours: int = 72):
    """
    Same as discrete_iql.build_transitions, but also returns an *effective*
    severity per transition for tau lookup: max(severity_outcome) over the
    next `lookahead_hours` hours (inclusive of the current hour), not just
    the current hour's actual severity. Default 72h (3 days) matches the
    original daily version's 3-day lookahead in real-world terms.

    Why: tau indexed by the current hour's severity alone only elevates
    conservatism once an event has already started -- structurally
    incapable of producing early warnings, since severity_outcome=0 for
    every precursor hour right up until onset. Using a forward-looking max
    lets hours leading up to a severe event inherit that event's higher
    tau, so the value function is pushed to weight the safer/higher-alert
    direction *before* the threshold crossing, not just at or after it.
    """
    df = df.sort_values("datetime").reset_index(drop=True)
    states, actions, rewards, next_states, dones = _build_transitions_base(df, state_cols)

    raw_severity = df["severity_outcome"].fillna(0).to_numpy(dtype=np.int64)
    n = len(raw_severity)
    effective_severity = raw_severity.copy()
    for i in range(n):
        window_end = min(n, i + lookahead_hours + 1)
        effective_severity[i] = raw_severity[i:window_end].max()

    return states, actions, rewards, next_states, dones, effective_severity


def train_cost_calibrated(states, actions, rewards, next_states, dones, effective_severities,
                           action_size, n_steps=10_000, batch_size=256, device="cpu"):
    tau_lookup = compute_tau_by_severity()
    print("Per-severity expectile tau:", tau_lookup)

    model = DiscreteIQL(states.shape[1], action_size, device=device)
    n = len(states)
    for step in range(n_steps):
        idx = np.random.randint(0, n, size=min(batch_size, n))
        tau_batch = np.array(
            [tau_lookup[int(sev)] for sev in effective_severities[idx]], dtype=np.float32
        )
        tau_tensor = torch.as_tensor(tau_batch, device=device)

        losses = model.update(states[idx], actions[idx], rewards[idx],
                               next_states[idx], dones[idx], tau_scalar=tau_tensor)
        model.soft_update_targets()
        if step % 1000 == 0:
            print(f"step {step}: v_loss={losses['v_loss']:.4f} "
                  f"q_loss={losses['q_loss']:.4f} pi_loss={losses['pi_loss']:.4f}")
    return model


if __name__ == "__main__":
    df = pd.read_csv("kerrville_mdp_dataset.csv")
    df = df.dropna(subset=STATE_COLS).reset_index(drop=True)
    train_df, eval_df = time_split(df)

    states, actions, rewards, next_states, dones, effective_severities = build_transitions_with_severity(
        train_df, STATE_COLS
    )
    action_size = int(df["alert_action"].max()) + 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training cost-calibrated discrete IQL on {len(states)} transitions, "
          f"action_size={action_size}, device={device}")
    model = train_cost_calibrated(
        states, actions, rewards, next_states, dones, effective_severities,
        action_size, n_steps=10_000, device=device,
    )
    model.save("cost_calibrated_iql.pt")
    print("Saved cost_calibrated_iql.pt")