"""
Discrete-action IQL (Implicit Q-Learning), implemented from scratch since
d3rlpy's IQL class only supports continuous action spaces.

Core mechanism (Kostrikov et al. 2021, adapted to discrete actions):
  1. V(s) trained via expectile regression toward min(Q1,Q2)(s,a) -- this is
     the piece that makes IQL "conservative" without needing to query
     out-of-distribution actions.
  2. Q(s,a) trained via standard TD regression toward r + gamma * V(s').
  3. Policy extracted via advantage-weighted regression (AWR): states/actions
     with higher (Q - V) advantage get upweighted in a behavior-cloning-style
     loss.

`tau_scalar` in `update()` is the expectile parameter (>0.5 = more
conservative / more weight on positive residuals). It's exposed as a
per-call argument, not a fixed constant, so the future cost-calibrated
variant (tau as a function of state/action/severity) can be dropped in by
replacing this scalar with a per-sample tensor -- same training loop.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_baselines import STATE_COLS, build_episodes, time_split

GAMMA = 0.999  # bumped from 0.99 for hourly steps -- keeps a similar real-time
               # effective horizon (~1/(1-gamma) steps) to the old daily version
EXPECTILE_TAU = 0.7   # IQL default range is ~0.5-0.9; higher = more conservative
AWR_BETA = 3.0        # advantage-weighting temperature
AWR_CLIP = 100.0      # clip advantage weights for training stability
HIDDEN = 128


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def build_transitions(df: pd.DataFrame, state_cols: list):
    """
    Turn the daily MDP table into (s, a, r, s', done) transition tuples,
    respecting episode boundaries from build_episodes (no bootstrapping
    across an episode terminal).
    """
    df = df.sort_values("datetime").reset_index(drop=True)
    terminals = build_episodes(df)
    states = df[state_cols].to_numpy(dtype=np.float32)
    actions = df["alert_action"].to_numpy(dtype=np.int64)
    rewards = df["reward"].to_numpy(dtype=np.float32)

    n = len(df)
    next_states = np.zeros_like(states)
    dones = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if terminals[i] == 1.0 or i == n - 1:
            next_states[i] = states[i]  # terminal: bootstrap masked by done=1
            dones[i] = 1.0
        else:
            next_states[i] = states[i + 1]
            dones[i] = 0.0
    return states, actions, rewards, next_states, dones


class DiscreteIQL:
    def __init__(self, state_dim: int, action_size: int, device: str = "cpu"):
        self.device = device
        self.action_size = action_size

        self.q1 = MLP(state_dim, action_size).to(device)
        self.q2 = MLP(state_dim, action_size).to(device)
        self.q1_target = MLP(state_dim, action_size).to(device)
        self.q2_target = MLP(state_dim, action_size).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.v = MLP(state_dim, 1).to(device)
        self.policy = MLP(state_dim, action_size).to(device)

        self.opt_q = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=3e-4
        )
        self.opt_v = torch.optim.Adam(self.v.parameters(), lr=3e-4)
        self.opt_pi = torch.optim.Adam(self.policy.parameters(), lr=3e-4)

    @staticmethod
    def expectile_loss(diff: torch.Tensor, tau) -> torch.Tensor:
        """
        tau may be a python float (uniform expectile) or a tensor the same
        shape as diff (per-sample expectile) -- this is the hook the
        cost-calibrated variant will use later.
        """
        weight = torch.where(diff > 0, tau, 1 - tau)
        return (weight * diff.pow(2)).mean()

    def update(self, s, a, r, s2, done, tau_scalar=EXPECTILE_TAU):
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.long, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        s2 = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            q1_sa = self.q1_target(s).gather(1, a.unsqueeze(1)).squeeze(1)
            q2_sa = self.q2_target(s).gather(1, a.unsqueeze(1)).squeeze(1)
            q_sa = torch.min(q1_sa, q2_sa)

        # --- V update: expectile regression toward min(Q1,Q2)(s,a) ---
        v_s = self.v(s).squeeze(1)
        diff = q_sa - v_s
        v_loss = self.expectile_loss(diff, tau_scalar)
        self.opt_v.zero_grad()
        v_loss.backward()
        self.opt_v.step()

        # --- Q update: TD target using V(s'), never querying unseen actions ---
        with torch.no_grad():
            v_s2 = self.v(s2).squeeze(1)
            target = r + GAMMA * (1 - done) * v_s2

        q1_pred = self.q1(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q2_pred = self.q2(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)
        self.opt_q.zero_grad()
        q_loss.backward()
        self.opt_q.step()

        # --- policy update: advantage-weighted regression (AWR) ---
        with torch.no_grad():
            adv = q_sa - v_s
            weight = torch.exp(AWR_BETA * adv).clamp(max=AWR_CLIP)
        logits = self.policy(s)
        log_probs = F.log_softmax(logits, dim=1).gather(1, a.unsqueeze(1)).squeeze(1)
        pi_loss = -(weight * log_probs).mean()
        self.opt_pi.zero_grad()
        pi_loss.backward()
        self.opt_pi.step()

        return {"v_loss": v_loss.item(), "q_loss": q_loss.item(), "pi_loss": pi_loss.item()}

    def soft_update_targets(self, tau: float = 0.005):
        for target, source in [(self.q1_target, self.q1), (self.q2_target, self.q2)]:
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)

    def predict(self, s):
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.policy(s)
        return logits.argmax(dim=1).cpu().numpy()

    def save(self, path: str):
        torch.save({
            "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(), "q2_target": self.q2_target.state_dict(),
            "v": self.v.state_dict(), "policy": self.policy.state_dict(),
        }, path)

    def load(self, path: str, map_location: str = "cpu"):
        ckpt = torch.load(path, map_location=map_location)
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        self.v.load_state_dict(ckpt["v"])
        self.policy.load_state_dict(ckpt["policy"])


def train(states, actions, rewards, next_states, dones, action_size,
          n_steps=10_000, batch_size=256, device="cpu"):
    model = DiscreteIQL(states.shape[1], action_size, device=device)
    n = len(states)
    for step in range(n_steps):
        idx = np.random.randint(0, n, size=min(batch_size, n))
        losses = model.update(states[idx], actions[idx], rewards[idx],
                               next_states[idx], dones[idx])
        model.soft_update_targets()
        if step % 1000 == 0:
            print(f"step {step}: v_loss={losses['v_loss']:.4f} "
                  f"q_loss={losses['q_loss']:.4f} pi_loss={losses['pi_loss']:.4f}")
    return model


if __name__ == "__main__":
    df = pd.read_csv("kerrville_mdp_dataset.csv")
    df = df.dropna(subset=STATE_COLS).reset_index(drop=True)
    train_df, eval_df = time_split(df)

    states, actions, rewards, next_states, dones = build_transitions(train_df, STATE_COLS)
    action_size = int(df["alert_action"].max()) + 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training discrete IQL on {len(states)} transitions, "
          f"action_size={action_size}, device={device}")
    model = train(states, actions, rewards, next_states, dones, action_size,
                  n_steps=10_000, device=device)
    model.save("discrete_iql.pt")
    print("Saved discrete_iql.pt")