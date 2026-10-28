"""
Exo-HQL: Exogenous Hindsight Q-distillation for flood alerting.

Standard offline RL (BC / CQL / IQL / cost-calibrated IQL in this repo)
only ever sees the cost of the ONE action NWS actually took, and has to
guess the rest -- hence conservatism penalties, expectile tuning, etc.

In the exo-MDP (exo_mdp.py) the river is action-independent, so the
hindsight DP (hindsight_oracle.py) gives the exact cost-to-go
Q_t(z, a) of EVERY action a, at ANY endogenous alert state z, for every
training hour. That turns policy learning into full-information
cost-sensitive classification:

    min_pi  E_{t, z} [ sum_a pi(a | x_t, z) * (Q_t(z, a) - min_a' Q_t(z, a')) ]

i.e. minimize expected hindsight regret. No off-policy correction, no
conservatism hyper-parameter, no tau.

That objective is optimized through a stakes-weighted cross-entropy
surrogate: target = softmin(regret / TEMP), weight = max_a regret (how
much this decision matters). The plain expected-regret loss has vanishing
gradients once pi saturates on "no alert" (dL/dlogit ~ p(1-p)R), so the
11 flood hours among 132k training hours can never pull it back -- it
collapses to never alerting (verified empirically).

Two design points:
  * the policy is CLOSED-LOOP: it reads the exogenous observation x_t
    (STATE_COLS) AND its own alert state z = (last level, hours alerted,
    trust). It can therefore learn not to cry wolf when trust is low, and
    to hold an alert once it has been issued.
  * z is sampled per training hour (calm state, the state the logged NWS
    alerts actually produced, plus random feasible states), so the policy
    is trained on the states it will reach when it acts on its own --
    exactly evaluable because Q is known everywhere.

Caveat (state it in the paper): hindsight Q is clairvoyant, so Q-values
of hours just before an event credit alerting with information the
policy cannot observe. The policy can only act on whatever precursor
signal x_t carries; with gauge-only features that is little, which is
why multi-gauge rainfall forcing is the natural next step.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from train_baselines import STATE_COLS, time_split
from exo_mdp import ExoConfig, build_exo_trace, replay, step, trust_from_debt
from hindsight_oracle import solve_oracle, _encode

N_RANDOM_STATES = 14
HIDDEN = 128
EPOCHS = 5
BATCH = 4096
LR = 1e-3
TEMP = 0.5   # softmin temperature over regrets (cost units)


def endogenous_features(a_prev, h, k, cfg: ExoConfig) -> np.ndarray:
    a_prev, h, k = np.atleast_1d(a_prev), np.atleast_1d(h), np.atleast_1d(k)
    onehot = np.eye(cfg.n_actions, dtype=np.float32)[a_prev]
    return np.column_stack([onehot, h / cfg.max_alert_hours, trust_from_debt(k, cfg)]).astype(np.float32)


def logged_state_path(actions: np.ndarray, trace: dict, cfg: ExoConfig):
    """Endogenous states the logged (NWS) action sequence actually visits."""
    a_prev, h, k = 0, 0, 0
    path = np.empty((len(actions), 3), dtype=np.int64)
    for t, a in enumerate(actions):
        path[t] = (a_prev, h, k)
        _, (a_prev, h, k) = step(a_prev, h, k, int(min(a, cfg.n_actions - 1)),
                                 int(trace["y"][t]), int(trace["eff"][t]),
                                 int(trace["onset_peak"][t]), int(trace["tick"][t]), cfg)
        a_prev, h, k = int(a_prev), int(h), int(k)
    return path


def sample_states(T: int, logged_path: np.ndarray, cfg: ExoConfig, rng) -> np.ndarray:
    """(T, n, 3) feasible endogenous states per hour: calm, logged, random."""
    n = 2 + N_RANDOM_STATES
    z = np.zeros((T, n, 3), dtype=np.int64)
    z[:, 1] = logged_path
    a = rng.integers(0, cfg.n_actions, size=(T, N_RANDOM_STATES))
    h = np.where(a > 0, rng.integers(1, cfg.max_alert_hours + 1, size=a.shape), 0)  # h>0 only while alerting
    k = rng.integers(0, cfg.max_debt + 1, size=a.shape)
    z[:, 2:] = np.stack([a, h, k], axis=-1)
    return z


class Policy(nn.Module):
    def __init__(self, in_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def shield_floor(df: pd.DataFrame) -> np.ndarray:
    """Safety shield: never alert below the official NWS-threshold rule."""
    from evaluate_exo import rule_based_4level
    return rule_based_4level(df)


class ExoHQL:
    def __init__(self, cfg: ExoConfig, use_endogenous: bool = True, shield: bool = False, seed: int = 0):
        self.cfg = cfg
        self.use_endogenous = use_endogenous
        self.shield = shield
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

    def _inputs(self, x, z_feats):
        if not self.use_endogenous:
            z_feats = np.zeros_like(z_feats)  # ablation: open-loop, blind to its own alert state
        return np.concatenate([x, z_feats], axis=-1)

    def fit(self, train_df: pd.DataFrame):
        cfg = self.cfg
        x = train_df[STATE_COLS].to_numpy(dtype=np.float32)
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        x = (x - self.mu) / self.sd

        trace = build_exo_trace(train_df, cfg)
        T = len(x)
        z = sample_states(T, logged_state_path(train_df["alert_action"].to_numpy(), trace, cfg), cfg, self.rng)
        shape = (cfg.n_actions, cfg.max_alert_hours + 1, cfg.max_debt + 1)
        z_idx = _encode(z[..., 0], z[..., 1], z[..., 2], shape)
        _, _, Q = solve_oracle(trace, cfg, sample_states=z_idx)       # (T, n, A)
        regret = Q - Q.min(axis=-1, keepdims=True)

        n = z.shape[1]
        z_feats = endogenous_features(z[..., 0].ravel(), z[..., 1].ravel(), z[..., 2].ravel(), cfg)
        inputs = self._inputs(np.repeat(x, n, axis=0), z_feats)
        targets = regret.reshape(T * n, cfg.n_actions)

        X = torch.as_tensor(inputs)
        R = torch.as_tensor(targets)
        if self.shield:
            # actions below the rule floor are unavailable: re-base regret on
            # the allowed set, and give disallowed actions zero target mass
            floor = torch.as_tensor(np.repeat(shield_floor(train_df), n))
            allowed = torch.arange(cfg.n_actions)[None, :] >= floor[:, None]
            R = torch.where(allowed, R, torch.inf)
            R = R - R.min(dim=-1, keepdim=True).values
            self._allowed = allowed
        finite_R = torch.where(torch.isfinite(R), R, 0.0)
        soft_target = torch.softmax(-R / TEMP, dim=-1)
        stakes = finite_R.max(dim=-1).values
        stakes = stakes / stakes.mean()
        R = finite_R
        self.policy = Policy(X.shape[1], cfg.n_actions)
        opt = torch.optim.Adam(self.policy.parameters(), lr=LR)
        for epoch in range(EPOCHS):
            perm = torch.randperm(len(X))
            total_loss, total_regret = 0.0, 0.0
            for i in range(0, len(X), BATCH):
                b = perm[i:i + BATCH]
                logits = self.policy(X[b])
                if self.shield:
                    logits = logits.masked_fill(~self._allowed[b], -1e9)
                log_probs = torch.log_softmax(logits, dim=-1)
                ce = -(soft_target[b] * log_probs.clamp(min=-1e4)).sum(-1)
                loss = (stakes[b] * ce).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(b)
                total_regret += (log_probs.exp() * R[b]).sum(-1).sum().item()
            print(f"  epoch {epoch}: loss={total_loss / len(X):.4f} "
                  f"expected regret/sample={total_regret / len(X):.4f}")
        return self

    def save(self, path: str):
        torch.save({"policy": self.policy.state_dict(), "mu": self.mu, "sd": self.sd,
                    "use_endogenous": self.use_endogenous, "shield": self.shield}, path)

    @classmethod
    def load(cls, path: str, cfg: ExoConfig):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(cfg, use_endogenous=ckpt["use_endogenous"], shield=ckpt["shield"])
        model.mu, model.sd = ckpt["mu"], ckpt["sd"]
        model.policy = Policy(len(STATE_COLS) + cfg.n_actions + 2, cfg.n_actions)
        model.policy.load_state_dict(ckpt["policy"])
        return model

    def as_closed_loop(self, df: pd.DataFrame):
        """act(t, a_prev, h, k) for exo_mdp.replay on the given split."""
        x = (df[STATE_COLS].to_numpy(dtype=np.float32) - self.mu) / self.sd
        floor = shield_floor(df) if self.shield else np.zeros(len(df), dtype=np.int64)
        # one row per hour, so a plain numpy forward pass avoids torch's
        # per-call overhead (~10x faster replay, identical outputs)
        layers = [(m.weight.detach().numpy().T, m.bias.detach().numpy())
                  for m in self.policy.net if isinstance(m, nn.Linear)]

        def act(t, a_prev, h, k):
            out = self._inputs(x[t:t + 1], endogenous_features(a_prev, h, k, self.cfg))[0]
            for i, (W, b) in enumerate(layers):
                out = out @ W + b
                if i < len(layers) - 1:
                    out = np.maximum(out, 0.0)
            out[:floor[t]] = -np.inf
            return int(out.argmax())
        return act


if __name__ == "__main__":
    from evaluate_exo import collect_policies, evaluate_all, results_table

    df = pd.read_csv("kerrville_mdp_dataset.csv")
    df = df.dropna(subset=STATE_COLS).reset_index(drop=True)
    train_df, eval_df = time_split(df)
    action_size = int(df["alert_action"].max()) + 1

    cfg = ExoConfig()
    eval_trace = build_exo_trace(eval_df, cfg)
    results, _ = evaluate_all(collect_policies(eval_df, action_size), eval_trace, cfg)

    seeds = (0, 1, 2)
    variants = [
        ("exo_hql", True, False),
        ("exo_hql_no_alert_state", False, False),
        ("exo_hql_shielded", True, True),
        ("exo_hql_shielded_no_alert_state", False, True),
    ]
    for name, use_endo, shield in variants:
        for seed in seeds:
            print(f"Training {name} seed={seed}")
            model = ExoHQL(cfg, use_endogenous=use_endo, shield=shield, seed=seed).fit(train_df)
            results[f"{name}_s{seed}"] = replay(model.as_closed_loop(eval_df), eval_trace, cfg)

    table = results_table(results)
    pd.set_option("display.width", 200)
    print("\n=== Exo-MDP replay on eval split, incl. Exo-HQL (lower cost is better) ===")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    table.to_csv("exo_hql_results.csv", index=False)

    print("\n=== Exo-HQL mean +- std over seeds ===")
    for name, _, _ in variants:
        rows = table[table["model"].str.fullmatch(rf"{name}_s\d+")]
        print(f"  {name:24s} cost={rows['exo_cost'].mean():.1f} +- {rows['exo_cost'].std():.1f}  "
              f"regret={rows['regret_to_oracle'].mean():.1f}  lead={rows['mean_lead_hours'].mean():.2f}h  "
              f"false_alarm={rows['false_alarm'].mean():.1f}")
    print("\nSaved exo_hql_results.csv")
