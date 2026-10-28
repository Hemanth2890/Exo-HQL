"""
Hindsight-optimal alerting policy via exact backward DP on the exo-MDP.

Given the full exogenous trace (it knows the future hydrograph), the oracle
picks the cost-minimizing alert sequence over the endogenous grid
(a_prev, h, k) defined in exo_mdp.py. Its cost is a hard lower bound for
every causal policy under the same cost model, so `J(policy) - J(oracle)`
is a regret that is meaningful in absolute terms, not just a ranking.

Speed: almost every hour has the same exogenous signature (quiet, no
lookahead event), so per-signature cost and next-state tables are cached
and the backward pass is a single gather + min per hour.
"""

import numpy as np

from exo_mdp import ExoConfig, COST_KEYS, step


def _grid(cfg: ExoConfig):
    A, H, K = cfg.n_actions, cfg.max_alert_hours + 1, cfg.max_debt + 1
    a_prev, h, k = np.meshgrid(np.arange(A), np.arange(H), np.arange(K), indexing="ij")
    return a_prev.ravel(), h.ravel(), k.ravel(), (A, H, K)


def _encode(a_prev, h, k, shape):
    _, H, K = shape
    return (a_prev * H + h) * K + k


def solve_oracle(trace: dict, cfg: ExoConfig, sample_states: np.ndarray = None):
    """
    Returns (oracle_actions, oracle_cost) for the trajectory starting from
    the calm state (a_prev=0, h=0, k=0).

    sample_states: optional (T, n) array of encoded endogenous states. If
    given, also returns the hindsight action-values Q_t(z, a) at those
    states, shape (T, n, n_actions) -- the exact cost-to-go of EVERY action,
    not just the logged one, which is what exo_hql.py trains on.
    """
    a_prev, h, k, shape = _grid(cfg)
    n_states = a_prev.size
    T = len(trace["y"])

    cache = {}

    def tables(t):
        sig = (int(trace["y"][t]), int(trace["eff"][t]),
               int(trace["onset_peak"][t]), int(trace["tick"][t]))
        if sig not in cache:
            cost = np.empty((cfg.n_actions, n_states))
            nxt = np.empty((cfg.n_actions, n_states), dtype=np.int64)
            for a in range(cfg.n_actions):
                costs, (ap2, h2, k2) = step(a_prev, h, k, np.full(n_states, a), *sig, cfg)
                cost[a] = sum(costs[key] for key in COST_KEYS)
                nxt[a] = _encode(ap2, h2, k2, shape)
            cache[sig] = (cost, nxt)
        return cache[sig]

    V = np.zeros(n_states)
    policy = np.empty((T, n_states), dtype=np.int8)
    q_samples = None
    if sample_states is not None:
        q_samples = np.empty((T, sample_states.shape[1], cfg.n_actions), dtype=np.float32)
    for t in range(T - 1, -1, -1):
        cost, nxt = tables(t)
        Q = cost + V[nxt]
        policy[t] = Q.argmin(axis=0)
        V = Q.min(axis=0)
        if q_samples is not None:
            q_samples[t] = Q[:, sample_states[t]].T

    # forward rollout from the calm initial state
    s = _encode(0, 0, 0, shape)
    actions = np.empty(T, dtype=np.int64)
    for t in range(T):
        actions[t] = policy[t, s]
        _, nxt = tables(t)
        s = nxt[actions[t], s]
    if q_samples is not None:
        return actions, float(V[_encode(0, 0, 0, shape)]), q_samples
    return actions, float(V[_encode(0, 0, 0, shape)])
