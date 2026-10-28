"""
Figures for the paper (paper/figures/*.pdf), built from the same code paths
as evaluate_exo.py / exo_hql.py so every plotted number matches the tables.

  fig_case_2025.pdf      stage + alert rasters around the July 2025 flood
  fig_cost_breakdown.pdf exo-MDP cost components per policy
  fig_trust.pdf          modelled public trust over the eval period
  fig_tradeoff.pdf       false-alarm cost vs missed-protection cost
  fig_sensitivity.pdf    regret under alternative cost models
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_baselines import STATE_COLS, time_split
from exo_mdp import ExoConfig, COST_KEYS, build_exo_trace, step, trust_from_debt
from hindsight_oracle import solve_oracle
from evaluate_exo import collect_policies
from exo_hql import ExoHQL

OUT = os.path.join("paper", "figures")
COL_W = 3.5  # IEEE column width (in)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "serif",
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
})

LABELS = {
    "hindsight_oracle": "Hindsight oracle",
    "logged_nws": "Logged NWS",
    "rule_based_4level": "Rule (4-level)",
    "rule_based_threshold": "Rule (3-level)",
    "never_alert": "Never alert",
    "behavior_cloning": "BC",
    "discrete_cql": "CQL",
    "discrete_iql": "IQL",
    "cost_calibrated_iql": "CC-IQL",
    "exo_hql_shielded": "Exo-HQL+shield",
    "exo_hql": "Exo-HQL",
}


def save(fig, name):
    """PDF for LaTeX, PNG for quick preview."""
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=200)


def record(act, trace, cfg):
    """Replay a closed-loop or open-loop policy, recording actions and debt."""
    if not callable(act):
        seq = np.asarray(act)
        act = lambda t, a_prev, h, k: seq[t]
    T = len(trace["y"])
    actions, debt = np.zeros(T, dtype=int), np.zeros(T, dtype=int)
    a_prev, h, k = 0, 0, 0
    for t in range(T):
        a = int(np.clip(act(t, a_prev, h, k), 0, cfg.n_actions - 1))
        actions[t], debt[t] = a, k
        _, (a_prev, h, k) = step(a_prev, h, k, a, int(trace["y"][t]), int(trace["eff"][t]),
                                 int(trace["onset_peak"][t]), int(trace["tick"][t]), cfg)
        a_prev, h, k = int(a_prev), int(h), int(k)
    return actions, debt


def fig_case(eval_df, actions_by_name, onset_idx):
    lo, hi = onset_idx - 36, onset_idx + 36
    t = np.arange(lo, hi) - onset_idx
    stage = eval_df["gage_height_ft"].to_numpy()[lo:hi]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_W, 2.9), sharex=True,
                                   gridspec_kw={"height_ratios": [1.2, 1]})
    ax1.plot(t, stage, color="k", lw=1)
    for ft, name in [(7, "action"), (9, "minor"), (12, "moderate"), (20, "major")]:
        ax1.axhline(ft, color="0.6", lw=0.6, ls="--")
        ax1.text(t[0] + 0.5, ft + 0.4, name, fontsize=6, color="0.4")
    ax1.axvline(0, color="tab:red", lw=0.8)
    ax1.set_ylabel("Stage (ft)")

    names = list(actions_by_name)
    grid = np.array([actions_by_name[n][lo:hi] for n in names])
    cmap = matplotlib.colors.ListedColormap(["#f2f2f2", "#fdd49e", "#fc8d59", "#b30000"])
    ax2.imshow(grid, aspect="auto", cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest",
               extent=[t[0] - 0.5, t[-1] + 0.5, len(names) - 0.5, -0.5])
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels([LABELS[n] for n in names])
    ax2.axvline(0, color="tab:red", lw=0.8)
    ax2.set_xlabel("Hours relative to flood onset (4 Jul 2025)")
    handles = [matplotlib.patches.Patch(color=cmap(i), label=l)
               for i, l in enumerate(["None", "Watch", "Warning", "Severe"])]
    ax2.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.45, -0.42),
               frameon=False, handlelength=1)
    save(fig, "fig_case_2025")
    plt.close(fig)


def fig_cost_breakdown(table):
    order = ["hindsight_oracle", "rule_based_4level", "exo_hql_shielded", "exo_hql", "never_alert",
             "behavior_cloning", "cost_calibrated_iql", "discrete_iql", "logged_nws", "discrete_cql"]
    comp = {"false_alarm": "False alarm", "over_severity": "Over-severity",
            "under_alert": "Under-alert", "onset": "Onset (lead time)", "switching": "Switching"}
    colors = ["#4575b4", "#91bfdb", "#fc8d59", "#d73027", "#999999"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    left = np.zeros(len(order))
    for (key, label), c in zip(comp.items(), colors):
        vals = np.array([table.loc[n, key] for n in order])
        ax.barh(range(len(order)), vals, left=left, color=c, label=label, height=0.7)
        left += vals
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABELS[n] for n in order])
    ax.invert_yaxis()
    ax.set_xlabel("Exact counterfactual cost on eval split")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.4, 1.0), frameon=False,
              handlelength=1, columnspacing=0.8)
    save(fig, "fig_cost_breakdown")
    plt.close(fig)


def fig_trust(eval_df, debts, onsets, cfg):
    dt = pd.to_datetime(eval_df["datetime"]).dt.tz_localize(None)
    fig, ax = plt.subplots(figsize=(COL_W, 1.8))
    styles = {"logged_nws": ("k", "-"), "discrete_iql": ("tab:purple", "-"),
              "cost_calibrated_iql": ("tab:orange", "-"), "exo_hql_shielded": ("tab:green", "-")}
    for name, (c, ls) in styles.items():
        ax.plot(dt, trust_from_debt(debts[name], cfg), color=c, ls=ls, lw=0.9, label=LABELS[name])
    for i in onsets:
        ax.axvline(dt.iloc[i], color="tab:red", lw=0.6, ls=":")
    ax.set_ylabel("Modelled trust $T(k)$")
    ax.set_ylim(0.25, 1.03)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
              handlelength=1.2, columnspacing=0.8)
    fig.autofmt_xdate()
    save(fig, "fig_trust")
    plt.close(fig)


def fig_tradeoff(table):
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    for name in LABELS:
        if name not in table.index or name == "exo_hql":
            continue
        r = table.loc[name]
        x, y = r["false_alarm"] + r["over_severity"] + 1, r["under_alert"] + r["onset"]
        ours = name.startswith("exo_hql")
        ax.scatter(x, y, s=30 if ours else 14, marker="*" if ours else "o",
                   color="tab:green" if ours else "0.3", zorder=3)
        offset = {"cost_calibrated_iql": (-8, -9), "rule_based_threshold": (3, 4),
                  "rule_based_4level": (3, -7)}.get(name, (3, 2))
        ax.annotate(LABELS[name], (x, y), fontsize=6, xytext=offset, textcoords="offset points")
    r = table.loc["exo_hql"]
    ax.scatter(r["false_alarm"] + r["over_severity"] + 1, r["under_alert"] + r["onset"],
               s=30, marker="*", color="tab:olive", zorder=3)
    ax.annotate("Exo-HQL", (r["false_alarm"] + 1, r["under_alert"] + r["onset"]), fontsize=6,
                xytext=(3, 2), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("False-alarm + over-severity cost (+1, log)")
    ax.set_ylabel("Under-alert + onset cost")
    save(fig, "fig_tradeoff")
    plt.close(fig)


def fig_sensitivity():
    sw = pd.read_csv("exo_sensitivity.csv")
    names = ["rule_based_4level", "rule_based_threshold", "never_alert", "behavior_cloning",
             "cost_calibrated_iql", "discrete_iql", "logged_nws", "discrete_cql"]
    labels = [f"{a:g}/{b:g}" for a, b in zip(sw["onset_scale"], sw["switch_cost"])]
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    for n in names:
        ax.plot(range(len(sw)), sw[n], marker="o", ms=2.5, lw=0.9, label=LABELS[n])
    ax.set_xticks(range(len(sw)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_xlabel(r"Onset-cost scale / switching cost $\lambda_{sw}$")
    ax.set_ylabel("Regret to oracle")
    ax.set_yscale("log")
    ax.legend(ncol=4, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.32), frameon=False)
    save(fig, "fig_sensitivity")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv("kerrville_mdp_dataset.csv")
    df = df.dropna(subset=STATE_COLS).reset_index(drop=True)
    train_df, eval_df = time_split(df)
    action_size = int(df["alert_action"].max()) + 1
    cfg = ExoConfig()
    trace = build_exo_trace(eval_df, cfg)

    policies = collect_policies(eval_df, action_size)
    policies["hindsight_oracle"], _ = solve_oracle(trace, cfg)
    policies["exo_hql_shielded"] = ExoHQL(cfg, use_endogenous=True, shield=True, seed=0) \
        .fit(train_df).as_closed_loop(eval_df)

    actions, debts = {}, {}
    for name, pol in policies.items():
        actions[name], debts[name] = record(pol, trace, cfg)

    onsets = np.flatnonzero(trace["onset_peak"] > 0)
    july_2025 = next(i for i in onsets if pd.Timestamp(eval_df["datetime"].iloc[i]).year == 2025)
    fig_case(eval_df, {n: actions[n] for n in ["hindsight_oracle", "logged_nws", "rule_based_4level",
                                               "cost_calibrated_iql", "exo_hql_shielded"]}, july_2025)

    # cost table: baselines from exo_eval_results, Exo-HQL as seed means
    res = pd.read_csv("exo_hql_results.csv")
    cols = list(COST_KEYS) + ["exo_cost"]
    table = res.set_index("model")[cols]
    for name in ("exo_hql", "exo_hql_shielded"):
        table.loc[name] = res[res["model"].str.fullmatch(rf"{name}_s\d+")][cols].mean()
    fig_cost_breakdown(table)
    fig_tradeoff(table)
    fig_trust(eval_df, debts, onsets, cfg)
    fig_sensitivity()
    print(f"Saved figures to {OUT}/")
