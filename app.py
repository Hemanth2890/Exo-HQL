"""
Streamlit demo for "Counterfactually Exact Offline Learning for Flood Early
Warning" (Exo-HQL).

    streamlit run app.py

Pages:
  Overview              the problem, the idea, the data
  Compare methods       exact counterfactual replay of every method + oracle
  Flood event explorer  hour-by-hour alerts of chosen methods around a flood
  Evaluate Exo-HQL      change the cost model / method options, (re)train
                        Exo-HQL and evaluate it against all baselines
"""

from dataclasses import replace

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from train_baselines import STATE_COLS, time_split
from exo_mdp import ExoConfig, COST_KEYS, build_exo_trace, replay, trust_from_debt
from hindsight_oracle import solve_oracle
from evaluate_exo import collect_policies
from exo_hql import ExoHQL

st.set_page_config(page_title="Exo-HQL Flood Warning", page_icon="🌊", layout="wide")

LABELS = {
    "hindsight_oracle": "Hindsight oracle (bound)",
    "exo_hql_shielded": "Exo-HQL + shield (ours)",
    "exo_hql": "Exo-HQL (ours)",
    "rule_based_4level": "Rule, 4-level",
    "rule_action_stage": "Rule, watch at action stage",
    "rule_based_threshold": "Rule, 3-level",
    "never_alert": "Never alert",
    "logged_nws": "Logged NWS alerts",
    "behavior_cloning": "Behavior cloning",
    "cost_calibrated_iql": "CC-IQL (our prior)",
    "discrete_iql": "IQL",
    "discrete_cql": "CQL",
}
GROUP = {
    "hindsight_oracle": "Oracle", "exo_hql_shielded": "Ours", "exo_hql": "Ours",
    "rule_based_4level": "Rule", "rule_action_stage": "Rule", "rule_based_threshold": "Rule",
    "never_alert": "Rule", "logged_nws": "NWS", "behavior_cloning": "Offline RL",
    "cost_calibrated_iql": "Offline RL", "discrete_iql": "Offline RL", "discrete_cql": "Offline RL",
}
COMPONENT_LABELS = {"false_alarm": "False alarm", "over_severity": "Over-severity",
                    "under_alert": "Under-alert", "onset": "Onset (missing lead time)",
                    "switching": "Switching"}
LEVELS = ["None", "Watch", "Warning", "Severe"]
LEVEL_COLORS = ["#eeeeee", "#fdd49e", "#fc8d59", "#b30000"]


# ---------------------------------------------------------------- data + models

@st.cache_data(show_spinner="Loading Kerrville hourly dataset...")
def load_data():
    df = pd.read_csv("kerrville_mdp_dataset.csv")
    df = df.dropna(subset=STATE_COLS).reset_index(drop=True)
    train_df, eval_df = time_split(df)
    return train_df, eval_df


@st.cache_resource(show_spinner="Loading baseline models (BC, CQL, IQL, CC-IQL)...")
def baseline_actions():
    _, eval_df = load_data()
    action_size = int(eval_df["alert_action"].max()) + 1
    return collect_policies(eval_df, max(action_size, 3))


@st.cache_resource(show_spinner="Loading pretrained Exo-HQL models...")
def pretrained_hql():
    cfg = ExoConfig()
    return {"exo_hql_shielded": ExoHQL.load("exo_hql_shielded.pt", cfg),
            "exo_hql": ExoHQL.load("exo_hql.pt", cfg)}


@st.cache_resource(show_spinner=False)
def train_hql(cfg_key: tuple, shield: bool, use_alert_state: bool, seed: int):
    train_df, _ = load_data()
    return ExoHQL(cfg_from_key(cfg_key), use_endogenous=use_alert_state, shield=shield, seed=seed).fit(train_df)


def cfg_to_key(cfg: ExoConfig) -> tuple:
    return (cfg.switch_cost, cfg.onset_cost, cfg.lead_target_hours, cfg.min_trust, cfg.recovery_hours)


def cfg_from_key(key: tuple) -> ExoConfig:
    sw, onset, lead, trust, rec = key
    return replace(ExoConfig(), switch_cost=sw, onset_cost=tuple(onset), lead_target_hours=lead,
                   min_trust=trust, recovery_hours=rec)


def run_policy(policy, trace, cfg):
    """Exact replay in one pass: actions, credibility debt and cost breakdown."""
    T = len(trace["y"])
    actions, debt = np.zeros(T, dtype=int), np.zeros(T, dtype=int)
    act = policy if callable(policy) else (lambda t, a_prev, h, k, seq=np.asarray(policy): seq[t])

    def recording_act(t, a_prev, h, k):
        a = int(np.clip(act(t, a_prev, h, k), 0, cfg.n_actions - 1))
        actions[t], debt[t] = a, k
        return a

    result = replay(recording_act, trace, cfg)
    return {"actions": actions, "debt": debt, "result": result}


@st.cache_data(show_spinner="Running exact counterfactual replay for every method...")
def evaluate_everything(cfg_key: tuple):
    cfg = cfg_from_key(cfg_key)
    _, eval_df = load_data()
    trace = build_exo_trace(eval_df, cfg)
    policies = dict(baseline_actions())
    policies["hindsight_oracle"], _ = solve_oracle(trace, cfg)
    for name, model in pretrained_hql().items():
        policies[name] = model.as_closed_loop(eval_df)
    return {name: run_policy(p, trace, cfg) for name, p in policies.items()}


def results_frame(runs: dict, cfg: ExoConfig) -> pd.DataFrame:
    oracle = runs["hindsight_oracle"]["result"]["total"]
    rows = []
    for name, run in runs.items():
        r = run["result"]
        leads = [e["lead_hours"] for e in r["events"]]
        rows.append({
            "key": name, "Method": LABELS.get(name, name), "Type": GROUP.get(name, "Ours"),
            "Total cost": r["total"], "Regret to oracle": r["total"] - oracle,
            **{COMPONENT_LABELS[k]: r[k] for k in COST_KEYS},
            "Alert hours": int((run["actions"] > 0).sum()),
            "Lead time per event (h)": " / ".join(str(l) for l in leads),
            "Min trust": float(trust_from_debt(run["debt"], cfg).min()),
        })
    return pd.DataFrame(rows).sort_values("Total cost").reset_index(drop=True)


# ---------------------------------------------------------------- charts

def cost_breakdown_chart(table: pd.DataFrame):
    long = table.melt(id_vars=["Method"], value_vars=list(COMPONENT_LABELS.values()),
                      var_name="Component", value_name="Cost")
    order = table["Method"].tolist()
    return alt.Chart(long).mark_bar().encode(
        x=alt.X("sum(Cost):Q", title="Exact counterfactual cost (lower is better)"),
        y=alt.Y("Method:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
        color=alt.Color("Component:N", scale=alt.Scale(
            domain=list(COMPONENT_LABELS.values()),
            range=["#4575b4", "#91bfdb", "#fc8d59", "#d73027", "#999999"])),
        tooltip=["Method", "Component", alt.Tooltip("Cost:Q", format=".1f")],
    ).properties(height=34 * len(order))


def trust_chart(runs: dict, names: list, eval_df: pd.DataFrame, cfg: ExoConfig):
    dt = pd.to_datetime(eval_df["datetime"]).dt.tz_localize(None)
    frames = []
    for n in names:
        trust = trust_from_debt(runs[n]["debt"], cfg)
        frames.append(pd.DataFrame({"time": dt, "Trust": trust, "Method": LABELS[n]}).iloc[::6])
    return alt.Chart(pd.concat(frames)).mark_line(strokeWidth=1.2).encode(
        x=alt.X("time:T", title=None), y=alt.Y("Trust:Q", scale=alt.Scale(domain=[0.25, 1.02])),
        color="Method:N", tooltip=["Method", "time:T", alt.Tooltip("Trust:Q", format=".2f")],
    ).properties(height=260)


def event_charts(eval_df, runs, names, onset_idx, window):
    lo, hi = max(0, onset_idx - window), min(len(eval_df), onset_idx + window)
    rel = np.arange(lo, hi) - onset_idx
    stage = pd.DataFrame({"Hours from onset": rel,
                          "Stage (ft)": eval_df["gage_height_ft"].to_numpy()[lo:hi]})
    thresholds = pd.DataFrame({"ft": [7, 9, 12, 20], "label": ["action", "minor", "moderate", "major"]})
    x_scale = alt.Scale(domain=[-window, window])
    line = alt.Chart(stage).mark_line(color="#1f77b4", strokeWidth=2).encode(
        x=alt.X("Hours from onset:Q", scale=x_scale, title="Hours from flood onset"), y="Stage (ft):Q",
        tooltip=["Hours from onset", alt.Tooltip("Stage (ft):Q", format=".2f")])
    rules = alt.Chart(thresholds).mark_rule(strokeDash=[4, 3], color="gray").encode(
        y=alt.Y("ft:Q", title="Stage (ft)"))
    text = alt.Chart(thresholds).mark_text(align="left", dx=4, dy=-6, color="gray", fontSize=11).encode(
        y=alt.Y("ft:Q", title="Stage (ft)"), x=alt.value(0), text="label:N")
    onset = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="red").encode(x=alt.X("x:Q", scale=x_scale))
    top = (line + rules + text + onset).properties(height=230)

    raster = pd.concat([pd.DataFrame({
        "start": rel - 0.5, "end": rel + 0.5, "Hour": rel, "Method": LABELS[n],
        "Alert": [LEVELS[a] for a in runs[n]["actions"][lo:hi]]}) for n in names])
    bottom = alt.Chart(raster).mark_rect().encode(
        x=alt.X("start:Q", scale=x_scale, title="Hours from flood onset"), x2="end:Q",
        y=alt.Y("Method:N", sort=[LABELS[n] for n in names], title=None,
                axis=alt.Axis(labelLimit=260, labelOverlap=False)),
        color=alt.Color("Alert:N", scale=alt.Scale(domain=LEVELS, range=LEVEL_COLORS)),
        tooltip=["Method", alt.Tooltip("Hour:Q", title="Hours from onset"), "Alert"],
    ).properties(height=34 * len(names))
    return top, bottom


# ---------------------------------------------------------------- pages

def page_overview():
    train_df, eval_df = load_data()
    st.title("🌊 Exo-HQL: counterfactually exact learning for flood early warning")
    st.caption("Guadalupe River at Kerrville, TX (USGS 08166200 / NWS KRRT2), hourly data 2007–2026")

    st.markdown("""
**The key observation: a river does not respond to the warning issued about it.**
Stage, discharge and flood severity evolve the same way under every alerting policy.
Only a small *alert state* depends on our decisions:

* the alert level currently in force,
* how long it has been in force (the **lead time**),
* how much the public still **trusts** alerts. Trust drops with every false alarm (the "cry-wolf" effect).

That makes the problem an **exogenous MDP**, which gives us three things standard offline RL doesn't have:
""")
    c1, c2, c3 = st.columns(3)
    c1.info("**1. Exact evaluation**\n\nReplay any policy against the logged river record and get its *exact* cost, with no off-policy estimation.")
    c2.info("**2. A hindsight oracle**\n\nDynamic programming over 2,500 alert states gives the best possible cost, so every method gets an absolute regret.")
    c3.info("**3. Full-information training**\n\nThe oracle's DP gives the true cost of *every* action at every hour. Exo-HQL learns from all of them.")

    st.subheader("Exo-HQL (proposed method)")
    st.markdown("""
1. Run the hindsight DP on the training years to get $Q^H_t(z,a)$ for **every** action and alert state.
2. Sample 16 alert states per hour (calm, the state the real NWS alerts produced, and 14 random ones).
3. Train a closed-loop policy $\\pi(a \\mid x_t, z)$ with a **stakes-weighted soft-min cross-entropy**.
   The plain expected-regret loss collapses to "never alert".
4. Optional **safety shield**: never warn below the official NWS threshold rule.
""")
    st.subheader("Dataset")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Training hours", f"{len(train_df):,}", "2007-10 → 2022-12", delta_color="off")
    d2.metric("Evaluation hours", f"{len(eval_df):,}", "2023-01 → 2026-09", delta_color="off")
    d3.metric("Training floods", int(train_df["event_start"].sum()), "max: moderate", delta_color="off")
    d4.metric("Evaluation floods", int(eval_df["event_start"].sum()), "incl. 4 Jul 2025 major", delta_color="off")


def page_compare():
    st.title("Compare methods")
    st.caption("Exact counterfactual replay on the held-out years 2023–2026 under the paper's default cost model.")
    _, eval_df = load_data()
    cfg = ExoConfig()
    runs = evaluate_everything(cfg_to_key(cfg))
    table = results_frame(runs, cfg)

    best_rl = table[table["Type"] == "Offline RL"].iloc[0]
    ours = table[table["key"] == "exo_hql_shielded"].iloc[0]
    rule = table[table["key"] == "rule_based_4level"].iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Exo-HQL + shield", f"{ours['Total cost']:.0f}")
    m2.metric("Best rule (4-level)", f"{rule['Total cost']:.0f}")
    m3.metric(f"Best offline RL ({best_rl['Method']})", f"{best_rl['Total cost']:.0f}")
    m4.metric("Hindsight oracle (lower bound)", f"{table.loc[table['key'] == 'hindsight_oracle', 'Total cost'].iloc[0]:.0f}")

    st.subheader("Cost breakdown")
    st.altair_chart(cost_breakdown_chart(table), width="stretch")
    st.caption("Offline RL and the NWS logs are dominated by false alarms (blue). The best policies are "
               "dominated by onset cost (red), i.e. missing lead time that gauge readings cannot provide.")

    st.subheader("Results table")
    show = table.drop(columns=["key"])
    st.dataframe(show.style.format({c: "{:.1f}" for c in show.columns if show[c].dtype.kind == "f"}),
                 width="stretch", hide_index=True)

    st.subheader("Public trust over time")
    picks = st.multiselect("Methods", list(runs), format_func=LABELS.get,
                           default=["logged_nws", "cost_calibrated_iql", "discrete_iql", "exo_hql_shielded"])
    if picks:
        st.altair_chart(trust_chart(runs, picks, eval_df, cfg), width="stretch")
    st.caption("Each wasted alert hour adds credibility debt; trust T(k) is the share of people who still act on a warning.")
    st.info("Exo-HQL rows use the pretrained seed-0 models; the paper reports the mean over 3 seeds "
            "(634 ± 13 and 433 ± 16).")


def page_events():
    st.title("Flood event explorer")
    _, eval_df = load_data()
    cfg = ExoConfig()
    runs = evaluate_everything(cfg_to_key(cfg))
    trace = build_exo_trace(eval_df, cfg)
    onsets = np.flatnonzero(trace["onset_peak"] > 0)
    labels = {int(i): f"{pd.Timestamp(eval_df['datetime'].iloc[i]):%d %b %Y %H:%M} UTC  (peak: "
              f"{['', 'minor', 'moderate', 'major'][trace['onset_peak'][i]]})" for i in onsets}

    c1, c2 = st.columns([2, 1])
    onset = c1.selectbox("Flood event", list(labels), format_func=labels.get,
                         index=next(k for k, i in enumerate(labels) if pd.Timestamp(eval_df['datetime'].iloc[i]).year == 2025))
    window = c2.slider("Hours either side of onset", 12, 72, 36, step=6)
    names = st.multiselect("Methods to show", list(runs), format_func=LABELS.get,
                           default=["hindsight_oracle", "logged_nws", "rule_based_4level",
                                    "cost_calibrated_iql", "exo_hql_shielded", "exo_hql"])
    if not names:
        return
    top, bottom = event_charts(eval_df, runs, names, onset, window)
    st.altair_chart(top, width="stretch")
    st.altair_chart(bottom, width="stretch")

    rows = []
    for n in names:
        ev = next(e for e in runs[n]["result"]["events"] if pd.Timestamp(e["datetime"]) == pd.Timestamp(eval_df["datetime"].iloc[onset]))
        rows.append({"Method": LABELS[n], "Lead time (h)": ev["lead_hours"], "Trust at onset": ev["trust"],
                     "Alert at onset": LEVELS[ev["action_at_onset"]], "Onset cost": ev["onset_cost"]})
    st.dataframe(pd.DataFrame(rows).style.format({"Trust at onset": "{:.2f}", "Onset cost": "{:.1f}"}),
                 width="stretch", hide_index=True)


def page_evaluate():
    st.title("Evaluate Exo-HQL")
    st.markdown("Change the cost model and the method's options, then train Exo-HQL on 2007–2022 and "
                "evaluate it on 2023–2026 against every baseline **under the same cost model**.")
    train_df, eval_df = load_data()

    with st.form("eval"):
        st.subheader("Cost model")
        c1, c2, c3 = st.columns(3)
        onset_scale = c1.select_slider("Onset-cost scale", [0.5, 1.0, 1.5, 2.0], value=1.0,
                                       help="Multiplies the one-off cost of an under-prepared flood onset (10/40/200 by peak severity).")
        switch = c2.select_slider("Switching cost λ_sw", [0.0, 0.25, 0.5, 1.0, 2.0], value=0.5,
                                  help="Cost per alert level changed; discourages flapping.")
        lead = c3.slider("Lead time for full preparedness (h)", 3, 24, 12)
        c4, c5 = st.columns(2)
        min_trust = c4.slider("Minimum trust T_min", 0.0, 1.0, 0.3, 0.05,
                              help="Compliance when credibility debt is maxed out. 1.0 = no cry-wolf effect.")
        recovery = c5.select_slider("Trust recovery: 1 unit every … hours", [24, 48, 96, 192], value=96)

        st.subheader("Exo-HQL options")
        o1, o2, o3 = st.columns(3)
        shield = o1.checkbox("Safety shield", value=True,
                             help="Never issue a lower alert than the official NWS threshold rule.")
        alert_state = o2.checkbox("Use alert state", value=True,
                                  help="Let the policy see its own lead time and the public's trust level.")
        seed = o3.number_input("Seed", 0, 99, 0)
        submitted = st.form_submit_button("Train & evaluate", type="primary")

    cfg = replace(ExoConfig(), onset_cost=tuple(c * onset_scale for c in ExoConfig().onset_cost),
                  switch_cost=switch, lead_target_hours=lead, min_trust=min_trust, recovery_hours=recovery)
    key = cfg_to_key(cfg)
    if not submitted and "last_eval" not in st.session_state:
        st.info("Set the options and press **Train & evaluate**. Training takes about a minute on a laptop CPU "
                "(hindsight DP over 132k hours, then 5 epochs over 2.1M samples). Repeat runs are cached.")
        return
    if submitted:
        st.session_state["last_eval"] = (key, shield, alert_state, int(seed))
    key, shield, alert_state, seed = st.session_state["last_eval"]
    cfg = cfg_from_key(key)

    with st.status("Training Exo-HQL...", expanded=False) as status:
        status.write("Solving hindsight DP on the training years and fitting the policy")
        model = train_hql(key, shield, alert_state, seed)
        status.write("Replaying all methods on 2023–2026")
        trace = build_exo_trace(eval_df, cfg)
        ours = run_policy(model.as_closed_loop(eval_df), trace, cfg)
        runs = dict(evaluate_everything(key))
        status.update(label="Done", state="complete")

    name = "exo_hql_custom"
    LABELS[name] = "Exo-HQL (your run)"
    GROUP[name] = "Ours"
    runs = {k: v for k, v in runs.items() if not k.startswith("exo_hql")}
    runs[name] = ours
    table = results_frame(runs, cfg)
    me = table[table["key"] == name].iloc[0]
    oracle = table[table["key"] == "hindsight_oracle"].iloc[0]
    rule = table[table["key"] == "rule_based_4level"].iloc[0]
    best_rl = table[table["Type"] == "Offline RL"].iloc[0]

    st.subheader("Your Exo-HQL run")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total cost", f"{me['Total cost']:.1f}")
    m2.metric("Regret to oracle", f"{me['Regret to oracle']:.1f}")
    m3.metric("vs 4-level rule", f"{me['Total cost'] - rule['Total cost']:+.1f}", delta_color="inverse")
    m4.metric(f"vs best offline RL ({best_rl['Method']})", f"{me['Total cost'] - best_rl['Total cost']:+.1f}",
              delta_color="inverse")
    st.caption(f"Rank {int(table.index[table['key'] == name][0])} of {len(table) - 1} methods "
               f"(oracle excluded; oracle cost {oracle['Total cost']:.1f}). Lead time per event: "
               f"{me['Lead time per event (h)']} h. Alert hours: {me['Alert hours']}.")

    st.altair_chart(cost_breakdown_chart(table), width="stretch")
    show = table.drop(columns=["key"])
    st.dataframe(show.style.format({c: "{:.1f}" for c in show.columns if show[c].dtype.kind == "f"}),
                 width="stretch", hide_index=True)

    onsets = np.flatnonzero(trace["onset_peak"] > 0)
    july_2025 = next(i for i in onsets if pd.Timestamp(eval_df["datetime"].iloc[i]).year == 2025)
    st.subheader("Your run on the 4 July 2025 flood")
    top, bottom = event_charts(eval_df, runs, ["hindsight_oracle", "rule_based_4level", "logged_nws", name],
                               july_2025, 36)
    st.altair_chart(top, width="stretch")
    st.altair_chart(bottom, width="stretch")


PAGES = {"Overview": page_overview, "Compare methods": page_compare,
         "Flood event explorer": page_events, "Evaluate Exo-HQL": page_evaluate}
with st.sidebar:
    st.header("Exo-HQL demo")
    page = st.radio("Page", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("Counterfactually Exact Offline Learning for Flood Early Warning: "
               "An Exogenous MDP View with Hindsight Q-Distillation")
    st.caption("Hemanthkumar Sathish · Manda Sai Nikhil · VIT")
PAGES[page]()
