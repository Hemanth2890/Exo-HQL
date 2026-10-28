"""
Plot the model comparison charts (missed-event rate, false-alarm rate, mean
lead time) from eval_results.csv -- produces PNGs to drop straight into the
paper/report.

Requires: pip install matplotlib pandas --break-system-packages
Run after evaluate_models.py (needs eval_results.csv in the same folder).
"""

import pandas as pd
import matplotlib.pyplot as plt

MODEL_LABELS = {
    "rule_based_threshold": "Rule-based\nthreshold",
    "behavior_cloning": "Behavior\ncloning",
    "discrete_cql": "Discrete\nCQL",
    "discrete_iql": "Discrete\nIQL",
    "cost_calibrated_iql": "Cost-calibrated\nIQL",
}

# same fixed categorical order/colors used everywhere else in the project
COLORS = {
    "rule_based_threshold": "#2a78d6",
    "behavior_cloning": "#eb6834",
    "discrete_cql": "#1baf7a",
    "discrete_iql": "#eda100",
    "cost_calibrated_iql": "#e87ba4",
}


def bar_panel(ax, df, col, title, ylabel, fmt):
    models = df["model"]
    values = df[col]
    colors = [COLORS[m] for m in models]
    labels = [MODEL_LABELS[m] for m in models]

    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.margins(y=0.15)

    for bar, v in zip(bars, values):
        ax.annotate(
            fmt(v),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )


def plot_all(path_in: str = "eval_results.csv", path_out: str = "model_comparison.png"):
    df = pd.read_csv(path_in)

    lead_col = "mean_lead_time_hours" if "mean_lead_time_hours" in df.columns else "mean_lead_time_days"
    unit = "hours" if lead_col.endswith("hours") else "days"
    cost_col = "total_expected_cost_leadcredit" if "total_expected_cost_leadcredit" in df.columns else "total_expected_cost"

    panels = [
        ("missed_event_rate", "Missed-event rate", "Share of real events missed",
         lambda v: f"{v*100:.1f}%"),
        ("false_alarm_rate", "False-alarm rate", "Share of quiet hours w/ active alert",
         lambda v: f"{v*100:.2f}%"),
        (lead_col, "Mean lead time", f"Warning lead time ({unit}, caught events only)",
         lambda v: f"{v:.2f}{unit[0]}"),
        (cost_col, "Total expected cost", "Lead-credited cost, COST_MATRIX summed over eval set",
         lambda v: f"{v:.0f}"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, (col, title, ylabel, fmt) in zip(axes, panels):
        bar_panel(ax, df, col, title, ylabel, fmt)

    fig.suptitle(
        "Flood alert model comparison -- Kerrville, TX held-out evaluation",
        fontsize=13, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    fig.savefig(path_out, dpi=200, bbox_inches="tight")
    print(f"Saved {path_out}")

    # also save each panel individually, useful when the paper wants them
    # as separate figures rather than one combined image
    single_names = ["missed_event_rate.png", "false_alarm_rate.png", "mean_lead_time.png", "total_expected_cost.png"]
    for (col, title, ylabel, fmt), name in zip(panels, single_names):
        single_fig, single_ax = plt.subplots(figsize=(5.5, 4.5))
        bar_panel(single_ax, df, col, title, ylabel, fmt)
        single_fig.tight_layout()
        single_fig.savefig(name, dpi=200, bbox_inches="tight")
        plt.close(single_fig)
        print(f"Saved {name}")

    plt.close(fig)


if __name__ == "__main__":
    plot_all()