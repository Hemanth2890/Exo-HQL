import pandas as pd
df = pd.read_csv("kerrville_mdp_dataset.csv", parse_dates=["datetime"])
print(len(df), df.datetime.min(), df.datetime.max())
print(df.severity_outcome.value_counts().sort_index())
print(df.alert_action.value_counts().sort_index())
print(df.event_start.sum(), "event onsets")
print(df.groupby(df.datetime.dt.year).severity_outcome.max())
print(df[["gage_height_ft", "discharge_cfs"]].describe())