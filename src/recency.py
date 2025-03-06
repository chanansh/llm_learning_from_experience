import numpy as np
def get_recency(df):
    risky = df["Choice"]==1
    high = df["Payoff"]==df["High"]
    next_choice = df["Choice"].shift(-1)
    risky_given_previous_was_risky = next_choice[risky]
    previous_risky_was_high = high[risky]
    p_risky_given_previous_risky = risky_given_previous_was_risky.groupby(previous_risky_was_high).mean()
    p_risky_given_previous_risky = p_risky_given_previous_risky.reindex([False, True], fill_value=np.nan)
    recency_df = p_risky_given_previous_risky.to_frame().T
    recency_df.columns = ["p risky given previous low", "p risky given previous high"]
    return recency_df

def get_mean_recency(df):
    recency_df = df.groupby(["Problem", "Id"]).apply(get_recency, include_groups=False)
    recency_df = recency_df.groupby("Problem").mean()
    recency_df['recency diff'] = recency_df.diff(axis=1).iloc[:,1]
    df_agg = df[["Problem", "Phigh", "Medium", "High", "Low"]].drop_duplicates()
    recency_df = df_agg.merge(recency_df, on="Problem")
    return recency_df
