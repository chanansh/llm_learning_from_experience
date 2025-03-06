import pandas as pd


def get_primacy(df):
    try:
        first_risky = df[df["Choice"] == 1].iloc[0]
        first_risky_payout = first_risky["Payoff"]
        first_risky_payout_high = first_risky_payout == first_risky["High"]
        return first_risky_payout_high
    except IndexError:
        return None


def get_primacy_effect(est, p_high_criteria="(Phigh < 0.15) | (Phigh > 0.85)"):
    primacy = (
        est.groupby(["Id", "Problem"])
        .apply(get_primacy, include_groups=False)
        .reset_index(name="First Risky High")
    )
    est_primacy = pd.merge(est, primacy, on=["Id", "Problem"])
    criteria = est_primacy.eval(p_high_criteria)
    primacy_effect_per_problem = (
        est_primacy[criteria]
        .groupby(["Problem", "First Risky High", "Trial"])["Choice"]
        .mean()
        .reset_index()
    )
    primacy_effect = (
        primacy_effect_per_problem.groupby(["First Risky High", "Trial"])["Choice"]
        .mean()
        .reset_index()
    )
    return primacy_effect
