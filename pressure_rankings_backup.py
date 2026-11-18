import pandas as pd
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)

ipl = pd.read_csv("data/processed/ipl_clean_deliveries.csv")
t20 = pd.read_csv("data/processed/t20i_clean_deliveries.csv")

ipl["competition"] = "IPL"
t20["competition"] = "T20I"
df = pd.concat([ipl, t20], ignore_index=True)

df["pressure"] = (
    (df["phase"].astype(str) == "Death") |
    (pd.to_numeric(df["wickets_lost_before"], errors="coerce") >= 5)
)
df["innings_key"] = (
    df["competition"].astype(str) + "_" +
    df["match_id"].astype(str) + "_" +
    df["innings"].astype(str)
)

def batter_stats(data, pressure_value):
    x = data[data["pressure"] == pressure_value].copy()
    x["is_ball_faced"] = x["is_ball_faced"].fillna(False).astype(bool)
    x["runs"] = pd.to_numeric(x["runs"], errors="coerce").fillna(0)
    s = x.groupby("striker_id").agg(
        player_name=("striker_name", "first"),
        runs=("runs", "sum"),
        balls=("is_ball_faced", "sum"),
        innings=("innings_key", "nunique")
    )
    d = x[x["dismissed_player_id"].notna()].groupby(
        "dismissed_player_id"
    )["is_wicket"].sum()
    s["dismissals"] = d.reindex(s.index).fillna(0)
    s["strike_rate"] = 100 * s["runs"] / s["balls"].replace(0, pd.NA)
    s["average"] = s["runs"] / s["dismissals"].replace(0, pd.NA)
    return s

bp = batter_stats(df, True).add_suffix("_pressure")
bn = batter_stats(df, False).add_suffix("_normal")
bat = bp.join(bn, how="inner")
bat["sr_improvement"] = bat["strike_rate_pressure"] - bat["strike_rate_normal"]
bat["avg_improvement"] = bat["average_pressure"] - bat["average_normal"]

bat = bat[
    (bat["balls_pressure"] >= 100) &
    (bat["innings_pressure"] >= 8)
].copy()

def pct(series):
    return series.rank(pct=True).fillna(0.5)

bat["raw_score"] = (
    0.35 * pct(bat["strike_rate_pressure"]) +
    0.25 * pct(bat["average_pressure"]) +
    0.25 * pct(bat["sr_improvement"]) +
    0.15 * pct(bat["avg_improvement"])
)
bat["confidence"] = (
    (bat["balls_pressure"] / (bat["balls_pressure"] + 100)) *
    (bat["innings_pressure"] / (bat["innings_pressure"] + 5))
)
bat["rating"] = 100 * bat["raw_score"] * (0.70 + 0.30 * bat["confidence"])
bat = bat.sort_values("rating", ascending=False).reset_index()
bat.insert(0, "rank", range(1, len(bat) + 1))
bat.to_csv("outputs/combined_batter_pressure_rankings.csv", index=False)

def bowler_stats(data, pressure_value):
    x = data[data["pressure"] == pressure_value].copy()
    for c in ["bowler_runs_conceded", "total_runs"]:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0)
    x["bowler_wicket"] = x["bowler_wicket"].fillna(False).astype(int)
    s = x.groupby("bowler_id").agg(
        player_name=("bowler_name", "first"),
        runs_conceded=("bowler_runs_conceded", "sum"),
        deliveries=("match_id", "size"),
        wickets=("bowler_wicket", "sum"),
        zero_run_deliveries=("total_runs", lambda z: (z == 0).sum()),
        innings=("innings_key", "nunique")
    )
    s["runs_per_delivery"] = s["runs_conceded"] / s["deliveries"]
    s["wicket_rate"] = s["wickets"] / s["deliveries"]
    s["zero_run_pct"] = 100 * s["zero_run_deliveries"] / s["deliveries"]
    s["average"] = s["runs_conceded"] / s["wickets"].replace(0, pd.NA)
    return s

wp = bowler_stats(df, True).add_suffix("_pressure")
wn = bowler_stats(df, False).add_suffix("_normal")
bowl = wp.join(wn, how="inner")

bowl["run_suppression"] = (
    bowl["runs_per_delivery_normal"] -
    bowl["runs_per_delivery_pressure"]
)
bowl["wicket_rate_improvement"] = (
    bowl["wicket_rate_pressure"] -
    bowl["wicket_rate_normal"]
)

bowl = bowl[
    (bowl["deliveries_pressure"] >= 100) &
    (bowl["innings_pressure"] >= 8)
].copy()

bowl["raw_score"] = (
    0.25 * pct(-bowl["runs_per_delivery_pressure"]) +
    0.20 * pct(bowl["wicket_rate_pressure"]) +
    0.15 * pct(bowl["zero_run_pct_pressure"]) +
    0.15 * pct(-bowl["average_pressure"]) +
    0.15 * pct(bowl["run_suppression"]) +
    0.10 * pct(bowl["wicket_rate_improvement"])
)
bowl["confidence"] = (
    (bowl["deliveries_pressure"] / (bowl["deliveries_pressure"] + 100)) *
    (bowl["innings_pressure"] / (bowl["innings_pressure"] + 5))
)
bowl["rating"] = 100 * bowl["raw_score"] * (0.70 + 0.30 * bowl["confidence"])
bowl = bowl.sort_values("rating", ascending=False).reset_index()
bowl.insert(0, "rank", range(1, len(bowl) + 1))
bowl.to_csv("outputs/combined_bowler_pressure_rankings.csv", index=False)

print("\nELIGIBILITY SUMMARY")
print("Combined deliveries:", len(df))
print("Pressure deliveries:", int(df["pressure"].sum()))
print("Eligible batters:", len(bat))
print("Eligible bowlers:", len(bowl))

print("\nTOP 15 BATTERS")
print(bat[["rank", "player_name_pressure", "rating", "runs_pressure",
           "balls_pressure", "innings_pressure", "strike_rate_pressure",
           "strike_rate_normal", "average_pressure",
           "average_normal"]].head(15).round(2).to_string(index=False))

print("\nTOP 15 BOWLERS")
print(bowl[["rank", "player_name_pressure", "rating", "wickets_pressure",
            "deliveries_pressure", "innings_pressure",
            "runs_per_delivery_pressure", "runs_per_delivery_normal",
            "average_pressure", "zero_run_pct_pressure"]]
      .head(15).round(2).to_string(index=False))

print("\nSaved:")
print("outputs/combined_batter_pressure_rankings.csv")
print("outputs/combined_bowler_pressure_rankings.csv")

