
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

OUTPUT_DIR = Path("outputs/sensitivity_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ELIGIBILITY_THRESHOLDS = [
    (100, 8),
    (150, 10),
    (200, 12),
]

CONFIDENCE_WEIGHTS = [0.00, 0.15, 0.30]


# --------------------------------------------------
# LOAD AND PREPARE DATA
# Same preparation as pressure_rankings.py
# --------------------------------------------------

ipl = pd.read_csv("data/processed/ipl_clean_deliveries.csv")
t20 = pd.read_csv("data/processed/t20i_clean_deliveries.csv")

ipl["competition"] = "IPL"
t20["competition"] = "T20I"

df = pd.concat([ipl, t20], ignore_index=True)

df["pressure"] = (
    (df["phase"].astype(str) == "Death")
    | (
        pd.to_numeric(
            df["wickets_lost_before"],
            errors="coerce"
        ) >= 5
    )
)

df["innings_key"] = (
    df["competition"].astype(str)
    + "_"
    + df["match_id"].astype(str)
    + "_"
    + df["innings"].astype(str)
)

print("Combined deliveries:", len(df))
print("Pressure deliveries:", int(df["pressure"].sum()))


# --------------------------------------------------
# COMMON PERCENTILE FUNCTION
# --------------------------------------------------

def pct(series):
    return series.rank(pct=True).fillna(0.5)


# --------------------------------------------------
# BATTER STATISTICS
# Same logic as pressure_rankings.py
# --------------------------------------------------

def batter_stats(data, pressure_value):

    x = data[data["pressure"] == pressure_value].copy()

    x["is_ball_faced"] = (
        x["is_ball_faced"].fillna(False).astype(bool)
    )

    x["runs"] = pd.to_numeric(
        x["runs"], errors="coerce"
    ).fillna(0)

    s = x.groupby("striker_id").agg(
        player_name=("striker_name", "first"),
        runs=("runs", "sum"),
        balls=("is_ball_faced", "sum"),
        innings=("innings_key", "nunique"),
    )

    d = (
        x[x["dismissed_player_id"].notna()]
        .groupby("dismissed_player_id")["is_wicket"]
        .sum()
    )

    s["dismissals"] = d.reindex(s.index).fillna(0)

    s["strike_rate"] = (
        100 * s["runs"] / s["balls"].replace(0, pd.NA)
    )

    s["average"] = (
        s["runs"] / s["dismissals"].replace(0, pd.NA)
    )

    return s


# --------------------------------------------------
# BOWLER STATISTICS
# Same logic as pressure_rankings.py
# --------------------------------------------------

def bowler_stats(data, pressure_value):

    x = data[data["pressure"] == pressure_value].copy()

    for c in ["bowler_runs_conceded", "total_runs"]:
        x[c] = pd.to_numeric(
            x[c], errors="coerce"
        ).fillna(0)

    x["bowler_wicket"] = (
        x["bowler_wicket"].fillna(False).astype(int)
    )

    s = x.groupby("bowler_id").agg(
        player_name=("bowler_name", "first"),
        runs_conceded=("bowler_runs_conceded", "sum"),
        deliveries=("match_id", "size"),
        wickets=("bowler_wicket", "sum"),
        zero_run_deliveries=(
            "total_runs",
            lambda z: (z == 0).sum()
        ),
        innings=("innings_key", "nunique"),
    )

    s["runs_per_delivery"] = (
        s["runs_conceded"] / s["deliveries"]
    )

    s["wicket_rate"] = (
        s["wickets"] / s["deliveries"]
    )

    s["zero_run_pct"] = (
        100 * s["zero_run_deliveries"] / s["deliveries"]
    )

    s["average"] = (
        s["runs_conceded"]
        / s["wickets"].replace(0, pd.NA)
    )

    return s


# --------------------------------------------------
# BUILD BATTER RANKINGS
# --------------------------------------------------

def build_batter_rankings(
    data,
    min_balls,
    min_innings,
    confidence_weight
):

    bp = batter_stats(data, True).add_suffix("_pressure")
    bn = batter_stats(data, False).add_suffix("_normal")

    bat = bp.join(bn, how="inner")

    bat["sr_improvement"] = (
        bat["strike_rate_pressure"]
        - bat["strike_rate_normal"]
    )

    bat["avg_improvement"] = (
        bat["average_pressure"]
        - bat["average_normal"]
    )

    bat = bat[
        (bat["balls_pressure"] >= min_balls)
        & (bat["innings_pressure"] >= min_innings)
    ].copy()

    if bat.empty:
        return bat

    bat["raw_score"] = (
        0.35 * pct(bat["strike_rate_pressure"])
        + 0.25 * pct(bat["average_pressure"])
        + 0.25 * pct(bat["sr_improvement"])
        + 0.15 * pct(bat["avg_improvement"])
    )

    bat["confidence"] = (
        bat["balls_pressure"] / (bat["balls_pressure"] + 100)
    ) * (
        bat["innings_pressure"] / (bat["innings_pressure"] + 5)
    )

    bat["rating"] = (
        100
        * bat["raw_score"]
        * (
            (1 - confidence_weight)
            + confidence_weight * bat["confidence"]
        )
    )

    bat = bat.sort_values(
        "rating", ascending=False
    ).reset_index()

    bat.insert(0, "rank", range(1, len(bat) + 1))

    return bat


# --------------------------------------------------
# BUILD BOWLER RANKINGS
# --------------------------------------------------

def build_bowler_rankings(
    data,
    min_deliveries,
    min_innings,
    confidence_weight
):

    wp = bowler_stats(data, True).add_suffix("_pressure")
    wn = bowler_stats(data, False).add_suffix("_normal")

    bowl = wp.join(wn, how="inner")

    bowl["run_suppression"] = (
        bowl["runs_per_delivery_normal"]
        - bowl["runs_per_delivery_pressure"]
    )

    bowl["wicket_rate_improvement"] = (
        bowl["wicket_rate_pressure"]
        - bowl["wicket_rate_normal"]
    )

    bowl = bowl[
        (bowl["deliveries_pressure"] >= min_deliveries)
        & (bowl["innings_pressure"] >= min_innings)
    ].copy()

    if bowl.empty:
        return bowl

    bowl["raw_score"] = (
        0.25 * pct(-bowl["runs_per_delivery_pressure"])
        + 0.20 * pct(bowl["wicket_rate_pressure"])
        + 0.15 * pct(bowl["zero_run_pct_pressure"])
        + 0.15 * pct(-bowl["average_pressure"])
        + 0.15 * pct(bowl["run_suppression"])
        + 0.10 * pct(bowl["wicket_rate_improvement"])
    )

    bowl["confidence"] = (
        bowl["deliveries_pressure"]
        / (bowl["deliveries_pressure"] + 100)
    ) * (
        bowl["innings_pressure"]
        / (bowl["innings_pressure"] + 5)
    )

    bowl["rating"] = (
        100
        * bowl["raw_score"]
        * (
            (1 - confidence_weight)
            + confidence_weight * bowl["confidence"]
        )
    )

    bowl = bowl.sort_values(
        "rating", ascending=False
    ).reset_index()

    bowl.insert(0, "rank", range(1, len(bowl) + 1))

    return bowl


# --------------------------------------------------
# RUN ELIGIBILITY SENSITIVITY ANALYSIS
# Using 85% raw score + 15% confidence
# --------------------------------------------------

threshold_results = {}

print("\n" + "=" * 65)
print("ELIGIBILITY THRESHOLD SENSITIVITY")
print("=" * 65)

for min_sample, min_innings in ELIGIBILITY_THRESHOLDS:

    bat = build_batter_rankings(
        df, min_sample, min_innings, 0.15
    )

    bowl = build_bowler_rankings(
        df, min_sample, min_innings, 0.15
    )

    threshold_results[(min_sample, min_innings)] = (
        bat, bowl
    )

    bat.to_csv(
        OUTPUT_DIR / f"batters_{min_sample}_{min_innings}.csv",
        index=False
    )

    bowl.to_csv(
        OUTPUT_DIR / f"bowlers_{min_sample}_{min_innings}.csv",
        index=False
    )

    print(
        f"\nThreshold: {min_sample} sample / "
        f"{min_innings} pressure innings"
    )

    print("Eligible batters:", len(bat))
    print("Eligible bowlers:", len(bowl))

    print("\nBatter top 10:")

    print(
        bat[["rank", "player_name_pressure", "rating"]]
        .head(10).to_string(index=False)
    )

    print("\nBowler top 10:")

    print(
        bowl[["rank", "player_name_pressure", "rating"]]
        .head(10).to_string(index=False)
    )


# --------------------------------------------------
# TOP 10 OVERLAP BETWEEN THRESHOLDS
# --------------------------------------------------

print("\n" + "=" * 65)
print("TOP 10 OVERLAP BETWEEN ELIGIBILITY THRESHOLDS")
print("=" * 65)


def show_overlap(results, role):

    for i in range(len(ELIGIBILITY_THRESHOLDS) - 1):

        t1 = ELIGIBILITY_THRESHOLDS[i]
        t2 = ELIGIBILITY_THRESHOLDS[i + 1]

        first = results[t1][0 if role == "batter" else 1]
        second = results[t2][0 if role == "batter" else 1]

        name_col = "player_name_pressure"

        names1 = set(first.head(10)[name_col].dropna())
        names2 = set(second.head(10)[name_col].dropna())

        common = names1.intersection(names2)

        print(
            f"{role.title()}: {t1} vs {t2}: "
            f"{len(common)}/10 common players"
        )

        print("Common:", ", ".join(sorted(common)))


show_overlap(threshold_results, "batter")
show_overlap(threshold_results, "bowler")


# --------------------------------------------------
# CONFIDENCE WEIGHT SENSITIVITY AT 150 / 10
# --------------------------------------------------

print("\n" + "=" * 65)
print("CONFIDENCE WEIGHT SENSITIVITY AT 150 / 10")
print("=" * 65)

for weight in CONFIDENCE_WEIGHTS:

    bat = build_batter_rankings(df, 150, 10, weight)
    bowl = build_bowler_rankings(df, 150, 10, weight)

    weight_label = int(weight * 100)

    bat.to_csv(
        OUTPUT_DIR / f"batters_confidence_{weight_label}.csv",
        index=False
    )

    bowl.to_csv(
        OUTPUT_DIR / f"bowlers_confidence_{weight_label}.csv",
        index=False
    )

    print(f"\nConfidence weight: {weight_label}%")

    print("Batter top 10:")

    print(
        bat[["rank", "player_name_pressure", "rating"]]
        .head(10).to_string(index=False)
    )

    print("\nBowler top 10:")

    print(
        bowl[["rank", "player_name_pressure", "rating"]]
        .head(10).to_string(index=False)
    )


# --------------------------------------------------
# COMPARE RECOMPUTED 150/10 WITH SAVED RANKINGS
# --------------------------------------------------

print("\n" + "=" * 65)
print("COMPARISON WITH CURRENT SAVED RANKINGS")
print("=" * 65)

baseline_bat, baseline_bowl = threshold_results[(150, 10)]


def compare_saved(recomputed, saved_path, id_column, role):

    if not Path(saved_path).exists():
        print(f"\nSaved {role} CSV not found: {saved_path}")
        return

    saved = pd.read_csv(saved_path)

    if id_column not in saved.columns:
        print(f"\nMissing ID column in saved {role} CSV.")
        return

    current = recomputed[["rank", id_column, "rating"]].copy()

    comparison = current.merge(
        saved[["rank", id_column, "rating"]],
        on=id_column,
        how="outer",
        suffixes=("_recomputed", "_saved"),
        indicator=True
    )

    comparison["rating_difference"] = (
        comparison["rating_recomputed"]
        - comparison["rating_saved"]
    )

    comparison.to_csv(
        OUTPUT_DIR / f"{role}_saved_comparison.csv",
        index=False
    )

    print(f"\n{role.title()} comparison:")

    print("Matching IDs:", int(
        (comparison["_merge"] == "both").sum()
    ))

    print("Only in recomputed:", int(
        (comparison["_merge"] == "left_only").sum()
    ))

    print("Only in saved:", int(
        (comparison["_merge"] == "right_only").sum()
    ))

    matched = comparison[
        comparison["_merge"] == "both"
    ]

    if not matched.empty:
        max_difference = matched[
            "rating_difference"
        ].abs().max()

        print(
            "Maximum absolute rating difference:",
            max_difference
        )


compare_saved(
    baseline_bat,
    "outputs/combined_batter_pressure_rankings.csv",
    "striker_id",
    "batter"
)

compare_saved(
    baseline_bowl,
    "outputs/combined_bowler_pressure_rankings.csv",
    "bowler_id",
    "bowler"
)


print("\n" + "=" * 65)
print("SENSITIVITY ANALYSIS COMPLETED")
print("Output folder:", OUTPUT_DIR)
print("=" * 65)
