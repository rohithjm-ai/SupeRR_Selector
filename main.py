import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")


def analyse_batters(file_name):

    print("\n" + "=" * 65)
    print(f"BATTER ANALYSIS: {file_name}")
    print("=" * 65)

    # Identify the corresponding match file
    match_file = file_name.replace("_deliveries.csv", "_matches.csv")

    df = pd.read_csv(DATA_DIR / file_name)
    matches = pd.read_csv(DATA_DIR / match_file)

    # Keep normal innings only
    df = df[df["innings"].isin([1, 2])].copy()

    # Calculate total runs including extras
    df["total_runs"] = (
        df["runs"].fillna(0)
        + df["extras1_runs"].fillna(0)
        + df["extras_2_runs"].fillna(0)
    )

    # Calculate actual innings scores
    scores = (
        df.groupby(["match_id", "innings"])["total_runs"]
        .sum()
        .unstack()
        .rename(columns={1: "first_score", 2: "second_score"})
        .reset_index()
    )

    # Merge scores with match information
    matches = matches.merge(scores, on="match_id", how="left")

    # Identify matches to exclude
    result_text = matches["result"].fillna("").astype(str)

    abandoned_or_no_result = result_text.str.contains(
        r"abandoned|no result", case=False, regex=True
    )

    dls_matches = result_text.str.contains(
        r"d/l|dls|duckworth|lewis", case=False, regex=True
    )

    missing_scores = matches["first_score"].isna() | matches["second_score"].isna()

    revised_target_matches = matches["revised_target"].notna()

    # Keep only suitable matches
    valid_matches = matches[
        ~(
            abandoned_or_no_result
            | dls_matches
            | missing_scores
            | revised_target_matches
        )
    ]

    valid_match_ids = valid_matches["match_id"]

    clean_df = df[df["match_id"].isin(valid_match_ids)].copy()

    print("\nMATCH CLEANING SUMMARY")
    print("Original matches:", len(matches))
    print("Clean matches:", len(valid_match_ids))
    print("Excluded matches:", len(matches) - len(valid_match_ids))
    print("Clean deliveries:", len(clean_df))

    # Identify innings phase
    clean_df["phase"] = pd.cut(
        clean_df["overid"], bins=[0, 6, 15, 20], labels=["Powerplay", "Middle", "Death"]
    )

    # Wickets lost before each delivery
    clean_df["wickets_lost_before"] = clean_df.groupby(["match_id", "innings"])[
        "is_wicket"
    ].transform(lambda x: x.fillna(False).astype(int).cumsum().shift(fill_value=0))

    # ---------------------------------------------------------
    # BOWLER ANALYSIS
    # ---------------------------------------------------------

    # Normalize extras type
    clean_df["extras_type_clean"] = (
        clean_df["extras_type"].fillna("").str.strip().str.lower()
    )

    # Runs charged to the bowler:
    # Batter runs + wide extras + no-ball (Overstep) extras.
    # Byes and legbyes are not charged to the bowler.
    charged_extras = clean_df["extras_type_clean"].isin(["wide", "overstep"])

    clean_df["bowler_runs_conceded"] = clean_df["runs"].fillna(0) + (
        clean_df["extras1_runs"].fillna(0) + clean_df["extras_2_runs"].fillna(0)
    ).where(charged_extras, 0)

    # Only these dismissal types are credited to the bowler.
    bowler_credited_types = ["caught", "bowled", "lbw", "stump", "hit wicket"]

    clean_df["bowler_wicket"] = clean_df["is_wicket"].fillna(False).astype(
        bool
    ) & clean_df["dismissal_type"].fillna("").str.strip().str.lower().isin(
        bowler_credited_types
    )

    def bowler_summary(subset, analysis_name, min_deliveries):

        # Aggregate bowler statistics
        bowling = (
            subset.groupby(["bowler_id", "bowler_name"])
            .agg(
                runs_conceded=("bowler_runs_conceded", "sum"),
                recorded_deliveries=("bowler_runs_conceded", "size"),
                # A dot delivery must have zero total runs
                # scored by the batting team.
                dot_deliveries=("total_runs", lambda x: (x == 0).sum()),
            )
            .reset_index()
            .rename(columns={"bowler_id": "player_id", "bowler_name": "player_name"})
        )

        # Count bowler-credited wickets
        wickets = (
            subset[subset["bowler_wicket"]]
            .groupby("bowler_id")
            .size()
            .reset_index(name="wickets")
            .rename(columns={"bowler_id": "player_id"})
        )

        # Count innings in which the bowler delivered
        innings_count = (
            subset[["bowler_id", "bowler_name", "match_id", "innings"]]
            .drop_duplicates()
            .groupby(["bowler_id", "bowler_name"])
            .size()
            .reset_index(name="innings_count")
            .rename(columns={"bowler_id": "player_id", "bowler_name": "player_name"})
        )

        # Merge wickets and innings counts
        bowling = bowling.merge(wickets, on="player_id", how="left")

        bowling = bowling.merge(
            innings_count, on=["player_id", "player_name"], how="left"
        )

        bowling["wickets"] = bowling["wickets"].fillna(0)

        # Bowling average = runs conceded / wickets
        bowling["bowling_average"] = bowling["runs_conceded"] / bowling[
            "wickets"
        ].replace(0, pd.NA)

        # Percentage of recorded deliveries with zero total runs
        bowling["dot_delivery_pct"] = (
            bowling["dot_deliveries"] / bowling["recorded_deliveries"] * 100
        )

        # Save all bowler statistics
        output_name = file_name.replace(
            "_deliveries.csv", f"_{analysis_name}_bowlers.csv"
        )

        bowling.to_csv(DATA_DIR / output_name, index=False)

        # Minimum sample-size filter
        qualified = bowling[
            (bowling["innings_count"] >= 15)
            & (bowling["recorded_deliveries"] >= min_deliveries)
        ].copy()

        columns = [
            "player_name",
            "innings_count",
            "recorded_deliveries",
            "runs_conceded",
            "wickets",
            "bowling_average",
            "dot_delivery_pct",
        ]

        print(f"\n{analysis_name.upper()} - QUALIFIED BOWLERS")
        print("Minimum innings: 15")
        print("Minimum recorded deliveries:", min_deliveries)

        print("\nTOP 10 BY WICKETS")

        print(
            qualified.sort_values(
                ["wickets", "bowling_average"],
                ascending=[False, True],
                na_position="last",
            )[columns]
            .head(10)
            .round(2)
            .to_string(index=False)
        )

        print("\nTOP 10 BY BOWLING AVERAGE")

        print(
            qualified.sort_values(
                ["bowling_average", "dot_delivery_pct"],
                ascending=[True, False],
                na_position="last",
            )[columns]
            .head(10)
            .round(2)
            .to_string(index=False)
        )

        print("\nTOP 10 BY DOT-DELIVERY PERCENTAGE")

        print(
            qualified.sort_values(
                ["dot_delivery_pct", "bowling_average"],
                ascending=[False, True],
                na_position="last",
            )[columns]
            .head(10)
            .round(2)
            .to_string(index=False)
        )

        # Save qualified bowler statistics
        qualified_output = file_name.replace(
            "_deliveries.csv", f"_{analysis_name}_qualified_bowlers.csv"
        )

        qualified.to_csv(DATA_DIR / qualified_output, index=False)

        print("Saved:", output_name)
        print("Saved:", qualified_output)

    # Overall bowling performance
    bowler_summary(clean_df, "overall", 300)

    # Death-over bowling performance
    bowler_summary(clean_df[clean_df["phase"] == "Death"], "death_pressure", 100)

    # Bowling when five or more wickets have fallen
    bowler_summary(
        clean_df[clean_df["wickets_lost_before"] >= 5], "wicket_pressure", 100
    )

    # Report unresolved other extras separately
    other_extras = clean_df[clean_df["extras_type_clean"] == "other"]

    print("\nUNCLASSIFIED OTHER EXTRAS")
    print("Deliveries:", len(other_extras))

    print(
        "Extra runs:",
        (
            other_extras["extras1_runs"].fillna(0)
            + other_extras["extras_2_runs"].fillna(0)
        ).sum(),
    )

    # ---------------------------------------------------------
    # BATTER ANALYSIS
    # ---------------------------------------------------------

    # Identify deliveries faced.
    # Wide deliveries are excluded from balls faced.
    clean_df["is_ball_faced"] = (
        clean_df["extras_type"].fillna("").str.lower().ne("wide")
    )

    # Identify boundaries
    clean_df["is_boundary"] = clean_df["is_four"].fillna(False).astype(bool) | clean_df[
        "is_six"
    ].fillna(False).astype(bool)

    # Batter runs and balls faced
    batting = (
        clean_df.groupby(["striker_id", "striker_name"])
        .agg(
            runs=("runs", "sum"),
            balls_faced=("is_ball_faced", "sum"),
            fours=("is_four", "sum"),
            sixes=("is_six", "sum"),
            boundary_runs=(
                "runs",
                lambda x: x[clean_df.loc[x.index, "is_boundary"]].sum(),
            ),
        )
        .reset_index()
        .rename(columns={"striker_id": "player_id", "striker_name": "player_name"})
    )

    # Count dismissals using dismissed_player_id
    dismissals = (
        clean_df[
            clean_df["is_wicket"].fillna(False)
            & clean_df["dismissed_player_id"].notna()
        ]
        .groupby("dismissed_player_id")
        .size()
        .reset_index(name="dismissals")
        .rename(columns={"dismissed_player_id": "player_id"})
    )

    batting = batting.merge(dismissals, on="player_id", how="left")

    batting["dismissals"] = batting["dismissals"].fillna(0)

    # Calculate batting metrics
    batting["batting_average"] = batting["runs"] / batting["dismissals"].replace(
        0, pd.NA
    )

    batting["strike_rate"] = (
        batting["runs"] / batting["balls_faced"].replace(0, pd.NA) * 100
    )

    batting["boundary_percentage"] = (
        batting["boundary_runs"] / batting["runs"].replace(0, pd.NA) * 100
    )

    # Calculate batter performance in a pressure situation
    def pressure_summary(subset, pressure_name):

        pressure_batting = (
            subset.groupby(["striker_id", "striker_name"])
            .agg(
                runs=("runs", "sum"),
                balls_faced=("is_ball_faced", "sum"),
                fours=("is_four", "sum"),
                sixes=("is_six", "sum"),
            )
            .reset_index()
            .rename(columns={"striker_id": "player_id", "striker_name": "player_name"})
        )

        # Count distinct batting innings
        innings_count = (
            subset[["striker_id", "striker_name", "match_id", "innings"]]
            .drop_duplicates()
            .groupby(["striker_id", "striker_name"])
            .size()
            .reset_index(name="innings_count")
            .rename(columns={"striker_id": "player_id", "striker_name": "player_name"})
        )

        # Count actual dismissals in this pressure situation
        pressure_dismissals = (
            subset[
                subset["is_wicket"].fillna(False)
                & subset["dismissed_player_id"].notna()
            ]
            .groupby("dismissed_player_id")
            .size()
            .reset_index(name="dismissals")
            .rename(columns={"dismissed_player_id": "player_id"})
        )

        pressure_batting = pressure_batting.merge(
            innings_count, on=["player_id", "player_name"], how="left"
        )

        pressure_batting = pressure_batting.merge(
            pressure_dismissals, on="player_id", how="left"
        )

        pressure_batting["dismissals"] = pressure_batting["dismissals"].fillna(0)

        pressure_batting["strike_rate"] = (
            pressure_batting["runs"]
            / pressure_batting["balls_faced"].replace(0, pd.NA)
            * 100
        )

        pressure_batting["batting_average"] = pressure_batting[
            "runs"
        ] / pressure_batting["dismissals"].replace(0, pd.NA)

        # Save original unfiltered pressure statistics
        output_name = file_name.replace(
            "_deliveries.csv", f"_{pressure_name}_batters.csv"
        )

        pressure_batting.to_csv(DATA_DIR / output_name, index=False)

        print("Saved:", output_name)

        # Require a minimum pressure-innings sample
        qualified = pressure_batting[
            (pressure_batting["innings_count"] >= 15)
            & (pressure_batting["balls_faced"] >= 100)
        ].copy()

        # Calculate batter runs in each individual pressure innings
        innings_runs = (
            subset.groupby(["striker_id", "striker_name", "match_id", "innings"])[
                "runs"
            ]
            .sum()
            .reset_index(name="pressure_runs")
        )

        # Measure consistency across pressure innings
        consistency = (
            innings_runs.groupby(["striker_id", "striker_name"])
            .agg(
                pressure_innings=("pressure_runs", "count"),
                innings_10_plus=("pressure_runs", lambda x: (x >= 10).sum()),
                innings_20_plus=("pressure_runs", lambda x: (x >= 20).sum()),
            )
            .reset_index()
            .rename(columns={"striker_id": "player_id", "striker_name": "player_name"})
        )

        # Calculate percentage of pressure innings with 10+ runs
        consistency["pct_innings_10_plus"] = (
            consistency["innings_10_plus"] / consistency["pressure_innings"] * 100
        )

        # Keep only the required consistency columns
        consistency = consistency[
            [
                "player_id",
                "player_name",
                "innings_10_plus",
                "innings_20_plus",
                "pct_innings_10_plus",
            ]
        ]

        # Add consistency metrics to qualified batters
        qualified = qualified.merge(
            consistency, on=["player_id", "player_name"], how="left"
        )

        print(
            f"\n{pressure_name.upper()} - QUALIFIED BATTERS "
            "(MINIMUM 15 INNINGS AND 100 BALLS)"
        )

        columns = [
            "player_name",
            "runs",
            "balls_faced",
            "innings_count",
            "dismissals",
            "batting_average",
            "strike_rate",
            "innings_10_plus",
            "innings_20_plus",
            "pct_innings_10_plus",
        ]

        print("\nTOP 10 BY BATTING AVERAGE")

        print(
            qualified.sort_values(["batting_average", "strike_rate"], ascending=False)[
                columns
            ]
            .head(10)
            .round(2)
            .to_string(index=False)
        )

        print("\nTOP 10 BY STRIKE RATE")

        print(
            qualified.sort_values(["strike_rate", "batting_average"], ascending=False)[
                columns
            ]
            .head(10)
            .round(2)
            .to_string(index=False)
        )

        print("\nTOP 10 BY 10+ PRESSURE INNINGS PERCENTAGE")

        print(
            qualified.sort_values(
                ["pct_innings_10_plus", "batting_average"], ascending=False
            )[columns]
            .head(10)
            .round(2)
            .to_string(index=False)
        )

        # Save qualified pressure statistics
        qualified_output = file_name.replace(
            "_deliveries.csv", f"_{pressure_name}_qualified.csv"
        )

        qualified.to_csv(DATA_DIR / qualified_output, index=False)

        print("Saved:", qualified_output)

    # Pressure situation 1: Death overs (16-20)
    death_pressure = clean_df[clean_df["phase"] == "Death"]

    pressure_summary(death_pressure, "death_pressure")

    # Pressure situation 2: Five or more wickets lost
    wicket_pressure = clean_df[clean_df["wickets_lost_before"] >= 5]

    pressure_summary(wicket_pressure, "wicket_pressure")

    # Sort by total runs for initial inspection
    batting = batting.sort_values("runs", ascending=False)

    print("\nTOP 15 BATTERS BY RUNS")

    print(
        batting[
            [
                "player_name",
                "runs",
                "balls_faced",
                "dismissals",
                "batting_average",
                "strike_rate",
                "fours",
                "sixes",
                "boundary_percentage",
            ]
        ]
        .head(15)
        .round(2)
        .to_string(index=False)
    )

    # Save clean dataset and batter statistics
    clean_df.to_csv(
        DATA_DIR / file_name.replace("_deliveries.csv", "_clean_deliveries.csv"),
        index=False,
    )

    batting.to_csv(
        DATA_DIR / file_name.replace("_deliveries.csv", "_batter_stats.csv"),
        index=False,
    )

    print("\nFILES SAVED SUCCESSFULLY")


if __name__ == "__main__":
    for file_name in ["ipl_deliveries.csv", "t20i_deliveries.csv"]:
        analyse_batters(file_name)
