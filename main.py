
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")


def analyse_batters(file_name):

    print("\n" + "=" * 65)
    print(f"BATTER ANALYSIS: {file_name}")
    print("=" * 65)

    # Identify the corresponding match file
    match_file = file_name.replace(
        "_deliveries.csv",
        "_matches.csv"
    )

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
        .rename(columns={
            1: "first_score",
            2: "second_score"
        })
        .reset_index()
    )

    # Merge scores with match information
    matches = matches.merge(
        scores,
        on="match_id",
        how="left"
    )

    # Identify matches to exclude
    result_text = matches["result"].fillna("").astype(str)

    abandoned_or_no_result = result_text.str.contains(
        r"abandoned|no result",
        case=False,
        regex=True
    )

    dls_matches = result_text.str.contains(
        r"d/l|dls|duckworth|lewis",
        case=False,
        regex=True
    )

    missing_scores = (
        matches["first_score"].isna()
        | matches["second_score"].isna()
    )

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

    clean_df = df[
        df["match_id"].isin(valid_match_ids)
    ].copy()

    print("\nMATCH CLEANING SUMMARY")
    print("Original matches:", len(matches))
    print("Clean matches:", len(valid_match_ids))
    print("Excluded matches:", len(matches) - len(valid_match_ids))
    print("Clean deliveries:", len(clean_df))

    # Identify innings phase
    clean_df["phase"] = pd.cut(
        clean_df["overid"],
        bins=[0, 6, 15, 20],
        labels=["Powerplay", "Middle", "Death"]
    )

    # Wickets lost before each delivery
    clean_df["wickets_lost_before"] = (
        clean_df.groupby(["match_id", "innings"])["is_wicket"]
        .transform(
            lambda x: (
                x.fillna(False)
                .astype(int)
                .cumsum()
                .shift(fill_value=0)
            )
        )
    )

    # Identify deliveries faced.
    # Wide deliveries are excluded from balls faced.
    clean_df["is_ball_faced"] = (
        clean_df["extras_type"]
        .fillna("")
        .str.lower()
        .ne("wide")
    )

    # Identify boundaries
    clean_df["is_boundary"] = (
        clean_df["is_four"].fillna(False).astype(bool)
        | clean_df["is_six"].fillna(False).astype(bool)
    )

    # Batter runs and balls faced
    batting = (
        clean_df.groupby(
            ["striker_id", "striker_name"]
        )
        .agg(
            runs=("runs", "sum"),
            balls_faced=("is_ball_faced", "sum"),
            fours=("is_four", "sum"),
            sixes=("is_six", "sum"),
            boundary_runs=(
                "runs",
                lambda x: x[
                    clean_df.loc[x.index, "is_boundary"]
                ].sum()
            )
        )
        .reset_index()
        .rename(columns={
            "striker_id": "player_id",
            "striker_name": "player_name"
        })
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
        .rename(columns={
            "dismissed_player_id": "player_id"
        })
    )

    batting = batting.merge(
        dismissals,
        on="player_id",
        how="left"
    )

    batting["dismissals"] = (
        batting["dismissals"].fillna(0)
    )

    # Calculate batting metrics
    batting["batting_average"] = (
        batting["runs"]
        / batting["dismissals"].replace(0, pd.NA)
    )

    batting["strike_rate"] = (
        batting["runs"]
        / batting["balls_faced"].replace(0, pd.NA)
        * 100
    )

    batting["boundary_percentage"] = (
        batting["boundary_runs"]
        / batting["runs"].replace(0, pd.NA)
        * 100
    )

    # Calculate batter performance in a pressure situation
    def pressure_summary(subset, pressure_name):

        pressure_batting = (
            subset.groupby(["striker_id", "striker_name"])
            .agg(
                runs=("runs", "sum"),
                balls_faced=("is_ball_faced", "sum"),
                fours=("is_four", "sum"),
                sixes=("is_six", "sum")
            )
            .reset_index()
            .rename(columns={
                "striker_id": "player_id",
                "striker_name": "player_name"
            })
        )

        # Count distinct batting innings
        innings_count = (
            subset[
                ["striker_id", "striker_name", "match_id", "innings"]
            ]
            .drop_duplicates()
            .groupby(["striker_id", "striker_name"])
            .size()
            .reset_index(name="innings_count")
            .rename(columns={
                "striker_id": "player_id",
                "striker_name": "player_name"
            })
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
            .rename(columns={
                "dismissed_player_id": "player_id"
            })
        )

        pressure_batting = pressure_batting.merge(
            innings_count,
            on=["player_id", "player_name"],
            how="left"
        )

        pressure_batting = pressure_batting.merge(
            pressure_dismissals,
            on="player_id",
            how="left"
        )

        pressure_batting["dismissals"] = (
            pressure_batting["dismissals"].fillna(0)
        )

        pressure_batting["strike_rate"] = (
            pressure_batting["runs"]
            / pressure_batting["balls_faced"].replace(0, pd.NA)
            * 100
        )

        pressure_batting["batting_average"] = (
            pressure_batting["runs"]
            / pressure_batting["dismissals"].replace(0, pd.NA)
        )

        print(f"\n{pressure_name.upper()} - TOP 10 BATTERS BY RUNS")

        print(
            pressure_batting.sort_values(
                "runs", ascending=False
            )[
                [
                    "player_name",
                    "runs",
                    "balls_faced",
                    "innings_count",
                    "dismissals",
                    "batting_average",
                    "strike_rate"
                ]
            ].head(10).round(2).to_string(index=False)
        )

        output_name = (
            file_name.replace(
                "_deliveries.csv",
                f"_{pressure_name}_batters.csv"
            )
        )

        pressure_batting.to_csv(
            DATA_DIR / output_name,
            index=False
        )

        print("Saved:", output_name)

    # Pressure situation 1: Death overs (16-20)
    death_pressure = clean_df[
        clean_df["phase"] == "Death"
    ]

    pressure_summary(
        death_pressure,
        "death_pressure"
    )

    # Pressure situation 2: Five or more wickets lost
    wicket_pressure = clean_df[
        clean_df["wickets_lost_before"] >= 5
    ]

    pressure_summary(
        wicket_pressure,
        "wicket_pressure"
    )

    # Sort by total runs for initial inspection
    batting = batting.sort_values(
        "runs",
        ascending=False
    )

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
                "boundary_percentage"
            ]
        ].head(15).round(2).to_string(index=False)
    )

    # Save clean dataset and batter statistics
    clean_df.to_csv(
        DATA_DIR / file_name.replace(
            "_deliveries.csv",
            "_clean_deliveries.csv"
        ),
        index=False
    )

    batting.to_csv(
        DATA_DIR / file_name.replace(
            "_deliveries.csv",
            "_batter_stats.csv"
        ),
        index=False
    )

    print("\nFILES SAVED SUCCESSFULLY")


if __name__ == "__main__":

    for file_name in [
        "ipl_deliveries.csv",
        "t20i_deliveries.csv"
    ]:
        analyse_batters(file_name)
