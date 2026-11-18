
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def verify_match_scores(file_name):
    deliveries = pd.read_csv(PROCESSED_DIR / file_name)
    matches = pd.read_csv(RAW_DIR / "matches.csv")

    # Calculate total runs for every delivery.
    deliveries["total_runs"] = (
        deliveries["runs"]
        + deliveries["extras1_runs"]
        + deliveries["extras_2_runs"]
    )

    # Calculate each team's innings score.
    innings_scores = (
        deliveries.groupby(
            ["match_id", "innings", "batting_team_id"]
        )
        .agg(team_score=("total_runs", "sum"))
        .reset_index()
    )

    # Consider matches decided by runs.
    matches["run_margin"] = pd.to_numeric(
        matches["result"].str.extract(
            r"Won by (\d+)\s+runs?",
            flags=0
        )[0],
        errors="coerce"
    )

    run_matches = matches[
        matches["run_margin"].notna()
        & ~matches["result"].str.contains(
            r"D/?L|DLS",
            case=False,
            na=False
        )
    ]

    results = []

    for _, match in run_matches.iterrows():

        match_innings = innings_scores[
            innings_scores["match_id"] == match["match_id"]
        ]

        winner = match_innings[
            match_innings["batting_team_id"] == match["win_team_id"]
        ]

        loser = match_innings[
            match_innings["batting_team_id"] != match["win_team_id"]
        ]

        if winner.empty or loser.empty:
            continue

        winner_score = winner["team_score"].iloc[0]
        loser_score = loser["team_score"].iloc[0]

        calculated_margin = winner_score - loser_score

        results.append({
            "match_id": match["match_id"],
            "result": match["result"],
            "calculated_margin": calculated_margin,
            "actual_margin": match["run_margin"],
            "match_status": (
                "PASS"
                if calculated_margin == match["run_margin"]
                else "CHECK"
            )
        })

    result_df = pd.DataFrame(results)

    print("\n" + "=" * 65)
    print(f"SCORE VALIDATION: {file_name}")
    print("=" * 65)

    print("\nMatches checked:", len(result_df))
    print("Exact matches:", (result_df["match_status"] == "PASS").sum())
    print("Mismatches:", (result_df["match_status"] == "CHECK").sum())

    print("\nMISMATCH DETAILS")

    mismatches = result_df[
        result_df["match_status"] == "CHECK"
    ]

    print(mismatches.head(10).to_string(index=False))


def main():
    verify_match_scores("ipl_deliveries.csv")
    verify_match_scores("t20i_deliveries.csv")


if __name__ == "__main__":
    main()
