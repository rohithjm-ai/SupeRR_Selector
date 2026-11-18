
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")


def inspect_targets(file_name):
    df = pd.read_csv(DATA_DIR / file_name)
    matches = pd.read_csv(RAW_DIR / "matches.csv")

    # Keep only normal innings
    df = df[df["innings"].isin([1, 2])].copy()

    # Calculate team totals
    df["total_runs"] = (
        df["runs"].fillna(0)
        + df["extras1_runs"].fillna(0)
        + df["extras_2_runs"].fillna(0)
    )

    scores = (
        df.groupby(["match_id", "innings"])
        ["total_runs"]
        .sum()
        .unstack(fill_value=0)
    )

    scores = scores.rename(columns={
        1: "first_innings_score",
        2: "second_innings_score"
    })

    result = matches.merge(
        scores,
        on="match_id",
        how="inner"
    )

    print("\n" + "=" * 65)
    print("TARGET INVESTIGATION:", file_name)
    print("=" * 65)

    print("\nMATCHES WITH SCORE DATA:", len(result))

    print("\nREVISED TARGET SUMMARY")
    print("Matches with revised target:",
          result["revised_target"].notna().sum())

    print("\nREVISED OVERS SUMMARY")
    print("First innings revised overs:",
          result["first_inn_revised_overs"].notna().sum())
    print("Second innings revised overs:",
          result["second_inn_revised_overs"].notna().sum())

    print("\nRESULT TYPE COUNTS")
    print(result["result"].value_counts().head(12).to_string())

    print("\nSAMPLE MATCH SCORE AND TARGET DATA")

    columns = [
        "match_id",
        "first_innings_score",
        "second_innings_score",
        "revised_target",
        "first_inn_revised_overs",
        "second_inn_revised_overs",
        "result"
    ]

    print(result[columns].head(15).to_string(index=False))


def main():
    inspect_targets("ipl_deliveries.csv")
    inspect_targets("t20i_deliveries.csv")


if __name__ == "__main__":
    main()
