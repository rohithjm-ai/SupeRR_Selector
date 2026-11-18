
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")


def investigate(file_name):
    df = pd.read_csv(DATA_DIR / file_name)
    matches = pd.read_csv(RAW_DIR / "matches.csv")

    df = df[df["innings"].isin([1, 2])].copy()

    scores = (
        df.groupby(["match_id", "innings"])["runs"]
        .sum()
        .unstack()
        .rename(columns={
            1: "first_score",
            2: "second_score"
        })
    )

    result = matches.merge(
        scores,
        on="match_id",
        how="left"
    )

    print("\n" + "=" * 65)
    print("REVISED TARGET AND MISSING SCORE INVESTIGATION:", file_name)
    print("=" * 65)

    print("\nMATCHES WITHOUT FIRST OR SECOND INNINGS SCORE")
    missing = result[
        result["first_score"].isna()
        | result["second_score"].isna()
    ]

    print("Count:", len(missing))

    print(
        missing[
            ["match_id", "result", "first_score", "second_score"]
        ].to_string(index=False)
    )

    print("\nALL MATCHES WITH REVISED TARGET")

    revised = result[result["revised_target"].notna()]

    columns = [
        "match_id",
        "first_score",
        "second_score",
        "revised_target",
        "first_inn_revised_overs",
        "second_inn_revised_overs",
        "result"
    ]

    print(revised[columns].to_string(index=False))

    print("\nRESULTS MENTIONING D/L OR DLS")

    dls = result[
        result["result"].str.contains(
            r"D/L|DLS|Duckworth",
            case=False,
            na=False
        )
    ]

    print(dls[columns].to_string(index=False))


def main():
    investigate("ipl_deliveries.csv")
    investigate("t20i_deliveries.csv")


if __name__ == "__main__":
    main()
