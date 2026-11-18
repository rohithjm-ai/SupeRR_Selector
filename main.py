
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")


def inspect_extra_innings(file_name):
    df = pd.read_csv(DATA_DIR / file_name)
    matches = pd.read_csv(RAW_DIR / "matches.csv")

    print("\n" + "=" * 65)
    print("EXTRA INNINGS INVESTIGATION:", file_name)
    print("=" * 65)

    extra = df[df["innings"] > 2].copy()

    print("\nTOTAL RECORDS IN INNINGS 3 AND 4:", len(extra))

    if extra.empty:
        print("No extra innings found.")
        return

    print("\nDELIVERY COUNTS BY INNINGS")
    print(extra.groupby("innings").size().to_string())

    print("\nMATCHES WITH EXTRA INNINGS")

    match_ids = extra["match_id"].unique()

    match_details = matches[
        matches["match_id"].isin(match_ids)
    ]

    columns = [
        "match_id",
        "match_date",
        "stage",
        "first_inn_revised_overs",
        "second_inn_revised_overs",
        "revised_target",
        "result"
    ]

    print(match_details[columns].to_string(index=False))

    print("\nOVER NUMBERS IN EXTRA INNINGS")
    print(
        extra.groupby(["match_id", "innings"])["overid"]
        .agg(["min", "max", "count"])
        .to_string()
    )


def main():
    inspect_extra_innings("ipl_deliveries.csv")
    inspect_extra_innings("t20i_deliveries.csv")


if __name__ == "__main__":
    main()
