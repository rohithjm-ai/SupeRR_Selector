
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")


def inspect_other(file_name):
    df = pd.read_csv(DATA_DIR / file_name)

    print("\n" + "=" * 65)
    print("OTHER CATEGORY INVESTIGATION:", file_name)
    print("=" * 65)

    other = df[df["extras_type"] == "other"].copy()

    print("\nTOTAL OTHER RECORDS:", len(other))

    print("\nDELIVERY STYLE COUNTS")
    print(
        other["delivery_style"]
        .fillna("Missing")
        .value_counts()
        .to_string()
    )

    print("\nDELIVERY STYLE AND RUNS")
    print(
        other.groupby(
            ["delivery_style", "runs", "extras1_runs", "extras_2_runs"],
            dropna=False
        ).size().to_string()
    )

    print("\nWICKET DETAILS")
    print(
        other[other["is_wicket"] == True][
            ["match_id", "striker_name", "bowler_name",
             "delivery_style", "dismissal_type"]
        ].to_string(index=False)
    )


def main():
    inspect_other("ipl_deliveries.csv")
    inspect_other("t20i_deliveries.csv")


if __name__ == "__main__":
    main()
