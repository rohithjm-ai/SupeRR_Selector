
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")


def inspect_pressure_data(file_name):
    df = pd.read_csv(DATA_DIR / file_name)

    print("\n" + "=" * 65)
    print("PRESSURE DATA INSPECTION:", file_name)
    print("=" * 65)

    print("\nINNINGS DISTRIBUTION")
    print(df["innings"].value_counts().sort_index().to_string())

    print("\nOVER ID SUMMARY")
    print(df["overid"].describe().to_string())

    print("\nOVER ID VALUES")
    print(
        df["overid"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nDELIVERIES PER OVER SUMMARY")
    deliveries_per_over = (
        df.groupby(["match_id", "innings", "overid"])
        .size()
    )
    print(deliveries_per_over.describe().to_string())

    print("\nFIRST 15 DELIVERY RECORDS")
    columns = [
        "match_id",
        "innings",
        "overid",
        "over_ball_num",
        "innings_ball_num",
        "striker_name",
        "runs",
        "extras_type",
        "is_wicket"
    ]

    print(df[columns].head(15).to_string(index=False))


def main():
    inspect_pressure_data("ipl_deliveries.csv")
    inspect_pressure_data("t20i_deliveries.csv")


if __name__ == "__main__":
    main()
