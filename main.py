
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")


def inspect_deliveries(file_name):
    df = pd.read_csv(DATA_DIR / file_name)

    print("\n" + "=" * 65)
    print("DELIVERY COUNT VALIDATION:", file_name)
    print("=" * 65)

    print("\nTOTAL DELIVERIES:", len(df))

    print("\nOVER BALL NUMBER DISTRIBUTION")
    print(df["over_ball_num"].value_counts().sort_index().to_string())

    print("\nINNINGS BALL NUMBER SUMMARY")
    print(df["innings_ball_num"].describe().to_string())

    print("\nEXTRAS TYPE COUNTS")
    print(df["extras_type"].fillna("Missing").value_counts().to_string())

    print("\nDELIVERIES BY EXTRAS TYPE")
    print(
        df.groupby("extras_type", dropna=False)
        .agg(
            deliveries=("match_id", "size"),
            extras1_runs=("extras1_runs", "sum"),
            extras2_runs=("extras_2_runs", "sum")
        )
        .to_string()
    )

    print("\nDUPLICATE INNINGS BALL NUMBERS")
    duplicate_count = df.duplicated(
        subset=["match_id", "innings", "innings_ball_num"],
        keep=False
    ).sum()
    print(duplicate_count)


def main():
    inspect_deliveries("ipl_deliveries.csv")
    inspect_deliveries("t20i_deliveries.csv")


if __name__ == "__main__":
    main()
