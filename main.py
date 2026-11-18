
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")


def inspect_extra_deliveries(file_name):
    df = pd.read_csv(DATA_DIR / file_name)

    print("\n" + "=" * 65)
    print("EXTRA DELIVERY INSPECTION:", file_name)
    print("=" * 65)

    print("\n1. DELIVERIES WHERE OVER_BALL_NUM > 6")

    extra_balls = df[df["over_ball_num"] > 6]

    print("Total records:", len(extra_balls))

    print("\nEXTRAS TYPE DISTRIBUTION")
    print(
        extra_balls["extras_type"]
        .fillna("Missing")
        .value_counts()
        .to_string()
    )

    print("\nSAMPLE RECORDS")
    columns = [
        "match_id",
        "innings",
        "overid",
        "over_ball_num",
        "innings_ball_num",
        "striker_name",
        "bowler_name",
        "runs",
        "is_wicket",
        "dismissal_type",
        "extras_type",
        "extras1_runs",
        "extras_2_runs"
    ]

    print(extra_balls[columns].head(20).to_string(index=False))

    print("\n2. INSPECT OTHER AND OVERSTEP")

    unusual = df[df["extras_type"].isin(["other", "Overstep"])]

    print(
        unusual.groupby(
            ["extras_type", "runs", "extras1_runs", "extras_2_runs"],
            dropna=False
        ).size().to_string()
    )


def main():
    inspect_extra_deliveries("ipl_deliveries.csv")
    inspect_extra_deliveries("t20i_deliveries.csv")


if __name__ == "__main__":
    main()
