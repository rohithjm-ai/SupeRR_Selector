
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")


def inspect_extra_types(file_name):
    df = pd.read_csv(DATA_DIR / file_name)

    print("\n" + "=" * 65)
    print("LEGAL DELIVERY INVESTIGATION:", file_name)
    print("=" * 65)

    columns = [
        "match_id",
        "innings",
        "overid",
        "over_ball_num",
        "innings_ball_num",
        "striker_name",
        "bowler_name",
        "runs",
        "extras_type",
        "extras1_runs",
        "extras_2_runs",
        "delivery_style",
        "bowl_speed",
        "is_wicket",
        "dismissal_type"
    ]

    for category in ["Overstep", "other", "Penalty"]:
        selected = df[df["extras_type"] == category]

        print(f"\n--- {category}: {len(selected)} records ---")

        if len(selected) > 0:
            print(
                selected[columns]
                .head(15)
                .to_string(index=False)
            )

    print("\n--- ALL EXTRAS CATEGORIES AND OVER BALL NUMBERS ---")

    print(
        df.groupby(
            ["extras_type", "over_ball_num"],
            dropna=False
        ).size().to_string()
    )


def main():
    inspect_extra_types("ipl_deliveries.csv")
    inspect_extra_types("t20i_deliveries.csv")


if __name__ == "__main__":
    main()
