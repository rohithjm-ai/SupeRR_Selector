
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")


def inspect_extra_details(file_name):
    df = pd.read_csv(DATA_DIR / file_name)

    print("\n" + "=" * 70)
    print(file_name)
    print("=" * 70)

    print("\nROWS WHERE extras_2_runs > 0")

    selected = df[df["extras_2_runs"] > 0]

    columns = [
        "match_id",
        "innings",
        "overid",
        "striker_name",
        "bowler_name",
        "runs",
        "is_four",
        "is_six",
        "extras_type",
        "extras1_runs",
        "extras_2_runs",
        "stroke",
        "delivery_style"
    ]

    print(selected[columns].head(20).to_string(index=False))

    print("\nROWS WHERE EXTRAS TYPE IS MISSING BUT EXTRAS EXIST")

    unusual = df[
        df["extras_type"].isna() &
        (
            (df["extras1_runs"] > 0) |
            (df["extras_2_runs"] > 0)
        )
    ]

    print(unusual[columns].to_string(index=False))


def main():
    inspect_extra_details("ipl_deliveries.csv")
    inspect_extra_details("t20i_deliveries.csv")


if __name__ == "__main__":
    main()
