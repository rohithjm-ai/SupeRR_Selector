from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")


def inspect_extras(file_name):
    df = pd.read_csv(DATA_DIR / file_name)

    print("\n" + "=" * 65)
    print(file_name)
    print("=" * 65)

    print("\nEXTRAS TYPE AND EXTRAS COLUMN VALUES")

    result = (
        df.groupby("extras_type", dropna=False)
        .agg(
            deliveries=("match_id", "size"),
            extras1_values=("extras1_runs", lambda x: sorted(x.unique())),
            extras2_values=("extras_2_runs", lambda x: sorted(x.unique())),
        )
        .reset_index()
    )

    print(result.to_string(index=False))

    print("\nROWS WHERE BOTH EXTRAS COLUMNS ARE NON-ZERO")

    both = df[(df["extras1_runs"] > 0) & (df["extras_2_runs"] > 0)]

    columns = [
        "striker_name",
        "bowler_name",
        "runs",
        "extras_type",
        "extras1_runs",
        "extras_2_runs",
    ]

    print(both[columns].head(15).to_string(index=False))


def main():
    inspect_extras("ipl_deliveries.csv")
    inspect_extras("t20i_deliveries.csv")


if __name__ == "__main__":
    main()
