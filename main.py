from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")


def inspect_scoring_data(file_name):

    df = pd.read_csv(DATA_DIR / file_name)

    print("\n" + "=" * 70)
    print(f"SCORING ANALYSIS: {file_name}")
    print("=" * 70)

    # Basic run totals
    print("\nRUNS AND EXTRAS")
    print("Sum of runs:", df["runs"].sum())
    print("Sum of extras1_runs:", df["extras1_runs"].sum())
    print("Sum of extras_2_runs:", df["extras_2_runs"].sum())

    # Extras analysis
    print("\nEXTRAS TYPES")
    print(df["extras_type"].value_counts(dropna=False).to_string())

    # Deliveries containing extras
    extras = df[(df["extras1_runs"] > 0) | (df["extras_2_runs"] > 0)]

    print("\nDELIVERIES WITH EXTRAS:", len(extras))

    print("\nEXAMPLE EXTRAS DELIVERIES")
    print(
        extras[
            [
                "striker_name",
                "bowler_name",
                "runs",
                "extras_type",
                "extras1_runs",
                "extras_2_runs",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    # Wicket analysis
    wickets = df[df["is_wicket"] == True]

    print("\nWICKET ANALYSIS")
    print("Total wicket deliveries:", len(wickets))

    print("\nDISMISSAL TYPES")
    print(wickets["dismissal_type"].value_counts(dropna=False).to_string())


def main():

    inspect_scoring_data("ipl_deliveries.csv")
    inspect_scoring_data("t20i_deliveries.csv")


if __name__ == "__main__":
    main()
