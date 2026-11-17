from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")


def inspect_run_structure(file_name):

    df = pd.read_csv(DATA_DIR / file_name)

    print("\n" + "=" * 75)
    print(f"RUN STRUCTURE: {file_name}")
    print("=" * 75)

    # Examine deliveries where extras and runs both exist
    print("\nDELIVERIES WITH BOTH RUNS AND EXTRAS")

    examples = df[
        ((df["extras1_runs"] > 0) | (df["extras_2_runs"] > 0)) & (df["runs"] > 0)
    ]

    columns = [
        "striker_name",
        "bowler_name",
        "runs",
        "is_four",
        "is_six",
        "extras_type",
        "extras1_runs",
        "extras_2_runs",
    ]

    print(examples[columns].head(15).to_string(index=False))

    # Summarise runs by extras type
    print("\nRUNS BY EXTRAS TYPE")

    summary = (
        df.groupby("extras_type", dropna=False)
        .agg(
            deliveries=("match_id", "size"),
            runs_sum=("runs", "sum"),
            extras1_sum=("extras1_runs", "sum"),
            extras2_sum=("extras_2_runs", "sum"),
        )
        .reset_index()
    )

    print(summary.to_string(index=False))

    # Check four and six indicators
    print("\nFOUR AND SIX VALIDATION")

    print("Four indicators:", df["is_four"].sum())
    print("Six indicators:", df["is_six"].sum())

    print(
        "Rows with four indicator but runs != 4:",
        len(df[df["is_four"] & (df["runs"] != 4)]),
    )

    print(
        "Rows with six indicator but runs != 6:",
        len(df[df["is_six"] & (df["runs"] != 6)]),
    )


def main():

    inspect_run_structure("ipl_deliveries.csv")
    inspect_run_structure("t20i_deliveries.csv")


if __name__ == "__main__":
    main()
