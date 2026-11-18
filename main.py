import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")

for file_name in ["ipl_deliveries.csv", "t20i_deliveries.csv"]:

    df = pd.read_csv(DATA_DIR / file_name)

    # Keep only normal innings
    df = df[df["innings"].isin([1, 2])].copy()

    # Identify innings phase
    df["phase"] = pd.cut(
        df["overid"],
        bins=[0, 6, 15, 20],
        labels=["Powerplay", "Middle", "Death"]
    )

    # Wickets lost before each delivery
    df["wickets_lost_before"] = (
        df.groupby(["match_id", "innings"])["is_wicket"]
        .transform(lambda x: x.fillna(False).astype(int).cumsum().shift(fill_value=0))
    )

    print("\n" + "=" * 60)
    print(f"PRESSURE INVESTIGATION: {file_name}")
    print("=" * 60)

    print("\nDELIVERIES BY INNINGS PHASE")
    print(df["phase"].value_counts().sort_index())

    print("\nDELIVERIES BY WICKETS LOST")
    print(df["wickets_lost_before"].value_counts().sort_index())

    print("\nDEATH OVERS BY WICKETS LOST")
    death = df[df["phase"] == "Death"]
    print(death["wickets_lost_before"].value_counts().sort_index())

    print("\nFIRST INNINGS VS SECOND INNINGS")
    print(df.groupby("innings", observed=False).size())