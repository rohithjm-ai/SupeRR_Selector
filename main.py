from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")


def inspect_competitions():
    matches = pd.read_csv(DATA_DIR / "matches.csv")
    competitions = pd.read_csv(DATA_DIR / "competitions.csv")
    deliveries = pd.read_csv(DATA_DIR / "ball_by_ball.csv")

    # Connect each match to its competition
    match_data = matches.merge(competitions, on="comp_id", how="left")

    # Connect each delivery to its competition
    delivery_data = deliveries.merge(
        matches[["match_id", "comp_id"]], on="match_id", how="left"
    )

    delivery_data = delivery_data.merge(competitions, on="comp_id", how="left")

    # Competition-level summary
    summary = (
        match_data.groupby(["comp_id", "comp_name", "match_format", "season"])
        .agg(matches=("match_id", "nunique"))
        .reset_index()
    )

    delivery_summary = (
        delivery_data.groupby("comp_id")
        .agg(deliveries=("match_id", "size"))
        .reset_index()
    )

    summary = summary.merge(delivery_summary, on="comp_id", how="left")

    print("\nCOMPETITION-WISE DATA SUMMARY")
    print("=" * 90)
    print(summary.to_string(index=False))

    print("\nTOTAL COMPETITIONS:", summary["comp_id"].nunique())
    print("TOTAL MATCHES:", matches["match_id"].nunique())
    print("TOTAL DELIVERIES:", len(deliveries))


if __name__ == "__main__":
    inspect_competitions()
