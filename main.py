from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def create_competition_datasets():

    # Load datasets
    matches = pd.read_csv(DATA_DIR / "matches.csv")
    competitions = pd.read_csv(DATA_DIR / "competitions.csv")
    deliveries = pd.read_csv(DATA_DIR / "ball_by_ball.csv")

    # Attach competition details to matches
    match_data = matches.merge(competitions, on="comp_id", how="left")

    # Separate IPL and T20I matches
    ipl_matches = match_data[
        match_data["comp_name"].str.contains(r"\bIPL\b", case=False, na=False)
    ].copy()

    t20i_matches = match_data[match_data["match_format"] == "T20I"].copy()

    # Get match IDs
    ipl_match_ids = ipl_matches["match_id"]
    t20i_match_ids = t20i_matches["match_id"]

    # Separate ball-by-ball deliveries
    ipl_deliveries = deliveries[deliveries["match_id"].isin(ipl_match_ids)].copy()

    t20i_deliveries = deliveries[deliveries["match_id"].isin(t20i_match_ids)].copy()

    # Create processed directory
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Save datasets
    ipl_matches.to_csv(PROCESSED_DIR / "ipl_matches.csv", index=False)

    t20i_matches.to_csv(PROCESSED_DIR / "t20i_matches.csv", index=False)

    ipl_deliveries.to_csv(PROCESSED_DIR / "ipl_deliveries.csv", index=False)

    t20i_deliveries.to_csv(PROCESSED_DIR / "t20i_deliveries.csv", index=False)

    # Display results
    print("\nCOMPETITION DATASET SUMMARY")
    print("=" * 60)

    print("\nIPL")
    print("Matches:", len(ipl_matches))
    print("Deliveries:", len(ipl_deliveries))

    print("\nT20I")
    print("Matches:", len(t20i_matches))
    print("Deliveries:", len(t20i_deliveries))

    print("\nTOTAL")
    print("Matches:", len(ipl_matches) + len(t20i_matches))
    print("Deliveries:", len(ipl_deliveries) + len(t20i_deliveries))

    print("\nFiles saved in:", PROCESSED_DIR)


if __name__ == "__main__":
    create_competition_datasets()
