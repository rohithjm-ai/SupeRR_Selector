
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")

BOWLER_CREDITED = [
    "Caught",
    "Bowled",
    "LBW",
    "Stump",
    "Hit Wicket"
]


def inspect_wickets(file_name):

    df = pd.read_csv(DATA_DIR / file_name)

    wickets = df[df["is_wicket"] == True].copy()

    wickets["dismissal_type"] = (
        wickets["dismissal_type"]
        .fillna("Missing")
        .str.strip()
    )

    wickets["bowler_credited"] = (
        wickets["dismissal_type"].isin(BOWLER_CREDITED)
    )

    print("\n" + "=" * 65)
    print("WICKET VALIDATION:", file_name)
    print("=" * 65)

    print("\nTOTAL WICKETS:", len(wickets))

    print("\nDISMISSAL TYPE COUNTS")

    print(
        wickets["dismissal_type"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nBOWLER-CREDITED WICKETS:",
          wickets["bowler_credited"].sum())

    print("OTHER WICKETS:",
          (~wickets["bowler_credited"]).sum())

    print("\nUNCLASSIFIED DISMISSALS")

    print(
        wickets.loc[
            ~wickets["bowler_credited"],
            "dismissal_type"
        ].value_counts().to_string()
    )


def main():

    inspect_wickets("ipl_deliveries.csv")
    inspect_wickets("t20i_deliveries.csv")


if __name__ == "__main__":
    main()
