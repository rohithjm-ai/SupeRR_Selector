from pathlib import Path
import pandas as pd

# Dataset directory
DATA_DIR = Path("data/raw")


def inspect_dataset(file_path):
    """Inspect a CSV dataset and display its basic information."""

    print("\n" + "=" * 70)
    print(f"FILE: {file_path.name}")
    print("=" * 70)

    # Load CSV
    df = pd.read_csv(file_path)

    # Dataset dimensions
    print(f"\nRows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    # Column names
    print("\nCOLUMN NAMES:")
    print(df.columns.tolist())

    # Data types
    print("\nDATA TYPES:")
    print(df.dtypes)

    # Missing values
    print("\nMISSING VALUES:")
    print(df.isnull().sum())

    # Duplicate records
    print(f"\nDuplicate rows: {df.duplicated().sum():,}")

    # Sample records
    print("\nFIRST 5 RECORDS:")
    print(df.head())


def main():
    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}")
        return

    print(f"Found {len(csv_files)} CSV files.")

    for file_path in csv_files:
        inspect_dataset(file_path)


if __name__ == "__main__":
    main()
