from pathlib import Path
from pandas as pd

DATA_DIR = Path(r"C:\Users\jmroh\OneDrive\Documents\Desktop\SupeRR_Selector\data\raw")
def inspect_dataset(file_path):
    
def main():
    csv_files=sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV file found in {DATA_DIR}")
        return
    print(f"CSV file found in the location {DATA_DIR} and its len{len(csv_files)}")
    for file_path in csv_files:
        inspect_dataset(file_path)