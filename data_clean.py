import argparse
from pathlib import Path
import pandas as pd

def restructure_file(df):
    # Drop unnecessary columns
    df = df.drop(columns=["Series Code"])
    print("[data_clean] Dropped 'Series Code' column")

    # Reformat year headers "1970 [YR1970]" to just "1970"
    rename_map = {}
    for col in df.columns:
        if col[:4].isdigit():
            rename_map[col] = col[:4]
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"[data_clean] Renamed year columns")

    print(df.head())
    return


def main():
    # Argument parsing set up to run from terminal with different inputs if desired
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/MacroData.csv", 
                    help="Path to input CSV (default: data/MacroData.csv)")
    ap.add_argument("--outdir", default="data/clean",
                    help="Directory for outputs (default: data/clean)")
    args = ap.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    df.columns = [c.strip() for c in df.columns]

    # Print summary printouts to make sure file is being read
    print(f"[data_clean] {in_path} -> {df.shape[0]} rows, {df.shape[1]} cols")
    print(df.head())

    restructure_file(df)

if __name__ == "__main__":
    main()