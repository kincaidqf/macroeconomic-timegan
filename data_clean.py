import argparse
from pathlib import Path
import pandas as pd

def main():
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
    print(f"[LOADED] {in_path} -> {df.shape[0]} rows, {df.shape[1]} cols")
    print(df.head())

if __name__ == "__main__":
    main()