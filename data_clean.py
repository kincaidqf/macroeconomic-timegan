import argparse
from pathlib import Path
import pandas as pd
import re

ROW_TO_HEADER = {
    r"inflation|consumer prices|cpi": "Inflation",
    r"\bunemploy": "Unemployment",
    r"gdp.*per.*capita|per.*capita.*gdp": "GDP per capita",
    r"gdp.*growth|growth.*gdp": "GDP Growth",
    r"population.*growth": "Population Growth",
}

def map_series_to_header(series_name: str) -> str | None:
    if not isinstance(series_name, str):
        return None
    series = series_name.lower()
    for row, header in ROW_TO_HEADER.items():
        if re.search(row, series):
            return header
    return None

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

    # Reshape from wide to long format
    id_vars = ["Country Name", "Country Code", "Series Name"]
    year_cols = [col for col in df.columns if col.isdigit()]
    df_long = df.melt(id_vars=id_vars, 
                      value_vars=year_cols, 
                      var_name="year",
                      value_name="value")
    
    # Map indicator names to standardized headers and filter
    df_long["Indicator"] = df_long["Series Name"].apply(map_series_to_header)
    keep_headers = ["Inflation", "Unemployment", "GDP Growth", "GDP per capita", "Population Growth"]
    df_long = df_long[df_long["Indicator"].isin(keep_headers)]

    # Set types on values
    df_long["year"] = pd.to_numeric(df_long["year"], errors="coerce").astype("Int64")
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")

    # Pivot to a country-year panel with the 5 columns
    # aggfunc means take first value if there are duplicates
    panel = df_long.pivot_table(
        index=["Country Name", "Country Code", "year"],
        columns="Indicator",
        values="value",
        aggfunc="first"
    ).reset_index()

    # Ensture missing indicator columns still appear as NaN and order them
    order = ["Inflation", "Unemployment", "GDP Growth", "GDP per capita", "Population Growth"]
    for col in order:
        if col not in panel.columns:
            panel[col] = pd.NA
    panel = panel[["Country Name", "Country Code", "year"] + order]

    panel = panel.rename(columns={"year": "Year"})

    print(f"[data_clean] Panel shape: {panel.shape}")
    print(panel.head(15))

    return panel


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

    df_long = restructure_file(df)

    



if __name__ == "__main__":
    main()