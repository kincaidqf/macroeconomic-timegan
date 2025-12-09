"""
prep_windows.py

Loads data/clean/Country*.csv, creates sliding windows, splits into
train/val/test by country, and scales features to [0,1] using
Min–Max fitted on TRAIN ONLY.

Defaults:
  - L = 24 (window length)
  - stride = 1
  - train: Country1, Country2, Country3, Country4, Country5, Country 6
  - val: Country7
  - test: Country8, Country9

"""

from pathlib import Path
from typing import List, Tuple, Dict
import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

FEATURES = ["Inflation", "Unemployment", "GDP Growth", "Population Growth"]

def load_country_series(folder: Path) -> Dict[str, pd.DataFrame]:
    """
    Return country_name (df) with columns year + FEATURES sorted by year
    """

    data: Dict[str, pd.DataFrame] = {}
    for file in sorted(folder.glob("Country*.csv")):
        df = pd.read_csv(file)
        cols = ["Year"] + FEATURES
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise ValueError(f"File {file} is missing columns: {missing}")
        df = df[cols].sort_values("Year").reset_index(drop=True)
        data[file.stem] = df

    if not data:
        raise FileNotFoundError(f"No Country*.csv files found in {folder}")
    return data


def make_windows(arr_2d: np.ndarray, L: int, stride: int = 1) -> List[np.ndarray]:
    """
    Create sliding windows of length L (L by D) with given stride from arr_2d
    arr_2d: shape (T, D)
    Drop windows with NaN value
    Returns (N x L x D) array
    - N = number of windows
    - L = window length
    - D = number of features (5)
    """
    T, D = arr_2d.shape
    out: List[np.ndarray] = []
    for start in range(0, T - L + 1, stride):
        window = arr_2d[start:start + L, :]
        if np.isnan(window).any():
            continue
        out.append(window.astype(np.float32))
    return out


def df_to_windows(df: pd.DataFrame, L: int, stride: int) -> List[np.ndarray]:
    """
    Convert a country dataframe to sliding windows, takes input dataframe and calls make_windows
    """
    X = df[FEATURES].to_numpy(dtype=float)
    return make_windows(X, L=L, stride=stride)


def split_by_country(countries: List[str],
                     val_countries: List[str],
                     test_countries: List[str]) -> Tuple[List[str], List[str], List[str]]:
    
    train = [c for c in countries if c not in set(val_countries) | set(test_countries)]
    val = [c for c in countries if c in set(val_countries)]
    test = [c for c in countries if c in set(test_countries)] 

    return train, val, test


def fit_minmax(windows: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit Min-Max scaler to list of windows
    Returns (minv, maxv, rangev) arrays of shape (D,)
    """
    if not windows:
        raise ValueError("No windows provided for Min-Max fitting")
    stacked = np.concatenate(windows, axis=0)  # Shape (N*L, D)
    minv = stacked.min(axis=0)
    maxv = stacked.max(axis=0)
    rangev = maxv - minv
    rangev[rangev == 0.0] = 1.0  # Prevent division by zero

    return minv, maxv, rangev


def apply_minmax(windows: List[np.ndarray], minv: np.ndarray, rangev: np.ndarray) -> List[np.ndarray]:
    """
    Apply Min-Max scaling to list of windows using provided minv and rangev
    """
    if not windows:
        return []
    scaled = [(w - minv) / rangev for w in windows]
    # Clip to [0, 1] to avoid numerical issues
    scaled = [np.clip(w, 0.0, 1.0).astype(np.float32) for w in scaled]

    return scaled  


def prepare_windows(data_dir: Path,
                    L: int = 24,
                    stride: int = 1,
                    val_countries: List[str] = None,
                    test_countries: List[str] = None):
    """
    - Load Country csvs from data folder
    - Split by country
    - Generate sliding windows for each country
    - Fit Min-Max scaler on TRAIN only
    returns scaled windows and scalar params
    """

    val_countries = val_countries or ["Country7"]
    test_countries = test_countries or ["Country8", "Country9"]

    all_data = load_country_series(data_dir)
    all_countries = sorted(all_data.keys())

    train_list, val_list, test_list = split_by_country(all_countries, val_countries, test_countries)

    # Build windows
    train_windows: List[np.ndarray] = []
    for country in train_list:
        train_windows += df_to_windows(all_data[country], L=L, stride=stride)
    val_windows: List[np.ndarray] = []
    for country in val_list:
        val_windows += df_to_windows(all_data[country], L=L, stride=stride)
    test_windows: List[np.ndarray] = []
    for country in test_list:
        test_windows += df_to_windows(all_data[country], L=L, stride=stride)

    # Fit scalar on TRAIN only
    minv, maxv, rangev = fit_minmax(train_windows)

    # Scale all sets
    train_scaled = apply_minmax(train_windows, minv, rangev)
    val_scaled = apply_minmax(val_windows, minv, rangev)
    test_scaled = apply_minmax(test_windows, minv, rangev)

    summary = {
        "countries": {
            "train": train_list,
            "val":   val_list,
            "test":  test_list,
        },
        "counts": {
            "train_windows": len(train_scaled),
            "val_windows":   len(val_scaled),
            "test_windows":  len(test_scaled),
        },
        "shapes": {
            "example_window_shape": tuple(train_scaled[0].shape) if train_scaled else None,
            "feature_count": len(FEATURES),
            "window_length": L,
            "stride": stride,
        },
        "scaler": {
            "minv": minv.tolist(),
            "maxv": maxv.tolist(),
        }
    }

    return train_scaled, val_scaled, test_scaled, (minv, rangev), summary


def prepare_windows_global(
    data_dir: Path = Path("data/clean_g2"),
    L: int = 24,
    stride: int = 1,
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Dict,
]:
    """
    Prepare sliding windows from ALL countries in the new 'g2' dataset.

    - Reads per-country CSVs from data_dir (e.g., data/clean_g2/Country1.csv, ...).
    - Each CSV is expected to have columns:
        Year,
        GDP per capita growth,
        Govt consumption,
        Unemployment,
        Inflation,
        Mortality rate
    - Builds length-L sliding windows with given stride.
    - Uses *all* windows for training; val/test are empty.

    Returns:
        train_scaled : list of (L, D) float32 arrays (all windows, scaled to [-1, 1])
        val_scaled   : [] (empty list)
        test_scaled  : [] (empty list)
        (minv, rng)  : tuple of (data_min_, data_range_) from MinMaxScaler in original units
        summary      : dict with counts and shapes
    """
    data_dir = Path(data_dir)

    # Load all Country*.csv files
    country_files = sorted(data_dir.glob("Country*.csv"))
    if not country_files:
        raise FileNotFoundError(f"No Country*.csv files found in {data_dir}")

    feature_cols = [
        "GDP per capita growth",
        "Govt consumption",
        "Unemployment",
        "Inflation",
        "Mortality rate",
    ]

    all_series = []  # list of (T, D) arrays

    for path in country_files:
        df = pd.read_csv(path)

        # Ensure required columns exist
        df = pd.read_csv(path)

        missing = [c for c in feature_cols + ["Year"] if c not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")

        # Drop rows where ANY feature is NaN
        df = df.dropna(how="any", subset=feature_cols)

        # Sort by Year
        df = df.sort_values("Year")

        # Extract features as numeric numpy array
        values = df[feature_cols].to_numpy(dtype="float32")

        # Skip if not enough clean timesteps
        if values.shape[0] < L:
            continue

        all_series.append(values)

    if not all_series:
        raise ValueError(f"No series with length >= {L} found in {data_dir}")

    # Build sliding windows for all countries
    windows = []  # list of (L, D)
    for arr in all_series:
        T, D = arr.shape
        for start in range(0, T - L + 1, stride):
            windows.append(arr[start : start + L, :])

    if not windows:
        raise ValueError("No windows created. Check L and stride.")

    windows = np.stack(windows, axis=0)  # (N, L, D)
    N, L_check, D = windows.shape

    # Fit MinMaxScaler on all windows (flatten across N and L)
    flat = windows.reshape(-1, D)  # (N*L, D)
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    scaler.fit(flat)

    flat_scaled = scaler.transform(flat)
    windows_scaled = flat_scaled.reshape(N, L_check, D).astype("float32")

    # Convert to list-of-arrays to stay consistent with existing code
    train_scaled = [w for w in windows_scaled]
    val_scaled: List[np.ndarray] = []
    test_scaled: List[np.ndarray] = []

    # Original-scale min and range (for inverse scaling)
    minv = scaler.data_min_.astype("float32")
    maxv = scaler.data_max_.astype("float32")
    rng = (maxv - minv).astype("float32")

    summary = {
        "counts": {
            "train_windows": int(N),
            "val_windows": 0,
            "test_windows": 0,
        },
        "shapes": {
            "window_length": int(L_check),
            "feature_count": int(D),
        },
        "features": feature_cols,
    }

    return train_scaled, val_scaled, test_scaled, (minv, rng), summary


def main():
    # Argument parsing set up to run from terminal with different inputs if desired
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/clean",
                    help="Directory with Country*.csv files (default: data/clean)")
    ap.add_argument("--L", type=int, default=24,
                    help="Window length L (default: 24)")
    ap.add_argument("--stride", type=int, default=1,
                    help="Stride for sliding windows (default: 1)")
    ap.add_argument("--val_countries", nargs="+", default=["Country7"],
                    help="List of countries for validation (default: Country7)")
    ap.add_argument("--test_countries", nargs="+", default=["Country8", "Country9"],
                    help="List of countries for testing (default: Country8 Country9)")
    args = ap.parse_args()

    train_scaled, val_scaled, test_scaled, (minv, rangev), summary = prepare_windows(
        data_dir=Path(args.data_dir),
        L=args.L,
        stride=args.stride,
        val_countries=args.val_countries,
        test_countries=args.test_countries
    )

    print("Window Prep Summary")
    print("Countries:")
    for split in ["train", "val", "test"]:
        print(f"  {split:>5}: {summary['countries'][split]}")
    print("Counts:")
    for k, v in summary["counts"].items():
        print(f"  {k:>15}: {v}")
    print("Shapes:")
    for k, v in summary["shapes"].items():
        print(f"  {k:>22}: {v}")
    print("Scaler (minv/maxv) per feature (FEATURES order):")
    print("  FEATURES:", FEATURES)
    print("  minv    :", np.round(minv, 4))
    print("  maxv    :", np.round(summary['scaler']['maxv'], 4))


if __name__ == "__main__":
    main()

