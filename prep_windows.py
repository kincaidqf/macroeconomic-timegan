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

FEATURES = ["Inflation", "Unemployment", "GDP Growth", "GDP per capita", "Population Growth"]

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

