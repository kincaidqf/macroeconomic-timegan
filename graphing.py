import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize real vs synthetic data for a given version."
    )
    parser.add_argument(
        "version",
        type=int,
        help="Version number (e.g. 0, 1, 8, 9) corresponding to artifacts/baseline_vX",
    )
    return parser.parse_args()


def plot_scatter_cloud(
    train: np.ndarray,
    synth: np.ndarray,
    version: int,
    base_dir: Path,
    feature_x: int = 1,
    feature_y: int = 2,
):
    """
    Create a 2D scatter 'cloud' plot:

    - Training data: very light, dense cloud of points
    - Synthetic data: darker overlay points

    Inputs:
        train: (N, L, D) real windows (scaled)
        synth: (N, L, D) synthetic windows (scaled)
        version: version number (for title / file name)
        base_dir: Path to artifacts/baseline_vX
        feature_x: index of feature for x-axis
        feature_y: index of feature for y-axis
    """

    N_t, L_t, D_t = train.shape
    N_s, L_s, D_s = synth.shape

    if D_t <= max(feature_x, feature_y) or D_s <= max(feature_x, feature_y):
        raise ValueError(
            f"Feature indices ({feature_x}, {feature_y}) out of range for D={D_t}"
        )

    # Flatten windows: (N * L, D)
    train_flat = train.reshape(-1, D_t)
    synth_flat = synth.reshape(-1, D_s)

    x_train = train_flat[:, feature_x]
    y_train = train_flat[:, feature_y]

    x_synth = synth_flat[:, feature_x]
    y_synth = synth_flat[:, feature_y]

    # Start a new figure
    plt.figure(figsize=(6, 6))

    # Plot training cloud: very light, many small points
    plt.scatter(
        x_train,
        y_train,
        s=5,
        alpha=0.05,          # very transparent
        color="tab:blue",
        edgecolors="none",
        label="Real (train)",
    )

    # Overlay synthetic points: darker, slightly larger
    plt.scatter(
        x_synth,
        y_synth,
        s=10,
        alpha=0.8,
        color="black",
        edgecolors="none",
        label="Synthetic",
    )

    # Labels – you can adjust to match your feature ordering
    feature_names = ["Inflation", "Unemployment", "GDP Growth", "Population Growth"]
    def fname(idx):
        if 0 <= idx < len(feature_names):
            return feature_names[idx]
        return f"Feature {idx}"

    plt.xlabel(fname(feature_x))
    plt.ylabel(fname(feature_y))
    plt.title(f"Real vs Synthetic (v{version})")

    plt.legend(loc="best")
    plt.tight_layout()

    out_path = base_dir / f"scatter_cloud_v{version}_f{feature_x}_f{feature_y}.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved figure to {out_path.resolve()}")

    # Optionally show the figure when running interactively
    # plt.show()
    plt.close()


def load_data(base_dir: Path):
    """
    Load training and synthetic data for a given version.

    Expects files:
        - train_scaled.npy
        - synthetic_scaled.npy

    Each should be shape (N, L, D).
    """
    train_path = base_dir / "train_orig.npy"
    synth_path = base_dir / "synthetic_orig.npy"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing training data: {train_path}")
    if not synth_path.exists():
        raise FileNotFoundError(f"Missing synthetic data: {synth_path}")

    train = np.load(train_path)  # (N, L, D)
    synth = np.load(synth_path)  # (N, L, D)

    if train.ndim != 3 or synth.ndim != 3:
        raise ValueError(
            f"Expected 3D arrays (N, L, D). Got train.shape={train.shape}, "
            f"synth.shape={synth.shape}"
        )

    return train, synth



def main():
    args = parse_args()
    version = args.version

    base_dir = Path(f"artifacts/baseline_v{version}")
    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist.")
        sys.exit(1)

    # We'll fill these in next:
    train, synth = load_data(base_dir)
    plot_scatter_cloud(train, synth, version, base_dir)


if __name__ == "__main__":
    main()
