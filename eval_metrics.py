import numpy as np
from pathlib import Path
from typing import Tuple, Dict
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def load_data(real_path: str, synth_path: str, match_shapes: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load saved .npy arrays 

    Inputs: 
        real_path - path to real data .npy file
        synth_path - path to synthetic data .npy file

    Expected array shape: (N, L, D)
        N: number of windows
        L: window length
        D: feature dimension

    Returns:
        real - numpy array of real data
        synth - numpy array of synthetic data  
    """

    real = np.load(real_path, allow_pickle=True)
    synth = np.load(synth_path, allow_pickle=True)

    # Type checks
    if isinstance(real, list): real = np.array(real, dtype=np.float32)
    if isinstance(synth, list): synth = np.array(synth, dtype=np.float32)

    # Shape check
    if real.ndim != 3 or synth.ndim != 3:
        raise ValueError(f"Expected 3D arrays (N, L, D), got real.ndim={real.ndim}, synth.ndim={synth.ndim}")
    
    N_r, L_r, D_r = real.shape
    N_s, L_s, D_s = synth.shape
    if L_r != L_s or D_r != D_s:
        raise ValueError(f"Shape mismatch between real and synth: real(L={L_r},D={D_r}), synth(L={L_s},D={D_s})")  
    
    if match_shapes and N_r != N_s:
        N = min(N_r, N_s)
        real = real[:N]
        synth = synth[:N]
    else:
        N = min(N_r, N_s)

    # Ensure float32 dtype
    real = real.astype(np.float32, copy=False)
    synth = synth.astype(np.float32, copy=False)

    # Final check
    assert real.shape[1:] == synth.shape[1:], "Post-processing shapes must match"
    return real, synth


def test_marginals(real: np.ndarray, synth: np.ndarray, return_per_feature: bool = True):
    """
    Per-feature comparison of marginal distributions

    Purpose:
        - Do real and synthetic marginal distributions match?
        - Use 1D tests feature-by-feature (ignoring time ordering)

    Tests:
        - Kolmogorov-Smirnov (KS) test
        - Wasserstein distance

    Inputs:
        real - numpy array of real data (N, L, D)

    Outputs:
        ks_mean: float mean KS statistic aggregate 
        ws_mean: float Wasserstein distance aggregate 
        ks_per_feature - list[float length D]
        ws_per_feature - list[float length D]
    """

    _, _, D = real.shape
    ks_stats, ks_pvals, wdists = [], [], []
    mean_diff, std_diff = [], []

    for d in range (D):
        r = real[:, :, d].ravel()
        s = synth[:, :, d].ravel()

        # Compute KS statistic and p-value
        ks_stat, ks_p = ks_2samp(r, s, alternative='two-sided', method="auto")
        ks_stats.append(float(ks_stat))
        ks_pvals.append(float(ks_p))

        # Compute Wasserstein distance
        wd = wasserstein_distance(r, s)
        wdists.append(float(wd))

        # Moment diffs (synth - real)
        mean_diff.append(float(np.mean(s) - np.mean(r)))
        std_diff.append(float(np.std(s, ddof=1) - np.std(r, ddof=1)))

    metrics = {
        # Aggregates
        "ks_mean": float(np.mean(ks_stats)),
        "ks_max": float(np.max(ks_stats)),
        "ks_p_mean": float(np.mean(ks_pvals)),
        "wd_mean": float(np.mean(wdists)),
        "wd_max": float(np.max(wdists)),
        "mean_abs_diff_mean": float(np.mean(np.abs(mean_diff))),
        "std_abs_diff_mean": float(np.mean(np.abs(std_diff))),
    }

    if return_per_feature:
        metrics.update({
            "ks_per_feature": ks_stats,
            "ks_p_per_feature": ks_pvals,
            "wd_per_feature": wdists,
            "mean_diff_per_feature": mean_diff,
            "std_diff_per_feature": std_diff,
        })

    return metrics


def test_correlation(real, synth):
    """
    Cross-feature correlation comparison

    Purpose:
        - Do the cross-sectional relationships between features match
        - Compare D by D correlation matricies from real vs synthetic
        - Pooled accross all time steps and windows
        - Not comparing time steps/realisticness, just similarities of correlations

    Tests:
        - Frobenius norm between correlation matrices
        - Maximum absolute difference between correlation matrices

    Inputs:
        real - numpy array of real data (N, L, D)
        synth - numpy array of synthetic data (N, L, D)

    Outputs:
        - corr_frobenius: float     ||Corr_real - Corr_synth||_F
        - corr_max_abs: float       max dif = |Corr_real - Corr_synth|
    """

    N_r, L, D = real.shape
    N_s, _, _ = synth.shape

    # Pool across windows/time - reshape to (N*L, D)
    Xr = real.reshape((N_r * L, D))
    Xs = synth.reshape((N_s * L, D))

    # To be safe drop rows with non-finite values (NaN, Inf)
    mask_r = np.isfinite(Xr).all(axis=1)
    mask_s = np.isfinite(Xs).all(axis=1)
    Xr = Xr[mask_r]
    Xs = Xs[mask_s]

    # Compute correlation matrices
    Corr_r = np.corrcoef(Xr, rowvar=False)  # Shape (D, D)
    Corr_s = np.corrcoef(Xs, rowvar=False)

    # Replace NaNs/infinities with zeros (in case of constant features)
    Corr_r = np.nan_to_num(Corr_r, nan=0.0, posinf=0.0, neginf=0.0)
    Corr_s = np.nan_to_num(Corr_s, nan=0.0, posinf=0.0, neginf=0.0)

    # Difference metrics
    delta = Corr_r - Corr_s
    frob = float(np.linalg.norm(delta, ord='fro'))
    max_abs = float(np.max(np.abs(delta)))
    mae = float(np.mean(np.abs(delta))) 

    metrics = {
        "corr_frobenius_diff": frob,
        "corr_max_abs_diff": max_abs,
        "corr_mae": mae,
        "corr_method": "pearson",
    }

    return metrics


def _acf_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Compute autocorrelation function for a 1D array x up to max_lag (inclusive).
    Returns array of length max_lag + 1, with acf[0] = 1.0.

    If variance is ~0 (constant series), returns zeros except acf[0] = 1.0.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if n <= 1:
        return np.ones(max_lag + 1, dtype=float)

    x = x - np.mean(x)
    var = np.mean(x ** 2)
    if var < 1e-12:
        acf = np.zeros(max_lag + 1, dtype=float)
        acf[0] = 1.0
        return acf

    acf = np.empty(max_lag + 1, dtype=float)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        # truncate overlapping part for this lag
        s1 = x[:-lag]
        s2 = x[lag:]
        acf[lag] = np.mean(s1 * s2) / var
    return acf


def test_acf(real, synth, max_lag: int = 8):
    """
    Autocorrelation function (ACF) comparison, test of temporal structury within features

    Purpose:
        - Do autocorrelation structures match between real and synthetic data

    Tests:
        - ACF for each window, then average accross windows
        - Compute RMSE between real and synthetic ACFs for each feature

    Inputs: 
        real - numpy array of real data (N, L, D)
        synth - numpy array of synthetic data (N, L, D)
        max_lag - int, maximum lag to compute ACF for

    Outpus:
        - acf_rmse_mean: float mean RMSE across features
        - acf_rmse_per_feature: list[float length D]
    """
    N_r, L, D = real.shape
    N_s, _, _ = synth.shape

    # Clip max_lag to at most L-1
    max_lag_eff = int(min(max_lag, L - 1))
    if max_lag_eff < 1:
        raise ValueError(f"max_lag must be >= 1 and < L; got max_lag={max_lag}, L={L}")

    # We'll store average ACF over windows for each feature
    acf_real = np.zeros((D, max_lag_eff), dtype=float)   # lags 1..max_lag_eff
    acf_synth = np.zeros((D, max_lag_eff), dtype=float)

    # Count how many windows contributed per feature (in case we skip any)
    count_real = np.zeros(D, dtype=int)
    count_synth = np.zeros(D, dtype=int)

    # Real data
    for n in range(N_r):
        for d in range(D):
            x = real[n, :, d]
            if not np.all(np.isfinite(x)):
                continue
            acf_vals = _acf_1d(x, max_lag_eff)  # length max_lag_eff+1
            acf_real[d, :] += acf_vals[1:]      # skip lag 0 (always 1)
            count_real[d] += 1

    # Synthetic data
    for n in range(N_s):
        for d in range(D):
            x = synth[n, :, d]
            if not np.all(np.isfinite(x)):
                continue
            acf_vals = _acf_1d(x, max_lag_eff)
            acf_synth[d, :] += acf_vals[1:]
            count_synth[d] += 1

    # Avoid division by zero: if a feature gets no valid windows, leave ACF as zeros
    for d in range(D):
        if count_real[d] > 0:
            acf_real[d, :] /= count_real[d]
        if count_synth[d] > 0:
            acf_synth[d, :] /= count_synth[d]

    # RMSE per feature between average real vs synthetic ACF curves
    rmse_per_feature = []
    for d in range(D):
        diff = acf_real[d, :] - acf_synth[d, :]
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        rmse_per_feature.append(rmse)

    acf_rmse = float(np.mean(rmse_per_feature))

    metrics = {
        "acf_rmse": acf_rmse,
        "acf_rmse_per_feature": rmse_per_feature,
        "acf_lags": list(range(1, max_lag_eff + 1)),
    }
    return metrics


def test_discriminative(real, synth, test_size: float = 0.3, random_state: int = 42):
    """
    Discriminative score (classifier accuracy between real and synthetic)

    Purpose:
        - Can a classifier distinguish real vs synthetic data?
        - If accuracy is close to 50%, synthetic data is realistic
    
    Inputs:
        real - numpy array of real data (N, L, D)
        synth - numpy array of synthetic data (N, L, D)

    Outputs:
        - accuracy: float classifier accuracy on test set
    """
    N_r, L, D = real.shape
    N_s, _, _ = synth.shape

    # Flatten windows: (N, L*D)
    X_real = real.reshape(N_r, L * D)
    X_synth = synth.reshape(N_s, L * D)

    # Labels: real=1, synthetic=0
    y_real = np.ones(N_r, dtype=int)
    y_synth = np.zeros(N_s, dtype=int)

    # Combine
    X = np.concatenate([X_real, X_synth], axis=0)
    y = np.concatenate([y_real, y_synth], axis=0)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Simple classifier: Logistic Regression
    # solver 'lbfgs' works fine for small/medium problems; increase max_iter just in case
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Evaluate
    y_proba = clf.predict_proba(X_test)[:, 1]  # probability of "real" (label 1)
    y_pred = clf.predict(X_test)

    auc = float(roc_auc_score(y_test, y_proba))
    acc = float(accuracy_score(y_test, y_pred))

    metrics = {
        "disc_auc": auc,
        "disc_acc": acc,
        "disc_test_size": float(test_size),
    }
    return metrics


def test_predictive(real, synth):
    """
    Predictive score (train on synthetic, test on real)

    Purpose:
        - Test utility of synthetic data for training downstream models
        - Train a one-step ahead predictor on synthetic data
        - Test performance on real data

    Inputs:
        real - numpy array of real data (N, L, D)
        synth - numpy array of synthetic data (N, L, D)

    Outputs:
        - mse: float mean squared error on real test set
        - mse_per_feature: list[float length D]
    """
    N_r, L, D = real.shape
    N_s, _, _ = synth.shape

    if L < 2:
        raise ValueError(f"Need at least 2 timesteps for prediction task, got L={L}")

    # Build supervised pairs:
    # X = flattened first L-1 timesteps of ALL features -> shape (N, (L-1)*D)
    # y_d = value of feature d at last timestep
    def build_X_y(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        N, L_, D_ = arr.shape
        X = arr[:, :-1, :].reshape(N, (L_ - 1) * D_)  # (N, (L-1)*D)
        y = arr[:, -1, :]                             # (N, D) targets for each feature
        return X, y

    X_real, y_real_all = build_X_y(real)   # y_real_all: (N_r, D)
    X_synth, y_synth_all = build_X_y(synth)

    tstr_mse_per_feature = []

    # For each feature d, train a separate regressor on synthetic and test on real
    for d in range(D):
        y_train = y_synth_all[:, d]   # synthetic targets
        y_test  = y_real_all[:, d]    # real targets

        # Simple linear model
        model = LinearRegression()
        model.fit(X_synth, y_train)

        y_pred = model.predict(X_real)
        mse_d = float(mean_squared_error(y_test, y_pred))
        tstr_mse_per_feature.append(mse_d)

    tstr_mse_mean = float(np.mean(tstr_mse_per_feature))

    metrics = {
        "tstr_mse_mean": tstr_mse_mean,
        "tstr_mse_per_feature": tstr_mse_per_feature,
        "tstr_model": "LinearRegression",
        "tstr_input_type": "flattened_(L-1)*D_all_features",
    }
    return metrics

def test_knn_novelty(real, synth, standardize: bool = True):
    """
    Novelty / coverage via nearest neighbors

    Purpose:
        - Are synthetic samples novel, or just copies of real data?
        - Do synthetic samples cover the real data distribution?
        - Compare distributions of nearest neighbor distances

    Inputs:
        real - numpy array of real data (N, L, D)
        synth - numpy array of synthetic data (N, L, D)

    Outputs:
        - knn_synth_mean: float mean nearest neighbor distance from synthetic to real
        - knn_real_mean: float mean nearest neighbor distance from real to synthetic
        - knn_asymmetry: float |knn_synth_mean - knn_real_mean|
    """
    N_r, L, D = real.shape
    N_s, _, _ = synth.shape

    # Flatten windows: (N, L*D)
    real_flat = real.reshape(N_r, L * D)
    synth_flat = synth.reshape(N_s, L * D)

    # Optionally standardize combined data so distances are more balanced across dimensions
    if standardize:
        scaler = StandardScaler()
        combined = np.vstack([real_flat, synth_flat])
        scaler.fit(combined)
        real_flat = scaler.transform(real_flat)
        synth_flat = scaler.transform(synth_flat)

    # 1) Distances from synthetic to real
    nn_real = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn_real.fit(real_flat)
    d_s2r, _ = nn_real.kneighbors(synth_flat)  # shape (N_s, 1)
    d_s2r = d_s2r[:, 0]

    # 2) Distances from real to synthetic
    nn_synth = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn_synth.fit(synth_flat)
    d_r2s, _ = nn_synth.kneighbors(real_flat)  # shape (N_r, 1)
    d_r2s = d_r2s[:, 0]

    mean_s2r = float(np.mean(d_s2r))
    mean_r2s = float(np.mean(d_r2s))
    med_s2r = float(np.median(d_s2r))
    med_r2s = float(np.median(d_r2s))

    metrics = {
        "knn_mean_synth_to_real": mean_s2r,
        "knn_median_synth_to_real": med_s2r,
        "knn_mean_real_to_synth": mean_r2s,
        "knn_median_real_to_synth": med_r2s,
        "knn_asymmetry_mean": float(mean_s2r - mean_r2s),
        "knn_asymmetry_median": float(med_s2r - med_r2s),
        "knn_standardized": bool(standardize),
    }
    return metrics


def main():
    real_path = "artifacts/baseline_v0/train_orig.npy"
    synth_path = "artifacts/baseline_v0/synthetic_orig.npy"

    real, synth = load_data(real_path, synth_path, match_shapes=True)

    # Check for successful loading
    # print(f"Loaded real data shape: {real.shape}")
    # print(f"Loaded synthetic data shape: {synth.shape}")

