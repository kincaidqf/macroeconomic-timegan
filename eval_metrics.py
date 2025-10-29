import numpy as np

def load_data(real_path, synth_path):
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

    # TODO: load arrays
    # TODO: check shapes match
    pass

def test_marginals(real, synth):
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
    pass

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
    pass


def test_acf(real, synth):
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
    pass

def test_discriminative(real, synth):
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
    pass

def test_predictive(real, synth):
    pass

def test_knn_novelty(real, synth):
    pass

def main():
    pass