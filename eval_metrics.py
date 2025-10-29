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
    pass

def test_correlation(real, synth):
    pass

def test_acf(real, synth):
    pass

def test_discriminative(real, synth):
    pass

def test_predictive(real, synth):
    pass

def test_knn_novelty(real, synth):
    pass

def main():
    pass