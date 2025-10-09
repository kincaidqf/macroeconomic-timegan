from typing import List, Dict
import numpy as np
import tensorflow as tf

from utils import (
    set_random_seed, infer_dims, sample_batch, sample_noise,
    stacked_rnn, xavier_init, build_placeholders, make_optimizer,
    DEFAULT_PARAMS
)

tf.compat.v1.disable_eager_execution() # Use TF1-style execution
"""
Original TimeGAN implementation used TF1-style sessions.
- Allos for static graph definition of the model with placeholders for inputs
- Graph is run inside a session which doesn't execute until it is fed data
- Easier to match original implementation and debug issues

TF2 eager execution instantly runs operations, would be difficult to adapt og implementation to TF2
"""

def timegan(train_set: List[np.ndarray], parameters: Dict = None):
    # Placeholder for TimeGAN implementation
    set_random_seed(42)
    params = dict(DEFAULT_PARAMS)
    if parameters:
        params.update(parameters)

    # Length of each window (number of time steps) and feature dimension (number of features)
    seq_len, feature_dim = infer_dims(train_set)
    # Number of units in each layer of the neural networks (length of hidden state vector, 24 in this case)
    hidden_dim = int(params["hidden_dim"])
    # Count of layers in the neural networks
    num_layers = int(params["num_layers"])
    # Type of RNN cell to use (gru or lstm)
    module = str(params["module"]).lower()
    # Number of parameter update steps
    iterations = int(params["iterations"])
    # Number of (L x D) windows in each batch
    batch_size = int(params["batch_size"])
    # Learning rate, adaptive based on Adam optimizer
    lr = float(params["learning_rate"])
    # Hyperparameter to balance weightin of supervised loss and unsupervised loss
        # 1 = equal weighting, >1 = more weight on supervised loss, <1 = more weight on unsupervised loss
    gamma = float(params["gamma"])
    # Latent space dimension, if None set to feature_dim
    z_dim = int(params["z_dim"] or feature_dim)

    # Placeholders for real data (X) and random noise (Z)
    X_ph, Z_ph = build_placeholders(seq_len, feature_dim, z_dim)

    # Learning rate optimizer
    adam = make_optimizer(lr)

    def embedder(X, T):
        return
    
    def recovery(H, T):
        return
    
    def generator(Z, T):
        return
    
    def supervisor(H, T):
        return
    
    def discriminator(H, T):
        return
    
    generated_data = list()

    return generated_data

