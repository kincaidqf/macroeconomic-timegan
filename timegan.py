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

    seq_len, feature_dim = infer_dims(train_set)
    hidden_dim = int(params["hidden_dim"])
    num_layers = int(params["num_layers"])
    module = str(params["module"]).lower()
    iterations = int(params["iterations"])
    batch_size = int(params["batch_size"])
    lr = float(params["learning_rate"])
    gamma = float(params["gamma"])
    z_dim = int(params["z_dim"] or feature_dim)

    X_ph, Z_ph = build_placeholders(seq_len, feature_dim, z_dim)

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

