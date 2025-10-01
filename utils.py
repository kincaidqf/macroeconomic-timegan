import random
from typing import List, Tuple, Dict
import numpy as np
import tensorflow as tf

# Keep TF1-style behaviour to use sessions like in original TimeGAN repo
tf.compat.v1.disable_eager_execution()

# Set random seeds for reproducibility
def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)


# Data shape helper functions
def infer_dims(train_set: List[np.ndarray]) -> Tuple[int, int, int]:
    """
    Infer dimensions from the first window
    """
    assert len(train_set) > 0, "Training set is empty"
    L, D = train_set[0].shape
    return int(L), int(D)


def sample_batch(data: List[np.ndarray], batch_size: int) -> np.ndarray:
    """
    Sample a random batch of windows from data (n x L x D) where n = batch_size
    """
    idx = np.random.randint(0, len(data), size=batch_size)
    batch = [data[i] for i in idx]
    return np.stack(batch, axis=0).astype(np.float32)


def sample_noise(batch_size: int, seq_len: int, z_dim: int) -> np.ndarray:
    """
    Per-timestep Gaussian noise (batch, seq_len, z_dim)
    """
    return np.random.normal(0.0, 1.0, size=(batch_size, seq_len, z_dim)).astype(np.float32)


# RNN helper functions
def make_rnn_cell(hidden_dim: int, module: str = "gru"):
    if module.lower() == "lstm":
        return tf.compat.v1.nn.rnn_cell.LSTMCell(hidden_dim, activation=tf.nn.tanh)
    else:
        return tf.compat.v1.nn.rnn_cell.GRUCell(hidden_dim, activation=tf.nn.tanh)
    

def stacked_rnn(hidden_dim: int, num_layers: int, module: str):
    cells = [make_rnn_cell(hidden_dim, module) for _ in range(num_layers)]
    return tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells)


# Initializers
def xavier_init(shape):
    """
    Glorot-style normal initializer for dense layers
    """
    std = np.sqrt(2.0 / (shape[0] + shape[1]))
    return tf.compat.v1.random_normal(shape=shape, stddev=std)


def build_placeholders(L: int, D: int, z_dim: int):
    """
    Placeholders:
    - X: real data (None, L, D)
    - Z: random noise (None, L, z_dim)
    """
    X = tf.compat.v1.placeholder(tf.float32, [None, L, D], name="X")
    Z = tf.compat.v1.placeholder(tf.float32, [None, L, z_dim], name="Z")
    return X, Z


def make_optimizer(lr: float = 1e-3):
    return tf.compat.v1.train.AdamOptimizer(learning_rate=lr)


# ----------------------
# Default params
# ----------------------
DEFAULT_PARAMS: Dict = {
    "hidden_dim": 24,
    "num_layers": 2,
    "module": "gru",            # 'gru' or 'lstm'
    "iterations": 10000,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "gamma": 1.0,
    "z_dim": None,              # if None, we set z_dim = feature_dim
    "print_every": 200,
}