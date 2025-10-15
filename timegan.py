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

    def embedder(X):
        # Don't need T because sliding windows all have the same length of seq_len

        """
        X: real sequences (batch, seq_len, feature_dim)
        Returns H: embedded (latent) real sequences (batch, seq_len, hidden_dim)
        """

        with tf.compat.v1.variable_scope("embedder", reuse=tf.compat.v1.AUTO_REUSE):
            # 1) Temporal encoder: stacked RNN over timesteps
            outputs = stacked_rnn(X, hidden_dim, num_layers, module, scope="embedder_rnn")

            # 2) Per-timestep projection to latent space
            # Flatten time+batch for one dense matrix multiplication (matmul) then reshape back
            flat = tf.reshape(outputs, [-1, hidden_dim])  # (batch*seq_len, hidden_dim)

            # Dense projection (hidden_dim -> hidden_dim) with Xavier init
            W_e = tf.Variable(xavier_init([hidden_dim, hidden_dim]), name="W_e")
            b_e = tf.Variable(tf.zeros([hidden_dim], dtype=tf.float32), name="b_e")

            # Activation: sigmoid keeps H in [0,1] which is the same range as [0,1] scaled input data
            h_flat = tf.nn.sigmoid(tf.matmul(flat, W_e) + b_e)  # (batch*seq_len, hidden_dim)

            # Reshape back to sequence form
            H = tf.reshape(h_flat, [-1, seq_len, hidden_dim])  #
            return H
    
    def recovery(H):
        """
        H: latent sequences (from embedder) 
        Returns X_hat: reconstructed sequences from H (batch, L, feature_dim)
        """
        with tf.compat.v1.variable_scope("recovery", reuse=tf.compat.v1.AUTO_REUSE):
            # 1) Stacked RNN over timesteps
            outputs = stacked_rnn(H, hidden_dim, num_layers, module, scope="recovery_rnn")

            # 2) Per-timestep projection to original feature space
            flat = tf.reshape(outputs, [-1, hidden_dim])  # (batch*seq_len, hidden_dim)

            # Dense projection (hidden_dim -> feature_dim) with Xavier init
            W_r = tf.Variable(xavier_init([hidden_dim, feature_dim]), name="W_r")
            b_r = tf.Variable(tf.zeros([feature_dim], dtype=tf.float32), name="b_r")

            # Activation: sigmoid keeps X_hat in [0,1] which is the same range as [0,1] scaled input data
            x_hat_flat = tf.nn.sigmoid(tf.matmul(flat, W_r) + b_r)  # (batch*seq_len, feature_dim)

            # Reshape back to sequence form
            X_hat = tf.reshape(x_hat_flat, [-1, seq_len, feature_dim])  # (batch, seq_len, feature_dim)
            return X_hat
        
    
    def generator(Z):
        """
        Z: noise sequences (batch, L, z_dim)
        Returns H_tilde: generated latent sequences (batch, L, hidden_dim)
        """
        with tf.compat.v1.variable_scope("generator", reuse=tf.compat.v1.AUTO_REUSE):
            # 1) Stacked RNN over timesteps
            g_outputs = stacked_rnn(Z, hidden_dim, num_layers, module, scope="generator_rnn")

            # 2) Per-timestep projection to latent space
            g_flat = tf.reshape(g_outputs, [-1, hidden_dim])  # (batch*seq_len, hidden_dim)

            # Dense projection (hidden_dim -> hidden_dim) with Xavier init
            W_g = tf.Variable(xavier_init([hidden_dim, hidden_dim]), name="W_g")
            b_g = tf.Variable(tf.zeros([hidden_dim], dtype=tf.float32), name="b_g")

            # Activation: sigmoid keeps H_tilde in [0,1] which is the same range as [0,1] scaled input data
            h_tilde_flat = tf.nn.sigmoid(tf.matmul(g_flat, W_g) + b_g)  # (batch*seq_len, hidden_dim)

            # Reshape back to sequence form
            H_tilde = tf.reshape(h_tilde_flat, [-1, seq_len, hidden_dim]) # (batch, seq_len, hidden_dim)
            return H_tilde

    
    def supervisor(H):
        """
        H: latent sequences (from embedder)
        Returns H_hat: supervised latent sequences (batch, seq_len, hidden_dim)
        """

        with tf.compat.v1.variable_scope("supervisor", reuse=tf.compat.v1.AUTO_REUSE):
            # 1) Stacked RNN over timesteps
            s_outputs = stacked_rnn(H, hidden_dim, num_layers, module, scope="supervisor_rnn")

            # 2) Per-timestep projection to latent space
            s_flat = tf.reshape(s_outputs, [-1, hidden_dim])  # (batch*seq_len, hidden_dim)

            W_s = tf.Variable(xavier_init([hidden_dim, hidden_dim]), name="W_s")
            b_s = tf.Variable(tf.zeros([hidden_dim], dtype=tf.float32), name="b_s")

            hhat_flat = tf.nn.sigmoid(tf.matmul(s_flat, W_s) + b_s)  # (batch*seq_len, hidden_dim)
            H_hat = tf.reshape(hhat_flat, [-1, seq_len, hidden_dim])  # (batch, seq_len, hidden_dim)
            return H_hat
    
    def discriminator(H_in):
        """
        H_in: latent sequences (from embedder or generator)
        Returns logits: shape (batch, 1)
        """

        with tf.compat.v1.variable_scope("discriminator", reuse=tf.compat.v1.AUTO_REUSE):
            # 1) Stacked RNN over timesteps
            d_outputs = stacked_rnn(H_in, hidden_dim, num_layers, module, scope="discriminator_rnn")

            # 2) Sequence summary: take last timestep's hidden state
            last = d_outputs[:, -1, :]  # (batch, hidden_dim)

            # 3) Linear head to single logit (used for stability instead of sigmoid)
            W_d = tf.Variable(xavier_init([hidden_dim, 1]), name="W_d")
            b_d = tf.Variable(tf.zeros([1], dtype=tf.float32), name="b_d")

            logits = tf.matmul(last, W_d) + b_d  # (batch, 1)
            return logits

    def vars_with_names(substrings):
        all_vars = tf.compat.v1.trainable_variables()
        picked = []
        for v in all_vars:
            name = v.name
            if any(s in name for s in substrings):
                picked.append(v)
        return picked

    # Autoencoder path (embedder -> recovery): X -> H -> X_hat
    H_real = embedder(X_ph)          # (batch, seq_len, hidden_dim)
    X_hat = recovery(H_real)        # (batch, seq_len, feature_dim)

    # Reconstruction loss (MSE)
    ae_loss = tf.reduce_mean(tf.square(X_ph - X_hat), name="ae_loss")

    H_tilde = generator(Z_ph)      # (batch, seq_len, hidden_dim)
    H_tilde_sup = supervisor(H_tilde)  # (batch, seq_len, hidden_dim)

    # Discriminator logits for real and fake
    logits_real = discriminator(H_real)          # (batch, 1)
    logits_fake = discriminator(H_tilde_sup)     # (batch, 1)

    # Collect variable list for each subnet 
    e_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="embedder")
    r_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="recovery")
    g_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
    s_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="supervisor")
    d_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

    handles = {
        "placeholders": {"X": X_ph, "Z": Z_ph},
        "tensors": {
            "H_real": H_real,
            "X_hat": X_hat,
            "H_tilde": H_tilde,
            "H_tilde_sup": H_tilde_sup,
            "logits_real": logits_real,
            "logits_fake": logits_fake,
        },
        "vars": {
            "embedder": e_vars,
            "recovery": r_vars,
            "generator": g_vars,
            "supervisor": s_vars,
            "discriminator": d_vars,
        },
        "params": {
            "seq_len": seq_len,
            "feature_dim": feature_dim,
        },
    }
    return handles

