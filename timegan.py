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
    
    def names(vars_): return sorted({v.name for v in vars_})

    def assert_disjoint(*var_groups):
        for i in range(len(var_groups)):
            for j in range(i+1, len(var_groups)):
                A = {v.name for v in var_groups[i]}
                B = {v.name for v in var_groups[j]}
                inter = A & B
                if inter:
                    raise RuntimeError(f"Variable lists overlap between groups {i} and {j}:\n" +
                                    "\n".join(sorted(inter)))

    
    def sample_synthetic(sess, Z_ph, X_tensors, num_samples, seq_len, z_dim):
        # X_tensors should give you access to the ops you need:
        #   generator(Z_ph) -> H_tilde
        #   supervisor(H_tilde) -> H_tilde_sup
        #   recovery(H_tilde_sup) -> X_tilde
        # If you already have H_tilde_sup and X_hat tensors wired, you can reuse them.
        Zb = np.random.normal(0, 1, size=(num_samples, seq_len, z_dim)).astype("float32")
        X_synth = sess.run(X_tensors["X_from_Z"], feed_dict={Z_ph: Zb})

        return X_synth

    # CALLING NETWORKS
    
    # Each input has shape (batch, seq_len, hidden_dim)

    # Autoencoder path (embedder -> recovery): X -> H -> X_hat
    H_real = embedder(X_ph)          
    X_hat = recovery(H_real)        
    H_hat = supervisor(H_real)
    # Synthetic latent path
    H_tilde = generator(Z_ph)      
    H_tilde_sup = supervisor(H_tilde)  
    # Discriminator logits for real and fake
    logits_real = discriminator(H_real)          # (batch, 1)
    logits_fake = discriminator(H_tilde_sup)     # (batch, 1)

    # LOSS FUNCTIONS
    # Reconstruction loss on autoencoder (MSE)
    ae_loss = tf.reduce_mean(tf.square(X_ph - X_hat), name="ae_loss")
    
    # Shifted MSE (supervised loss): H_real[:, 1:, :] vs H_hat[:, :-1, :]
    sup_loss = tf.reduce_mean(
        tf.square(H_real[:, 1:, :] - H_hat[:, :-1, :]),
        name="supervised_loss"
    )
    
    # Discriminator loss: BCE with logits 
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits_real, labels=tf.ones_like(logits_real)
        )
    )
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits_fake, labels=tf.ones_like(logits_fake)
        )
    )
    d_loss = tf.identity(d_loss_real + d_loss_fake, name="d_loss")

    # Generator adversarial loss: make the fake as indistinguishable from real as possible
    g_loss_adv = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits_fake, labels=tf.ones_like(logits_fake)
        )
    )

    # Total generator loss: adversarial + gamma * supervised
    g_loss = tf.identity(g_loss_adv + gamma * sup_loss, name="g_loss")

    # Calling variables here to ensure they are created before being used in optimizers
    e_vars = vars_with_names(["embedder_rnn", "W_e", "b_e"])
    r_vars = vars_with_names(["recovery_rnn", "W_r", "b_r"])
    # Generator & Supervisor
    g_vars = vars_with_names(["generator_rnn", "W_g", "b_g"])
    s_vars = vars_with_names(["supervisor_rnn", "W_s", "b_s"])
    # Discriminator
    d_vars = vars_with_names(["discriminator_rnn", "W_d", "b_d"])

    # Creating the autoencoder optimizer
    ae_train_op = adam.minimize(ae_loss, var_list=e_vars + r_vars)
    # Optimize loss function for Generator
    g_train_op = adam.minimize(g_loss, var_list=g_vars + s_vars)
    # Optimize loss function for Discriminator
    d_train_op = adam.minimize(d_loss, var_list=d_vars)

    # Assert variables don't overlap in a way that would mess up loss functions, test passed first time, uncomment to run
    # assert_disjoint(e_vars, r_vars, g_vars, s_vars, d_vars)

    handles = {
        "placeholders": {"X": X_ph, "Z": Z_ph},
        "tensors": {
            "H_real": H_real,
            "H_hat": H_hat,
            "H_tilde": H_tilde,
            "H_tilde_sup": H_tilde_sup,
            "X_hat": X_hat,
            "logits_real": logits_real,
            "logits_fake": logits_fake,
        },
        "losses": {
            "ae_loss": ae_loss,
            "sup_loss": sup_loss,
            "d_loss": d_loss,
            "g_loss": g_loss,
            "g_loss_adv": g_loss_adv,
        },
        "train_ops": {
            "ae": ae_train_op,
            "d": d_train_op,
            "g": g_train_op,
        },
        "params": {
            "seq_len": seq_len,
            "feature_dim": feature_dim,
        },
    }
    return handles

