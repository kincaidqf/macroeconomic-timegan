from pathlib import Path
import numpy as np
import tensorflow as tf

from prep_windows import prepare_windows
from timegan import timegan
from utils import sample_batch

tf.compat.v1.disable_eager_execution()

def main():
    return

def ae_warmup_test():
    # 1) Load data (defaults: L=24, stride=1, val=Country7, test=Country8,9)
    train_scaled, val_scaled, test_scaled, (minv, rng), summary = prepare_windows(
        data_dir=Path("data/clean")
    )
    print("Loaded:", summary["counts"]) 

    # 2) Build model graph
    handles = timegan(train_scaled, parameters=None)

    X_ph = handles["placeholders"]["X"]
    ae_loss_t = handles["losses"]["ae_loss"]
    ae_op = handles["train_ops"]["ae"]

    # 3) Train autoencoder on small batch for a few iterations
    batch_size = 64
    steps = 300

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for it in range(1, steps + 1):
            Xb = sample_batch(train_scaled, batch_size)  # (batch, L, D)
            loss, _ = sess.run([ae_loss_t, ae_op], feed_dict={X_ph: Xb})

            if it % 50 == 0 or it == 1:
                print(f"Step {it:>4}/{steps} | ae_loss: {loss:.6f}")

    '''
    Result of Test:
    Loaded: {'train_windows': 155, 'val_windows': 31, 'test_windows': 61}
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1760502852.850892 38347149 mlir_graph_optimization_pass.cc:437] MLIR V1 optimization pass is not enabled
    Step    1/300 | ae_loss: 0.054511
    Step   50/300 | ae_loss: 0.024929
    Step  100/300 | ae_loss: 0.018569
    Step  150/300 | ae_loss: 0.010809
    Step  200/300 | ae_loss: 0.008263
    Step  250/300 | ae_loss: 0.007677
    Step  300/300 | ae_loss: 0.006710
    '''

def params_test():
    # 1) Load windows (defaults: L=24, stride=1, val=Country7, test=Country8,9)
    train_scaled, val_scaled, test_scaled, (minv, rng), summary = prepare_windows(
        data_dir=Path("data/clean")
    )
    print("Loaded:", summary["counts"])

    # 2) Build graph via timegan()
    handles = timegan(train_scaled, parameters=None)

    X_ph = handles["placeholders"]["X"]
    Z_ph = handles["placeholders"]["Z"]
    H_real = handles["tensors"]["H_real"]
    X_hat = handles["tensors"]["X_hat"]
    H_tilde_sup = handles["tensors"]["H_tilde_sup"]
    logits_real = handles["tensors"]["logits_real"]
    logits_fake = handles["tensors"]["logits_fake"]

    L = handles["params"]["seq_len"]
    D = handles["params"]["feature_dim"]
    z_dim = D  # you set z_dim to feature_dim in timegan

    # 3) Make a tiny fake batch
    batch = 8
    Xb = np.stack([train_scaled[i] for i in range(batch)], axis=0)  # (batch, L, D)
    Zb = np.random.normal(0, 1, size=(batch, L, z_dim)).astype("float32")

    # 4) Session init + forward pass
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        out = sess.run(
            {
                "H_real": H_real,
                "X_hat": X_hat,
                "H_tilde_sup": H_tilde_sup,
                "logits_real": logits_real,
                "logits_fake": logits_fake,
            },
            feed_dict={X_ph: Xb, Z_ph: Zb},
        )

        for k, v in out.items():
            print(f"{k:>12} shape:", v.shape)

        # Optional: check variable groups
        for scope, vars_ in handles["vars"].items():
            print(f"{scope:>12} vars:", len(vars_))

    """
    Result of test:
    Loaded: {'train_windows': 155, 'val_windows': 31, 'test_windows': 61}
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1760481613.924437 38171673 mlir_graph_optimization_pass.cc:437] MLIR V1 optimization pass is not enabled
    H_real shape: (8, 24, 24)
    X_hat shape: (8, 24, 5)
    H_tilde_sup shape: (8, 24, 24)
    logits_real shape: (8, 1)
    logits_fake shape: (8, 1)
    embedder vars: 8
    recovery vars: 8
    generator vars: 8
    supervisor vars: 8
    discriminator vars: 16
    """


if __name__ == "__main__":
    main()
