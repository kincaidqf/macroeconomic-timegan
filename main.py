from pathlib import Path
import numpy as np
import tensorflow as tf

from prep_windows import prepare_windows
from timegan import timegan

tf.compat.v1.disable_eager_execution()

def main():
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

if __name__ == "__main__":
    main()
