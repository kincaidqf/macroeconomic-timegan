from pathlib import Path
import numpy as np
import tensorflow as tf
import json

from prep_windows import prepare_windows
from timegan import timegan
from utils import sample_batch

tf.compat.v1.disable_eager_execution()

def main():
    # 1) Load and summarize data
    train_scaled, val_scaled, test_scaled, (minv, rng), summary = prepare_windows(
        data_dir=Path("data/clean"),  # adjust if needed
        L=24,
        stride=1,
        val_countries=["Country7"],
        test_countries=["Country8", "Country9"],
    )
    print("Loaded:", summary["counts"])
    L = summary["shapes"]["window_length"]
    D = summary["shapes"]["feature_count"]
    z_dim = D  # we set z_dim = feature_dim in timegan

    # 2) Build graph
    handles = timegan(train_scaled, parameters=None)

    X_ph = handles["placeholders"]["X"]
    Z_ph = handles["placeholders"]["Z"]

    # AE tensors/op
    ae_loss_t = handles["losses"]["ae_loss"]
    ae_op     = handles["train_ops"]["ae"]

    # GAN tensors/ops
    d_loss_t  = handles["losses"]["d_loss"]
    g_loss_t  = handles["losses"]["g_loss"]
    sup_loss_t= handles["losses"]["sup_loss"]
    d_op      = handles["train_ops"]["d"]
    g_op      = handles["train_ops"]["g"]

    X_hat_t   = handles["tensors"]["X_hat"]

    # 3) Train
    batch_size   = 64
    ae_warmup_it = 600     # 300–1000 is typical; increase if recon not improving
    gan_iters    = 2000    # tune as needed (2k–10k); watch losses

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        # AE pretrain (embedder + recovery only) 
        print("\n[Stage 1] AE warm-up")
        for it in range(1, ae_warmup_it + 1):
            Xb = sample_batch(train_scaled, batch_size)
            loss, _ = sess.run([ae_loss_t, ae_op], feed_dict={X_ph: Xb})
            if it % 100 == 0 or it == 1:
                print(f"[AE] iter {it:4d}  loss={loss:.6f}")

        # Small sanity check
        Xb = sample_batch(train_scaled, 4)
        Xb_hat = sess.run(X_hat_t, feed_dict={X_ph: Xb})
        print("Recon check shapes:", Xb.shape, "->", Xb_hat.shape)

        # GAN phase (D/G alternating) 
        print("\n[Stage 2] GAN + supervised training")
        for it in range(1, gan_iters + 1):
            # 1) Discriminator step
            Xb = sample_batch(train_scaled, batch_size)
            Zb = np.random.normal(0, 1, size=(batch_size, L, z_dim)).astype("float32")
            _ = sess.run(d_op, feed_dict={X_ph: Xb, Z_ph: Zb})

            # 2) Generator/Supervisor steps (do twice per original spirit)
            for _ in range(2):
                Xb = sample_batch(train_scaled, batch_size)
                Zb = np.random.normal(0, 1, size=(batch_size, L, z_dim)).astype("float32")
                _ = sess.run(g_op, feed_dict={X_ph: Xb, Z_ph: Zb})

            # Logging
            if it % 100 == 0 or it == 1:
                Xb = sample_batch(train_scaled, batch_size)
                Zb = np.random.normal(0, 1, size=(batch_size, L, z_dim)).astype("float32")
                d_val, g_val, s_val = sess.run(
                    [d_loss_t, g_loss_t, sup_loss_t],
                    feed_dict={X_ph: Xb, Z_ph: Zb}
                )
                print(f"[GAN] iter {it:4d} | d_loss={d_val:.4f}  g_loss={g_val:.4f}  sup={s_val:.4f}")

        N = len(train_scaled)

        L = summary["shapes"]["window_length"]
        D = summary["shapes"]["feature_count"]
        z_dim = D  # we set z_dim = feature_dim in timegan  

        np.random.seed(42)

        # sample noise to pass in to generator
        Zb = np.random.normal(0, 1, size=(N, L, z_dim)).astype("float32")

        # Run synthetic data generating tensor
        X_scaled_synth = sess.run(
            handles["tensors"]["X_from_Z"],
            feed_dict={handles["placeholders"]["Z"]: Zb}
        ) # shape (N, L, D) scaled [0,1]

        X_synth_orig = X_scaled_synth * rng + minv  # reverse scaling

        out_dir = Path("artifacts/baseline_v1")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(out_dir / "synthetic_orig.npy", X_synth_orig)
        np.save(out_dir / "synthetic_scaled.npy", X_scaled_synth)

        # Stack windows into 3D arrays (N, L, D)
        train_orig = np.stack(train_scaled, axis=0) * rng + minv
        test_orig = np.stack(test_scaled, axis=0) * rng + minv
        val_orig = np.stack(val_scaled, axis=0) * rng + minv
        
        np.save(out_dir / "train_orig.npy", train_orig)
        np.save(out_dir / "test_orig.npy", test_orig)
        np.save(out_dir / "val_orig.npy", val_orig)

        # Stack windows into 3D arrays (N, L, D)
        train_scaled_arr = np.stack(train_scaled, axis=0).astype(np.float32)
        test_scaled_arr = np.stack(test_scaled, axis=0).astype(np.float32)
        val_scaled_arr = np.stack(val_scaled, axis=0).astype(np.float32)
        
        np.save(out_dir / "train_scaled.npy", train_scaled_arr)
        np.save(out_dir / "test_scaled.npy", test_scaled_arr)
        np.save(out_dir / "val_scaled.npy", val_scaled_arr)

        cfg = {
            "L": L,
            "D": D,
            "z_dim": z_dim,
            "batch_size": batch_size,
            "ae_warmup_it": ae_warmup_it,
            "gan_iters": gan_iters,
            "gamma": 1.0
        }

        (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

        print("Synthetic (scaled) min/max:", float(X_scaled_synth.min()), float(X_scaled_synth.max()))
        print("Synthetic (orig)   mean/std per feature (first window):",
            X_synth_orig[0].mean(axis=0), X_synth_orig[0].std(axis=0))
        print("Saved to:", out_dir.resolve())



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
