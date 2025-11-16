import numpy as np
from pathlib import Path

base0 = Path("artifacts/baseline_v0")
base1 = Path("artifacts/baseline_v1")
base2 = Path("artifacts/baseline_v1")

s0 = np.load(base0 / "synthetic_scaled.npy")
s1 = np.load(base1 / "synthetic_scaled.npy")
s2 = np.load(base2 / "synthetic_scaled.npy")

print("Shapes:", s0.shape, s1.shape, s2.shape)

print("s0 vs s1 allclose:", np.allclose(s0, s1))
print("s0 vs s2 allclose:", np.allclose(s0, s2))
print("s1 vs s2 allclose:", np.allclose(s1, s2))

print("s0 mean/std:", s0.mean(), s0.std())
print("s1 mean/std:", s1.mean(), s1.std())
print("s2 mean/std:", s2.mean(), s2.std())
