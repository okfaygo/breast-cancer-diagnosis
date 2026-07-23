"""
Dependency-free inference for the WDBC model.

The trained network is a small MLP (30 -> 64 relu -> 32 relu -> 1 sigmoid), so a
forward pass is three matrix multiplies. Keeping this in NumPy means the deployed
app needs neither TensorFlow nor scikit-learn — it loads in well under a second.

The weights come from model_bundle.npz, produced by export_bundle.py.
"""

from __future__ import annotations

import numpy as np

BUNDLE_PATH = "model_bundle.npz"


def load_bundle(path: str = BUNDLE_PATH) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def predict_proba(x: np.ndarray, bundle: dict[str, np.ndarray]) -> np.ndarray:
    """Probability of malignancy for raw (unscaled) feature rows.

    x: array of shape (n_samples, 30) in the bundle's feature order.
    """
    x = np.asarray(x, dtype="float64")
    if x.ndim == 1:
        x = x[None, :]

    # Same standardisation the scaler applied during training
    z = (x - bundle["scaler_mean"]) / bundle["scaler_scale"]

    h1 = np.maximum(0.0, z @ bundle["w1"] + bundle["b1"])       # relu
    h2 = np.maximum(0.0, h1 @ bundle["w2"] + bundle["b2"])      # relu
    logits = h2 @ bundle["w3"] + bundle["b3"]
    probs = 1.0 / (1.0 + np.exp(-logits))                        # sigmoid
    return probs.flatten()
