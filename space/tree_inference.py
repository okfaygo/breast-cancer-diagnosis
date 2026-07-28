"""
Dependency-free inference for the random forest and gradient boosting models.

Both are tree ensembles, so a prediction is just walking each tree to a leaf. The
tree structures are exported to trees_bundle.npz by export_trees.py; this module
walks them in NumPy so the app needs neither scikit-learn nor TensorFlow.

Storage layout: every tree's node arrays are concatenated into flat arrays, with an
`*_offsets` array marking each tree's slice. Child indices are local to each tree's
slice. This mirrors sklearn's `tree_` arrays (split on `X[feature] <= threshold`
goes left).
"""

from __future__ import annotations

import numpy as np

BUNDLE_PATH = "trees_bundle.npz"


def load_trees(path: str = BUNDLE_PATH) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _leaf_indices(X, feature, threshold, left, right) -> np.ndarray:
    """Walk one tree (local arrays) and return the leaf node index per sample."""
    node = np.zeros(X.shape[0], dtype=np.int64)
    while True:
        is_leaf = feature[node] < 0            # sklearn marks leaves with feature = -2
        if is_leaf.all():
            return node
        rows = np.where(~is_leaf)[0]
        current = node[rows]
        go_left = X[rows, feature[current]] <= threshold[current]
        node[rows] = np.where(go_left, left[current], right[current])


def _ensemble_leaf_payload(X, trees, prefix, payload_key) -> np.ndarray:
    """Sum a per-leaf quantity (probability or value) across every tree."""
    # scikit-learn casts inputs to float32 before testing splits, so samples right on
    # a threshold can branch differently in float64. Round to float32 to match exactly.
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]
    offsets = trees[f"{prefix}_offsets"]
    feature = trees[f"{prefix}_feature"]
    threshold = trees[f"{prefix}_threshold"]
    left = trees[f"{prefix}_left"]
    right = trees[f"{prefix}_right"]
    payload = trees[payload_key]

    total = np.zeros(X.shape[0], dtype="float64")
    n_trees = len(offsets) - 1
    for t in range(n_trees):
        s, e = int(offsets[t]), int(offsets[t + 1])
        leaves = _leaf_indices(X, feature[s:e], threshold[s:e], left[s:e], right[s:e])
        total += payload[s:e][leaves]
    return total, n_trees


def rf_predict_proba(X, trees) -> np.ndarray:
    """Random forest P(malignant): average of the per-tree leaf probabilities."""
    total, n_trees = _ensemble_leaf_payload(X, trees, "rf", "rf_prob")
    return total / n_trees


def gb_predict_proba(X, trees) -> np.ndarray:
    """Gradient boosting P(malignant): sigmoid of the additive raw score."""
    total, _ = _ensemble_leaf_payload(X, trees, "gb", "gb_value")
    raw = float(trees["gb_F0"]) + float(trees["gb_lr"]) * total
    return 1.0 / (1.0 + np.exp(-raw))
