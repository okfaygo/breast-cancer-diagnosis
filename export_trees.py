"""
Train the random forest and gradient boosting models and export their trees to
trees_bundle.npz, for the app's model-agreement panel.

Both are trained on the same 70% training split the neural network used (notebook
Stage 7), so the three models are directly comparable. Inference then runs in NumPy
via tree_inference.py — this script verifies that NumPy path matches scikit-learn
across all 569 samples before writing the bundle.

Run after changing the training setup:  python export_trees.py
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

from tree_inference import gb_predict_proba, rf_predict_proba

SEED = 42
OUT_PATH = "trees_bundle.npz"


def training_split():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df["diagnosis"] = df["target"].map({0: "M", 1: "B"})  # flip to Malignant = 1
    features = list(data.feature_names)
    X = df[features].to_numpy(dtype="float64")
    y = (df["diagnosis"] == "M").astype(int).to_numpy()
    # Reproduce the notebook's 70/15/15 split and train on the 70% portion.
    X_temp, _, y_temp, _ = train_test_split(X, y, test_size=0.15, random_state=SEED, stratify=y)
    X_train, _, y_train, _ = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=SEED, stratify=y_temp
    )
    return X, X_train, y_train


def flatten(trees, kind):
    """Concatenate each tree's node arrays; child indices stay local per tree."""
    feature, threshold, left, right, payload, offsets = [], [], [], [], [], [0]
    for est in trees:
        t = est.tree_
        feature.append(t.feature.astype(np.int32))
        threshold.append(t.threshold.astype(np.float64))
        left.append(t.children_left.astype(np.int32))
        right.append(t.children_right.astype(np.int32))
        if kind == "rf":
            counts = t.value[:, 0, :]                       # class counts at each node
            totals = counts.sum(axis=1)
            prob = np.divide(counts[:, 1], totals, out=np.zeros_like(totals), where=totals > 0)
            payload.append(prob)
        else:
            payload.append(t.value[:, 0, 0].astype(np.float64))  # regression leaf value
        offsets.append(offsets[-1] + t.node_count)
    return {
        "feature": np.concatenate(feature),
        "threshold": np.concatenate(threshold),
        "left": np.concatenate(left),
        "right": np.concatenate(right),
        "payload": np.concatenate(payload),
        "offsets": np.array(offsets, dtype=np.int64),
    }


def main():
    X_all, X_train, y_train = training_split()

    rf = RandomForestClassifier(n_estimators=300, random_state=SEED,
                                class_weight="balanced").fit(X_train, y_train)
    gb = GradientBoostingClassifier(random_state=SEED).fit(X_train, y_train)

    rf_arr = flatten(rf.estimators_, "rf")
    gb_trees = [stage[0] for stage in gb.estimators_]
    gb_arr = flatten(gb_trees, "gb")

    # Gradient boosting raw score = F0 + learning_rate * sum(tree outputs).
    # Derive F0 empirically so we don't depend on sklearn's private internals.
    tree_sum = sum(t.predict(X_all) for t in gb_trees)
    f0_arr = gb.decision_function(X_all) - gb.learning_rate * tree_sum
    if f0_arr.std() > 1e-9:
        raise SystemExit(f"F0 not constant (std={f0_arr.std():.2e}) — GB assumption broken")
    f0 = float(f0_arr.mean())

    bundle = {
        "rf_feature": rf_arr["feature"], "rf_threshold": rf_arr["threshold"],
        "rf_left": rf_arr["left"], "rf_right": rf_arr["right"],
        "rf_prob": rf_arr["payload"], "rf_offsets": rf_arr["offsets"],
        "gb_feature": gb_arr["feature"], "gb_threshold": gb_arr["threshold"],
        "gb_left": gb_arr["left"], "gb_right": gb_arr["right"],
        "gb_value": gb_arr["payload"], "gb_offsets": gb_arr["offsets"],
        "gb_F0": np.array(f0), "gb_lr": np.array(float(gb.learning_rate)),
    }
    np.savez_compressed(OUT_PATH, **bundle)
    size_kb = __import__("os").path.getsize(OUT_PATH) / 1024
    print(f"wrote {OUT_PATH} ({size_kb:.1f} KB) — RF {len(rf.estimators_)} trees, GB {len(gb_trees)} trees")

    # --- Parity: NumPy inference must match scikit-learn on all 569 samples ---
    for name, np_fn, sk_prob in [
        ("random forest", rf_predict_proba(X_all, bundle), rf.predict_proba(X_all)[:, 1]),
        ("gradient boosting", gb_predict_proba(X_all, bundle), gb.predict_proba(X_all)[:, 1]),
    ]:
        max_diff = float(np.max(np.abs(np_fn - sk_prob)))
        print(f"  {name:18s} max |numpy - sklearn| = {max_diff:.2e}")
        if max_diff > 1e-6:
            raise SystemExit(f"{name} parity failed — aborting")
    print("parity OK")


if __name__ == "__main__":
    main()
