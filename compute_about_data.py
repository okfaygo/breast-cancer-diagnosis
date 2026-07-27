"""
Compute the performance data shown on the app's "About the model" page.

This mirrors the analysis in breast_cancer_pipeline_wdbc.ipynb (Stages 5-9) and
saves the results into about_data.npz, so the app can render charts with only
streamlit + numpy — no TensorFlow, scikit-learn or matplotlib at runtime.

The single-split figures (ROC, confusion matrices, permutation importance) use the
exact deployed model via the NumPy forward pass in inference.py, so they match what
the app serves. The cross-validation figures retrain per fold, reproducing Stage 9.

Run after retraining:  python compute_about_data.py  (takes ~1-2 minutes)
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from inference import load_bundle, predict_proba

SEED = 42
OUT_PATH = "about_data.npz"
DEFAULT_THRESHOLD = 0.35


def load_xy():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df["diagnosis"] = df["target"].map({0: "M", 1: "B"})  # flip to Malignant = 1
    features = list(data.feature_names)
    X = df[features].to_numpy(dtype="float64")
    y = (df["diagnosis"] == "M").astype(int).to_numpy()
    return X, y, features


def build_model(input_dim):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model


def single_split_analysis(X, y, features, bundle):
    """ROC, confusion matrices and permutation importance on the held-out test set."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )

    # Deployed-model probabilities via the NumPy forward pass — matches the live app
    prob = predict_proba(X_test, bundle)

    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)

    cms = {
        "cm_050": confusion_matrix(y_test, (prob >= 0.50).astype(int)),
        "cm_035": confusion_matrix(y_test, (prob >= DEFAULT_THRESHOLD).astype(int)),
    }

    # Permutation importance: accuracy drop when each feature column is shuffled.
    rng = np.random.default_rng(SEED)
    base_acc = accuracy_score(y_test, (prob >= 0.50).astype(int))
    importances = np.zeros(len(features))
    for j in range(len(features)):
        drops = []
        for _ in range(10):
            X_perm = X_test.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            acc = accuracy_score(y_test, (predict_proba(X_perm, bundle) >= 0.50).astype(int))
            drops.append(base_acc - acc)
        importances[j] = float(np.mean(drops))

    return {
        "roc_fpr": fpr, "roc_tpr": tpr, "roc_auc": np.array(auc),
        "test_pos": np.array(int(y_test.sum())), "test_neg": np.array(int((y_test == 0).sum())),
        "importance_features": np.array(features),
        "importance_values": importances,
        **cms,
    }


def cross_validation(X, y):
    """5-fold stratified CV for the three model families — reproduces Stage 9."""
    import tensorflow as tf
    from tensorflow import keras

    os.environ["PYTHONHASHSEED"] = str(SEED)
    keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    model_names = ["Neural net", "Random forest", "Gradient boosting"]
    metrics = ["accuracy", "recall", "precision", "auc"]
    # rows: fold, cols: model
    per_fold = {m: np.zeros((5, 3)) for m in metrics}

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        X_tr_full, X_te, y_tr_full, y_te = X[tr], X[te], y[tr], y[te]
        X_tr, X_vl, y_tr, y_vl = train_test_split(
            X_tr_full, y_tr_full, test_size=0.15, random_state=SEED, stratify=y_tr_full
        )
        sc = StandardScaler().fit(X_tr)
        cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)

        nn = build_model(X.shape[1])
        nn.fit(sc.transform(X_tr), y_tr, validation_data=(sc.transform(X_vl), y_vl),
               epochs=150, batch_size=32, class_weight={0: cw[0], 1: cw[1]},
               callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                                        restore_best_weights=True)],
               verbose=0)
        rf = RandomForestClassifier(n_estimators=300, random_state=SEED,
                                    class_weight="balanced").fit(X_tr_full, y_tr_full)
        gb = GradientBoostingClassifier(random_state=SEED).fit(X_tr_full, y_tr_full)

        probs = [
            nn.predict(sc.transform(X_te), verbose=0).flatten(),
            rf.predict_proba(X_te)[:, 1],
            gb.predict_proba(X_te)[:, 1],
        ]
        for col, p in enumerate(probs):
            pred = (p >= 0.5).astype(int)
            per_fold["accuracy"][fold, col] = accuracy_score(y_te, pred)
            per_fold["recall"][fold, col] = recall_score(y_te, pred)
            per_fold["precision"][fold, col] = precision_score(y_te, pred)
            per_fold["auc"][fold, col] = roc_auc_score(y_te, p)

    out = {"cv_model_names": np.array(model_names)}
    for m in metrics:
        out[f"cv_{m}"] = per_fold[m]
    return out


def main():
    X, y, features = load_xy()
    bundle = load_bundle()

    print("single-split analysis (ROC, confusion, importance)…")
    single = single_split_analysis(X, y, features, bundle)

    print("cross-validation (retraining 5 folds)…")
    cv = cross_validation(X, y)

    np.savez_compressed(OUT_PATH, **single, **cv)
    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"wrote {OUT_PATH} ({size_kb:.1f} KB)")

    nn_recall = cv["cv_recall"][:, 0]
    print(f"sanity — NN CV recall per fold: {np.round(nn_recall, 3).tolist()}")
    print(f"         NN CV recall mean±std: {nn_recall.mean():.3f} ± {nn_recall.std():.3f}")
    print(f"         test-set AUC: {float(single['roc_auc']):.4f}")


if __name__ == "__main__":
    main()
