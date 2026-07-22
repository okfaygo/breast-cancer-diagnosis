"""
Export everything the web app needs into a single small .npz bundle.

The deployed app is a tiny MLP (30 -> 64 -> 32 -> 1), so it does not need
TensorFlow at inference time — just a few matrix multiplies in NumPy. Baking the
scaler parameters, feature ranges and example rows into the same file also removes
the scikit-learn dependency from the app.

Run after retraining the model:  python export_bundle.py
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow import keras

MODEL_PATH = "breast_cancer_model_wdbc.keras"
SCALER_PATH = "scaler_wdbc.pkl"
OUT_PATH = "model_bundle.npz"

# Real dataset rows used as one-click examples in the app (see notebook Stage 7).
EXAMPLES = [
    ("Typical benign", 79),
    ("Typical malignant", 408),
    ("The hard case", 385),
]


def main() -> None:
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    data = load_breast_cancer(as_frame=True)
    df = data.frame
    features = list(data.feature_names)

    # Dense layers only — dropout is a no-op at inference time.
    dense = [layer for layer in model.layers if isinstance(layer, keras.layers.Dense)]
    if len(dense) != 3:
        raise SystemExit(f"expected 3 Dense layers, found {len(dense)}")
    (w1, b1), (w2, b2), (w3, b3) = (layer.get_weights() for layer in dense)

    example_names = [name for name, _ in EXAMPLES]
    example_rows = np.array([df.loc[i, features].to_numpy(dtype="float64") for _, i in EXAMPLES])
    example_truth = [("Malignant" if df.loc[i, "target"] == 0 else "Benign") for _, i in EXAMPLES]

    np.savez_compressed(
        OUT_PATH,
        w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        features=np.array(features),
        feat_min=df[features].min().to_numpy(dtype="float64"),
        feat_max=df[features].max().to_numpy(dtype="float64"),
        feat_median=df[features].median().to_numpy(dtype="float64"),
        example_names=np.array(example_names),
        example_rows=example_rows,
        example_truth=np.array(example_truth),
    )

    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"wrote {OUT_PATH} ({size_kb:.1f} KB)")

    # --- Verify the NumPy forward pass matches Keras exactly ---
    from inference import predict_proba, load_bundle

    bundle = load_bundle(OUT_PATH)
    x = df[features].to_numpy(dtype="float64")
    keras_out = model.predict(scaler.transform(x), verbose=0).flatten()
    numpy_out = predict_proba(x, bundle)
    max_diff = float(np.max(np.abs(keras_out - numpy_out)))
    print(f"max |keras - numpy| over all {len(x)} rows: {max_diff:.3e}")
    if max_diff > 1e-5:
        raise SystemExit("NumPy inference does not match Keras — aborting")
    print("parity OK")


if __name__ == "__main__":
    main()
