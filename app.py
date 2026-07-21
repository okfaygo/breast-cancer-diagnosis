"""
Breast Cancer Diagnosis — Streamlit demo app.

Serves the WDBC neural network (breast_cancer_model_wdbc.keras + scaler_wdbc.pkl)
trained in breast_cancer_pipeline_wdbc.ipynb.

Run locally with:  python -m streamlit run app.py
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # quieten TensorFlow startup logs

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer

MODEL_PATH = "breast_cancer_model_wdbc.keras"
SCALER_PATH = "scaler_wdbc.pkl"

# Threshold tuned in Stage 8 of the notebook: favours recall (catching cancer)
# over precision, which is the right trade-off for a screening context.
DEFAULT_THRESHOLD = 0.35

# The features that dominated permutation importance. Everything else is still
# used by the model, but defaults to the dataset median so the form stays usable.
KEY_FEATURES = [
    "worst radius",
    "worst area",
    "worst concavity",
    "worst concave points",
    "worst texture",
    "mean concave points",
]

# Real dataset rows used as one-click examples (see notebook Stage 7).
EXAMPLES = {
    "Typical benign": 79,
    "Typical malignant": 408,
    "The hard case (missed by the tree models)": 385,
}


@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    """Load the trained Keras model and the scaler fitted on its training data."""
    from tensorflow import keras

    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@st.cache_data
def load_dataset():
    """WDBC data, used for slider ranges, medians and the example cases."""
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    # sklearn encodes 0 = malignant, 1 = benign — flip to this project's M = 1 convention
    df["diagnosis"] = df["target"].map({0: "M", 1: "B"})
    return df, list(data.feature_names)


def skey(feature: str) -> str:
    """Session-state key for a feature's slider."""
    return f"f::{feature}"


def apply_example(row_index: int, features: list[str], df: pd.DataFrame) -> None:
    """Fill every slider from a real dataset row (runs before widgets re-render)."""
    row = df.loc[row_index]
    for feature in features:
        st.session_state[skey(feature)] = float(row[feature])


def reset_to_median(features: list[str], df: pd.DataFrame) -> None:
    for feature in features:
        st.session_state[skey(feature)] = float(df[feature].median())


def feature_slider(feature: str, df: pd.DataFrame) -> float:
    """A slider bounded by the feature's real range in the dataset."""
    low, high = float(df[feature].min()), float(df[feature].max())
    if skey(feature) not in st.session_state:
        st.session_state[skey(feature)] = float(df[feature].median())
    return st.slider(
        feature,
        min_value=low,
        max_value=high,
        step=(high - low) / 200,
        key=skey(feature),
    )


def main() -> None:
    st.set_page_config(page_title="Breast Cancer Diagnosis", page_icon="🔬", layout="wide")

    df, features = load_dataset()

    st.title("Breast cancer diagnosis")
    st.caption(
        "Neural network trained on the Wisconsin Diagnostic Breast Cancer dataset "
        "(569 samples, 30 features)."
    )

    st.warning(
        "**Educational demo only — not a medical device and not for clinical use.** "
        "This model was trained on a small public research dataset to practise machine "
        "learning. It must never be used to make real medical decisions.",
        icon="⚠️",
    )

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.subheader("Tumor measurements")

        st.markdown("**Load an example case**")
        example_cols = st.columns(len(EXAMPLES))
        for col, (label, row_index) in zip(example_cols, EXAMPLES.items()):
            col.button(
                label,
                use_container_width=True,
                on_click=apply_example,
                args=(row_index, features, df),
            )

        st.button(
            "Reset all to median",
            on_click=reset_to_median,
            args=(features, df),
        )

        st.divider()

        for feature in KEY_FEATURES:
            feature_slider(feature, df)

        other_features = [f for f in features if f not in KEY_FEATURES]
        with st.expander(f"Advanced — {len(other_features)} more features"):
            st.caption("These default to the dataset median. The model uses all 30.")
            for feature in other_features:
                feature_slider(feature, df)

    with right:
        st.subheader("Prediction")

        threshold = st.slider(
            "Decision threshold",
            min_value=0.05,
            max_value=0.95,
            value=DEFAULT_THRESHOLD,
            step=0.05,
            help=(
                "Probabilities at or above this are called malignant. Lower values catch "
                "more cancers (higher recall) at the cost of more false alarms."
            ),
        )

        model, scaler = load_artifacts()
        sample = np.array([[st.session_state[skey(f)] for f in features]])
        probability = float(model.predict(scaler.transform(sample), verbose=0)[0][0])
        is_malignant = probability >= threshold

        st.metric("Probability of malignancy", f"{probability * 100:.1f}%")

        if is_malignant:
            st.error(f"**Malignant** — at or above the {threshold:.2f} threshold", icon="⚠️")
        else:
            st.success(f"**Benign** — below the {threshold:.2f} threshold", icon="✅")

        st.progress(probability)
        st.caption(
            f"Threshold {threshold:.2f}. The default of {DEFAULT_THRESHOLD} is tuned to "
            "favour recall, because a missed cancer is costlier than a false alarm."
        )

        st.divider()

        st.markdown("**Model performance** (5-fold cross-validation)")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Recall", "0.96", help="± 0.04 across folds")
        metric_cols[1].metric("AUC", "0.994")
        metric_cols[2].metric("Training data", "569")
        st.caption(
            "Cross-validated figures, not the flattering single-split score — see "
            "Stage 9 of the notebook."
        )


if __name__ == "__main__":
    main()
