"""
Breast Cancer Diagnosis — Streamlit demo app.

Serves the WDBC neural network trained in breast_cancer_pipeline_wdbc.ipynb.

Inference runs in NumPy from model_bundle.npz (see inference.py), so this app needs
neither TensorFlow nor scikit-learn — it starts in well under a second. Regenerate
the bundle with `python export_bundle.py` after retraining.

Run locally with:  python -m streamlit run app.py
"""

import numpy as np
import streamlit as st

from inference import load_bundle, predict_proba

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


@st.cache_resource(show_spinner="Loading model…")
def get_bundle():
    bundle = load_bundle()
    features = [str(f) for f in bundle["features"]]
    return bundle, features, {f: i for i, f in enumerate(features)}


def skey(feature: str) -> str:
    """Session-state key for a feature's slider."""
    return f"f::{feature}"


def apply_example(row: np.ndarray, features: list[str]) -> None:
    """Fill every slider from a real dataset row (runs before widgets re-render)."""
    for feature, value in zip(features, row):
        st.session_state[skey(feature)] = float(value)


def reset_to_median(bundle, features: list[str]) -> None:
    apply_example(bundle["feat_median"], features)


def feature_slider(feature: str, bundle, index: dict[str, int]) -> None:
    """A slider bounded by the feature's real range in the dataset."""
    i = index[feature]
    low, high = float(bundle["feat_min"][i]), float(bundle["feat_max"][i])
    if skey(feature) not in st.session_state:
        st.session_state[skey(feature)] = float(bundle["feat_median"][i])
    st.slider(
        feature,
        min_value=low,
        max_value=high,
        step=(high - low) / 200,
        key=skey(feature),
    )


def main() -> None:
    st.set_page_config(page_title="Breast Cancer Diagnosis", page_icon="🔬", layout="wide")

    bundle, features, index = get_bundle()

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
        example_names = [str(n) for n in bundle["example_names"]]
        example_truth = [str(t) for t in bundle["example_truth"]]
        example_cols = st.columns(len(example_names))
        for col, name, row, truth in zip(
            example_cols, example_names, bundle["example_rows"], example_truth
        ):
            col.button(
                name,
                use_container_width=True,
                help=f"Actual diagnosis: {truth}",
                on_click=apply_example,
                args=(row, features),
            )

        st.button("Reset all to median", on_click=reset_to_median, args=(bundle, features))

        st.divider()

        for feature in KEY_FEATURES:
            feature_slider(feature, bundle, index)

        other_features = [f for f in features if f not in KEY_FEATURES]
        with st.expander(f"Advanced — {len(other_features)} more features"):
            st.caption("These default to the dataset median. The model uses all 30.")
            for feature in other_features:
                feature_slider(feature, bundle, index)

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

        sample = np.array([[st.session_state[skey(f)] for f in features]])
        probability = float(predict_proba(sample, bundle)[0])

        st.metric("Probability of malignancy", f"{probability * 100:.1f}%")

        if probability >= threshold:
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
