"""Prediction page — enter tumor measurements and get a malignancy probability."""

import numpy as np
import streamlit as st

from appdata import DEFAULT_THRESHOLD, get_bundle, get_trees
from inference import predict_proba
from tree_inference import gb_predict_proba, rf_predict_proba

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


bundle, features, index = get_bundle()

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
    metric_cols[0].metric("Recall", "0.96", help="± 0.03 across folds")
    metric_cols[1].metric("AUC", "0.994")
    metric_cols[2].metric("Training data", "569")
    st.caption("See the **About the model** page for the full breakdown.")

st.divider()
st.markdown("#### Model agreement")
st.caption(
    "The neural net is the deployed model. A random forest and gradient boosting were "
    "trained on the same data for comparison. When they disagree, the case is genuinely "
    "borderline — try **The hard case** example to see it."
)

trees = get_trees()
rf_prob = float(rf_predict_proba(sample, trees)[0])
gb_prob = float(gb_predict_proba(sample, trees)[0])
agreement = [
    {"model": "Neural net (served)", "prob": round(probability, 3)},
    {"model": "Random forest", "prob": round(rf_prob, 3)},
    {"model": "Gradient boosting", "prob": round(gb_prob, 3)},
]

st.vega_lite_chart(
    {
        "height": 150,
        "layer": [
            {
                "data": {"values": agreement},
                "mark": {"type": "bar", "color": "#7f77dd"},
                "encoding": {
                    "y": {"field": "model", "type": "nominal", "sort": None, "title": None},
                    "x": {"field": "prob", "type": "quantitative",
                          "scale": {"domain": [0, 1]}, "title": "P(malignant)"},
                },
            },
            {
                "data": {"values": [{"threshold": threshold}]},
                "mark": {"type": "rule", "color": "#e24b4a", "strokeDash": [5, 4], "size": 2},
                "encoding": {"x": {"field": "threshold", "type": "quantitative"}},
            },
        ],
    },
    use_container_width=True,
)

verdicts = {name: (p >= threshold) for name, p in
            [("neural net", probability), ("random forest", rf_prob), ("gradient boosting", gb_prob)]}
if len(set(verdicts.values())) > 1:
    malignant = [n for n, v in verdicts.items() if v]
    benign = [n for n, v in verdicts.items() if not v]
    st.info(
        f"The models **disagree** at the {threshold:.2f} threshold — "
        f"{', '.join(malignant)} say malignant while {', '.join(benign)} say benign. "
        "That is the signature of a borderline tumor.",
        icon="⚖️",
    )
else:
    st.caption(f"All three models agree at the {threshold:.2f} threshold (dashed red line).")
