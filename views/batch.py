"""Batch scoring page — upload a CSV of tumors and score them all at once."""

import numpy as np
import pandas as pd
import streamlit as st

from appdata import DEFAULT_THRESHOLD, get_bundle
from inference import predict_proba

bundle, features, index = get_bundle()

st.subheader("Batch scoring")
st.markdown(
    "Upload a CSV with the 30 WDBC feature columns to score many tumors at once. "
    "Extra columns (like an id) are kept and passed through to the results."
)

# A template so the expected format is unambiguous — the three example tumors.
template = pd.DataFrame(bundle["example_rows"], columns=features)
template.insert(0, "id", [str(n) for n in bundle["example_names"]])
st.download_button(
    "Download a template CSV",
    template.to_csv(index=False).encode("utf-8"),
    file_name="wdbc_template.csv",
    mime="text/csv",
    help="Three example tumors with all 30 columns filled in.",
)

threshold = st.slider(
    "Decision threshold",
    min_value=0.05,
    max_value=0.95,
    value=DEFAULT_THRESHOLD,
    step=0.05,
    help="Probabilities at or above this are labelled malignant.",
)

uploaded = st.file_uploader("Upload a CSV", type="csv")

if uploaded is None:
    st.caption("No file yet — download the template above to see the expected columns.")
    st.stop()

try:
    data = pd.read_csv(uploaded)
except Exception as exc:  # noqa: BLE001 - surface any parse error to the user
    st.error(f"Couldn't read that file as CSV: {exc}")
    st.stop()

missing = [f for f in features if f not in data.columns]
if missing:
    st.error(
        f"The CSV is missing {len(missing)} required feature column(s). "
        f"For example: {', '.join(missing[:3])}. Use the template to see all 30."
    )
    st.stop()

feature_frame = data[features].apply(pd.to_numeric, errors="coerce")
bad_rows = int(feature_frame.isna().any(axis=1).sum())
if bad_rows:
    st.warning(f"{bad_rows} row(s) had missing or non-numeric values and were skipped.")
scored = data.loc[feature_frame.notna().all(axis=1)].copy()
feature_frame = feature_frame.dropna()

if scored.empty:
    st.error("No fully-numeric rows to score.")
    st.stop()

probs = predict_proba(feature_frame.to_numpy(dtype="float64"), bundle)
scored["malignancy_probability"] = np.round(probs, 4)
scored["prediction"] = np.where(probs >= threshold, "Malignant", "Benign")

n_malignant = int((scored["prediction"] == "Malignant").sum())
st.success(f"Scored {len(scored)} tumor(s).")
summary = st.columns(3)
summary[0].metric("Rows scored", len(scored))
summary[1].metric("Predicted malignant", n_malignant)
summary[2].metric("Predicted benign", len(scored) - n_malignant)

st.dataframe(scored, use_container_width=True)
st.download_button(
    "Download results CSV",
    scored.to_csv(index=False).encode("utf-8"),
    file_name="wdbc_scored.csv",
    mime="text/csv",
)
