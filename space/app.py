"""
Breast Cancer Diagnosis — Streamlit demo app (multi-page).

Serves the WDBC neural network trained in breast_cancer_pipeline_wdbc.ipynb.

Inference runs in NumPy from model_bundle.npz (see inference.py), so this app needs
neither TensorFlow nor scikit-learn — it starts in well under a second. The About
page reads precomputed metrics from about_data.npz (see compute_about_data.py).

Regenerate those artifacts after retraining:
    python export_bundle.py
    python compute_about_data.py

Run locally with:  python -m streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="Breast Cancer Diagnosis", page_icon="🔬", layout="wide")

# Header and disclaimer render on every page — the entry script runs before each one.
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

pages = [
    st.Page("views/predict.py", title="Predict", icon="🩺", default=True),
    st.Page("views/about.py", title="About the model", icon="📊"),
]
st.navigation(pages).run()
