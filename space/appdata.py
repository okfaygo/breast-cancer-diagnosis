"""Shared, cached data loaders for the app's pages."""

import numpy as np
import streamlit as st

from inference import load_bundle
from tree_inference import load_trees

# Threshold tuned in Stage 8 of the notebook: favours recall (catching cancer)
# over precision, which is the right trade-off for a screening context.
DEFAULT_THRESHOLD = 0.35


@st.cache_resource(show_spinner="Loading model…")
def get_bundle():
    """The model weights, scaler, feature ranges and example rows."""
    bundle = load_bundle()
    features = [str(f) for f in bundle["features"]]
    return bundle, features, {f: i for i, f in enumerate(features)}


@st.cache_resource(show_spinner="Loading analysis…")
def get_about_data(path: str = "about_data.npz"):
    """Precomputed performance data for the About page (see compute_about_data.py)."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@st.cache_resource(show_spinner="Loading comparison models…")
def get_trees():
    """Random forest + gradient boosting trees for the model-agreement panel."""
    return load_trees()
