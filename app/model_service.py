"""Model loading utilities for SafeBirth."""

import joblib
import streamlit as st

from config import MODEL_DIR


@st.cache_resource(show_spinner=False)
def load_model_artefacts():
    """Load trained artefacts once and cache them for the Streamlit session.

    The app requires best_model.pkl, scaler.pkl, label_encoder.pkl, and
    feature_cols.pkl. Caching avoids loading these files again on every click.
    """
    model = joblib.load(MODEL_DIR / "best_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")
    return model, scaler, encoder, feature_cols
