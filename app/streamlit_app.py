"""
SafeBirth: Maternal Risk Score Predictor

Run command:
    streamlit run app/streamlit_app.py
"""

import pandas as pd
import streamlit as st

from config import APP_TITLE, REQUIRED_COLUMNS
from model_service import load_model_artefacts
from prediction_service import predict_batch, predict_patient, validate_batch_columns
from ui_helpers import (
    inject_css,
    read_slider,
    render_about_project,
    render_engineered_features,
    render_header,
    render_input_summary,
    render_result_card,
    render_sidebar,
    render_visual_gallery,
)


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="SB",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

try:
    model, scaler, encoder, feature_cols = load_model_artefacts()
except FileNotFoundError:
    st.error(
        "Model files are missing. Please run notebooks/train_pipeline.py first "
        "or place the saved .pkl files inside the models folder."
    )
    st.stop()

render_sidebar()
render_header()


# hii
st.markdown("""
<style>
button[data-baseweb="tab"] {
    color: red !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)
# hii

single_tab, batch_tab, insight_tab, about_tab = st.tabs(
    ["Single Prediction", "Batch Prediction", "Model Insights", "Project Explanation"]
)


with single_tab:
    st.markdown('<div class="section-header">Patient Health Input Form</div>', unsafe_allow_html=True)
    st.write("All field names, units, and acceptable ranges are shown clearly before each slider.")

    with st.form("risk_prediction_form"):
        left_col, right_col = st.columns(2)

        with left_col:
            age = read_slider("Age")
            systolic_bp = read_slider("SystolicBP")
            diastolic_bp = read_slider("DiastolicBP")
            blood_sugar = read_slider("BS")

        with right_col:
            body_temp = read_slider("BodyTemp")
            heart_rate = read_slider("HeartRate")
            hemoglobin = read_slider("Hemoglobin")

        # predict_clicked = st.form_submit_button("Predict Maternal Risk")
        st.markdown("""
        <style>
        div[data-testid="stForm"] button {
            background-color: white !important;
            color: black !important;
            border: 2px solid #ddd !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
        }
        </style>
        """, unsafe_allow_html=True)

        predict_clicked = st.form_submit_button("Predict Maternal Risk")

    if predict_clicked:
        values = {
            "Age": age,
            "SystolicBP": systolic_bp,
            "DiastolicBP": diastolic_bp,
            "BS": blood_sugar,
            "BodyTemp": body_temp,
            "HeartRate": heart_rate,
            "Hemoglobin": hemoglobin,
        }

        with st.spinner("Analyzing maternal health readings..."):
            result = predict_patient(model, scaler, encoder, feature_cols, values)

        render_result_card(result["risk_label"], result["confidence"])

        st.subheader("Prediction Probability")
        st.bar_chart(result["probability_table"].set_index("Risk Level"))
        st.dataframe(result["probability_table"], use_container_width=True, hide_index=True)

        with st.expander("Input Summary", expanded=True):
            render_input_summary(values)

        with st.expander("Feature Engineering Details", expanded=True):
            render_engineered_features(result["engineered"])


with batch_tab:
    st.markdown('<div class="section-header">Batch CSV Prediction</div>', unsafe_allow_html=True)
    st.write("Upload a CSV file with the exact column names below.")
    st.code(", ".join(REQUIRED_COLUMNS), language="text")

    uploaded_file = st.file_uploader("Upload maternal health CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        missing_columns = validate_batch_columns(batch_df)

        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
        else:
            result_df = predict_batch(model, scaler, encoder, feature_cols, batch_df)
            st.success(f"Prediction completed for {len(result_df)} patient records.")
            st.dataframe(result_df, use_container_width=True, hide_index=True)

            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Prediction Results",
                csv_bytes,
                "safebirth_batch_predictions.csv",
                "text/csv",
            )


with insight_tab:
    st.markdown('<div class="section-header">EDA and Model Visualizations</div>', unsafe_allow_html=True)
    st.write("These charts are generated from the training pipeline and can be used in your project report.")
    render_visual_gallery()


with about_tab:
    st.markdown('<div class="section-header">How to Explain This Project</div>', unsafe_allow_html=True)
    render_about_project()


st.markdown(
    
    """
<hr>
<center>
SafeBirth - AI Maternal Healthcare System<br>
B.Tech CSE Pre Final Year Project
</center>

""",
    unsafe_allow_html=True,
)
