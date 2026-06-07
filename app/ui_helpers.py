"""Reusable Streamlit UI blocks for the SafeBirth app."""

import pandas as pd
import streamlit as st

from config import FIELD_INFO, REQUIRED_COLUMNS, RISK_STYLES, VISUALS_DIR


def inject_css():
    """Apply custom CSS for a readable, modern Streamlit interface."""
    st.markdown("""
<style>
.stApp { background: #f5f7fb; color: #0f172a; font-family: 'Segoe UI', sans-serif; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a, #1e293b); }
section[data-testid="stSidebar"] * { color: #ffffff !important; }
.main-title { font-size: 42px; line-height: 1.12; font-weight: 900; color: #111827; margin-bottom: 6px; }
.subtitle { color: #475569; font-size: 18px; margin-bottom: 22px; }
.top-band { background: linear-gradient(135deg, #ffffff, #eef2ff); border: 1px solid #dbe4ff; border-radius: 14px; padding: 24px; margin-bottom: 18px; box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08); }
.metric-strip { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 12px 0 20px; }
.metric-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px 16px; }
.metric-card b { display: block; font-size: 22px; color: #111827; }
.metric-card span { color: #64748b; font-size: 13px; font-weight: 700; }
.section-header { color: #ffffff; background: linear-gradient(90deg, #7c3aed, #2563eb); border-radius: 12px; padding: 13px 18px; font-weight: 800; font-size: 20px; margin: 8px 0 16px; }
.field-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 13px 15px; margin: 11px 0 6px; box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05); }
.field-name { font-size: 16px; color: #0f172a; font-weight: 900; }
.field-help { font-size: 13px; color: #64748b; margin-top: 3px; }
.field-unit { display: inline-block; margin-top: 7px; padding: 3px 8px; border-radius: 999px; background: #eef2ff; color: #3730a3; font-size: 12px; font-weight: 800; }
div[data-testid="stSlider"] { background: #ffffff; border-radius: 12px; padding: 8px 12px 3px; border: 1px solid #edf2f7; }
div[data-testid="stSlider"] label, div[data-testid="stFileUploader"] label { color: #0f172a !important; font-weight: 800 !important; font-size: 15px !important; }
div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button { background: linear-gradient(135deg, #ec4899, #7c3aed); color: white; border: 0; border-radius: 12px; padding: 14px; font-size: 17px; font-weight: 900; width: 100%; box-shadow: 0 8px 20px rgba(124, 58, 237, 0.28); }
.result-card { border-radius: 18px; padding: 28px; color: white; margin-top: 18px; box-shadow: 0 14px 32px rgba(15, 23, 42, 0.18); }
.risk-low { background: linear-gradient(135deg, #10b981, #047857); }
.risk-mid { background: linear-gradient(135deg, #f59e0b, #b45309); }
.risk-high { background: linear-gradient(135deg, #ef4444, #b91c1c); }
.result-title { font-size: 19px; font-weight: 800; opacity: 0.95; }
.result-risk { font-size: 46px; line-height: 1; font-weight: 950; margin: 12px 0; }
.result-advice { background: rgba(255, 255, 255, 0.16); border-radius: 12px; padding: 14px; line-height: 1.7; margin-top: 14px; }
.info-panel { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; margin-bottom: 14px; }
.disclaimer { background: #fff7ed; color: #9a3412; border: 1px solid #fed7aa; border-radius: 12px; padding: 13px 15px; font-weight: 700; }
@media (max-width: 800px) { .main-title { font-size: 31px; } .metric-strip { grid-template-columns: 1fr; } .result-risk { font-size: 34px; } }
</style>
""", unsafe_allow_html=True)


def render_sidebar():
    """Show project summary, workflow, and disclaimer in the sidebar."""
    with st.sidebar:
        st.title("SafeBirth")
        st.caption("Maternal Risk Score Predictor")
        st.markdown("---")
        st.markdown("""
**Inputs used**
- Age
- Systolic BP
- Diastolic BP
- Blood Sugar
- Body Temperature
- Heart Rate
- Hemoglobin

**Model pipeline**
1. Data cleaning
2. Feature engineering
3. Standard scaling
4. SMOTE balancing
5. Model prediction
""")
        st.markdown("---")
        st.warning("Medical diagnosis system.")


def render_header():
    """Render the dashboard header and high-level project facts."""
    st.markdown("""
<div class="top-band">
    <div class="main-title">SafeBirth - Maternal Risk Score Predictor</div>
    <div class="subtitle">A final year ML project for early maternal health risk screening.</div>
    <div class="metric-strip">
        <div class="metric-card"><b>7</b><span>Basic Health Inputs</span></div>
        <div class="metric-card"><b>3</b><span>Risk Classes</span></div>
        <div class="metric-card"><b>SMOTE</b><span>Imbalance Handling</span></div>
    </div>
</div>
""", unsafe_allow_html=True)


def render_field_intro(field_key):
    """Show a visible field name, unit, and simple explanation before a slider."""
    info = FIELD_INFO[field_key]
    st.markdown(f"""
<div class="field-card">
    <div class="field-name">{info['label']}</div>
    <div class="field-help">{info['help']}</div>
    <span class="field-unit">Unit: {info['unit']} | Range: {info['min']} to {info['max']}</span>
</div>
""", unsafe_allow_html=True)


def read_slider(field_key):
    """Create a slider from FIELD_INFO and return the selected value."""
    info = FIELD_INFO[field_key]
    render_field_intro(field_key)
    return st.slider(
        label=f"{info['label']} ({info['unit']})",
        min_value=info["min"], max_value=info["max"], value=info["default"], step=info["step"],
    )


def render_result_card(risk_label, confidence):
    """Render the final prediction using healthcare-style color coding."""
    style = RISK_STYLES.get(risk_label, RISK_STYLES["mid risk"])
    st.markdown(f"""
<div class="result-card {style['class']}">
    <div class="result-title">Prediction Result</div>
    <div class="result-risk">{style['title']}</div>
    <div><b>Model Confidence:</b> {confidence}%</div>
    <div class="result-advice"><b>Recommendation:</b> {style['advice']}</div>
</div>
""", unsafe_allow_html=True)


def render_input_summary(values):
    """Show all entered values in a clean table for transparency."""
    rows = []
    for key in REQUIRED_COLUMNS:
        info = FIELD_INFO[key]
        rows.append({"Field Name": info["label"], "Model Column": key, "Value": values[key], "Unit": info["unit"]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_engineered_features(engineered):
    """Explain the derived model features used after form submission."""
    display_df = pd.DataFrame([
        {"Engineered Feature": "PulsePressure", "Formula": "SystolicBP - DiastolicBP", "Value": round(float(engineered["PulsePressure"]), 2), "Why it matters": "Shows pressure difference and cardiovascular strain."},
        {"Engineered Feature": "Age_BS_Interaction", "Formula": "Age x BS", "Value": round(float(engineered["Age_BS_Interaction"]), 2), "Why it matters": "Captures combined age and blood sugar risk."},
        {"Engineered Feature": "AnemiaFlag", "Formula": "1 if Hemoglobin < 10 else 0", "Value": int(engineered["AnemiaFlag"]), "Why it matters": "Flags possible anemia-related pregnancy risk."},
    ])
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_visual_gallery():
    """Show report-ready visualizations generated by the training pipeline."""
    visual_files = [
        ("Risk Distribution", "01_risk_distribution.png"),
        ("Feature Histograms", "02_histograms.png"),
        ("Feature Boxplots", "03_boxplots.png"),
        ("Correlation Heatmap", "04_correlation_heatmap.png"),
        ("Pair Plot", "05_pairplot.png"),
        ("Confusion Matrices", "06_confusion_matrices.png"),
        ("Feature Importance", "07_feature_importance.png"),
        ("Model Comparison", "08_model_comparison.png"),
    ]
    for title, filename in visual_files:
        path = VISUALS_DIR / filename
        if path.exists():
            st.subheader(title)
            st.image(str(path), use_container_width=True)


def render_about_project():
    """Explain project flow in a report-friendly way."""
    st.markdown("""
<div class="info-panel"><b>Project Methodology</b><br>
SafeBirth loads maternal health records, performs EDA, handles missing values,
creates medical features, balances classes using SMOTE, trains Logistic Regression,
Random Forest, and XGBoost, then deploys the best model using Streamlit.</div>
<div class="info-panel"><b>Why High Risk Recall matters</b><br>
In healthcare screening, missing a high-risk pregnancy is more dangerous than giving
a cautious warning. Therefore, the model is selected with special focus on High Risk Recall.</div>
<div class="disclaimer">This tool is for academic demonstration and screening support only. It must not replace a doctor.</div>
""", unsafe_allow_html=True)
