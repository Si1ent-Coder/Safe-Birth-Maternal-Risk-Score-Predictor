"""Configuration values used by the SafeBirth Streamlit app."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
VISUALS_DIR = PROJECT_ROOT / "visuals"

APP_TITLE = "SafeBirth - Maternal Risk Score Predictor"
APP_SUBTITLE = "ML-powered maternal health screening dashboard"

REQUIRED_COLUMNS = [
    "Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate", "Hemoglobin",
]

FIELD_INFO = {
    "Age": {"label": "Age", "unit": "years", "help": "Patient age in completed years.", "min": 18, "max": 55, "default": 28, "step": 1},
    "SystolicBP": {"label": "Systolic Blood Pressure", "unit": "mmHg", "help": "Upper blood pressure value. Very high values may indicate hypertension risk.", "min": 70, "max": 200, "default": 110, "step": 1},
    "DiastolicBP": {"label": "Diastolic Blood Pressure", "unit": "mmHg", "help": "Lower blood pressure value. Used with systolic BP to calculate pulse pressure.", "min": 50, "max": 120, "default": 75, "step": 1},
    "BS": {"label": "Blood Sugar", "unit": "mmol/L", "help": "Blood glucose reading. Higher values can increase maternal risk.", "min": 5.0, "max": 20.0, "default": 7.0, "step": 0.1},
    "BodyTemp": {"label": "Body Temperature", "unit": "F", "help": "Body temperature in Fahrenheit. Fever may indicate infection or stress.", "min": 97.0, "max": 104.0, "default": 98.6, "step": 0.1},
    "HeartRate": {"label": "Heart Rate", "unit": "bpm", "help": "Resting heart rate. Higher readings can be a warning sign.", "min": 50, "max": 120, "default": 72, "step": 1},
    "Hemoglobin": {"label": "Hemoglobin", "unit": "g/dL", "help": "Hemoglobin level. Values below 10 g/dL are treated as anemia risk in this project.", "min": 5.0, "max": 17.0, "default": 11.5, "step": 0.1},
}

RISK_STYLES = {
    "low risk": {"title": "Low Risk", "class": "risk-low", "color": "#059669", "advice": "Continue routine antenatal checkups, balanced nutrition, hydration, and regular monitoring."},
    "mid risk": {"title": "Medium Risk", "class": "risk-mid", "color": "#d97706", "advice": "Consult an obstetrician soon and monitor BP, sugar, temperature, and hemoglobin more frequently."},
    "high risk": {"title": "High Risk", "class": "risk-high", "color": "#dc2626", "advice": "Immediate medical supervision is strongly recommended. Do not delay clinical evaluation."},
}
