"""Prediction and feature-engineering functions."""

import pandas as pd

from config import REQUIRED_COLUMNS


def create_engineered_features(df):
    """Add simple medical feature-engineering columns."""
    result = df.copy()
    result["PulsePressure"] = result["SystolicBP"] - result["DiastolicBP"]
    result["Age_BS_Interaction"] = result["Age"] * result["BS"]
    result["AnemiaFlag"] = (result["Hemoglobin"] < 10.0).astype(int)
    return result


def build_single_input(values, feature_cols):
    """Convert Streamlit input values into one model-ready DataFrame."""
    input_df = pd.DataFrame([values], columns=REQUIRED_COLUMNS)
    input_df = create_engineered_features(input_df)
    return input_df[feature_cols]


def predict_patient(model, scaler, encoder, feature_cols, values):
    """Predict risk level for one patient and return UI-ready results."""
    model_input = build_single_input(values, feature_cols)
    scaled_input = scaler.transform(model_input)
    encoded_prediction = int(model.predict(scaled_input)[0])
    probabilities = model.predict_proba(scaled_input)[0]
    risk_label = str(encoder.classes_[encoded_prediction]).lower()
    confidence = round(float(probabilities.max()) * 100, 2)
    probability_table = pd.DataFrame({
        "Risk Level": encoder.classes_,
        "Probability (%)": [round(float(p) * 100, 2) for p in probabilities],
    })
    engineered = model_input[["PulsePressure", "Age_BS_Interaction", "AnemiaFlag"]].iloc[0]
    return {"risk_label": risk_label, "confidence": confidence, "probability_table": probability_table, "engineered": engineered, "model_input": model_input}


def validate_batch_columns(df):
    """Return required columns missing from an uploaded CSV."""
    return [column for column in REQUIRED_COLUMNS if column not in df.columns]


def clean_batch_data(df):
    """Fill missing CSV values using median values."""
    clean_df = df.copy()
    for column in REQUIRED_COLUMNS:
        clean_df[column] = pd.to_numeric(clean_df[column], errors="coerce")
        if clean_df[column].isnull().any():
            clean_df[column] = clean_df[column].fillna(clean_df[column].median())
    return clean_df


def predict_batch(model, scaler, encoder, feature_cols, df):
    """Predict risk level and confidence for many patient records."""
    clean_df = clean_batch_data(df)
    engineered_df = create_engineered_features(clean_df)
    scaled_input = scaler.transform(engineered_df[feature_cols])
    encoded_predictions = model.predict(scaled_input)
    probabilities = model.predict_proba(scaled_input)
    output_df = clean_df.copy()
    output_df["PulsePressure"] = engineered_df["PulsePressure"]
    output_df["Age_BS_Interaction"] = engineered_df["Age_BS_Interaction"]
    output_df["AnemiaFlag"] = engineered_df["AnemiaFlag"]
    output_df["Predicted Risk"] = [encoder.classes_[int(pred)] for pred in encoded_predictions]
    output_df["Confidence (%)"] = [round(float(row.max()) * 100, 2) for row in probabilities]
    return output_df
