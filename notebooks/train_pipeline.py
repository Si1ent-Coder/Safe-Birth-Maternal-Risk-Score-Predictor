"""
SafeBirth: Maternal Risk Score Predictor
Full ML Training Pipeline - Steps 2 to 10

This script is intentionally written in a beginner-to-intermediate style.
Each function maps to a project-report step and can be explained easily in viva.
"""

import warnings
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "maternal_health.csv"
MODEL_DIR = PROJECT_ROOT / "models"
VISUALS_DIR = PROJECT_ROOT / "visuals"
REPORTS_DIR = PROJECT_ROOT / "reports"

NUMERIC_COLUMNS = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate", "Hemoglobin"]
TARGET_COLUMN = "RiskLevel"
RISK_ORDER = ["low risk", "mid risk", "high risk"]
PALETTE = {"low risk": "#16a34a", "mid risk": "#f59e0b", "high risk": "#dc2626"}

FEATURE_COLUMNS = [
    "Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate", "Hemoglobin",
    "PulsePressure", "Age_BS_Interaction", "AnemiaFlag",
]


def setup_directories():
    """Create output folders used by the training pipeline."""
    MODEL_DIR.mkdir(exist_ok=True)
    VISUALS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)


def load_dataset():
    """Step 2: Load dataset and print basic information."""
    print("\nSTEP 2 - DATA LOADING")
    df = pd.read_csv(DATA_PATH)
    print(f"Shape: {df.shape}")
    print("\nData types:")
    print(df.dtypes)
    print("\nNull values:")
    print(df.isnull().sum())
    print("\nClass distribution:")
    print(df[TARGET_COLUMN].value_counts())
    print("\nFirst 5 rows:")
    print(df.head())
    return df


def run_eda(df):
    """Step 3: Generate beginner-friendly EDA visuals."""
    print("\nSTEP 3 - EXPLORATORY DATA ANALYSIS")

    counts = df[TARGET_COLUMN].value_counts().reindex(RISK_ORDER)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(RISK_ORDER, counts.values, color=[PALETTE[x] for x in RISK_ORDER])
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 3, int(value), ha="center", fontweight="bold")
    ax.set_title("Risk Level Distribution", fontweight="bold")
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Number of Records")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "01_risk_distribution.png", dpi=180)
    plt.close()

    df[NUMERIC_COLUMNS].hist(figsize=(16, 9), bins=25, color="#2563eb", edgecolor="white")
    plt.suptitle("Feature Distributions", fontweight="bold")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "02_histograms.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for index, column in enumerate(NUMERIC_COLUMNS):
        sns.boxplot(data=df, x=TARGET_COLUMN, y=column, order=RISK_ORDER, ax=axes[index], palette=PALETTE)
        axes[index].set_title(f"{column} by Risk Level", fontweight="bold")
        axes[index].set_xlabel("")
    axes[-1].axis("off")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "03_boxplots.png", dpi=180)
    plt.close()

    encoded_df = df.copy()
    temp_encoder = LabelEncoder()
    encoded_df["RiskLevel_enc"] = temp_encoder.fit_transform(encoded_df[TARGET_COLUMN])
    corr = encoded_df[NUMERIC_COLUMNS + ["RiskLevel_enc"]].corr()
    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap", fontweight="bold")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "04_correlation_heatmap.png", dpi=180)
    plt.close()

    pairplot_columns = ["Age", "SystolicBP", "BS", "Hemoglobin", TARGET_COLUMN]
    pair_grid = sns.pairplot(df[pairplot_columns].dropna(), hue=TARGET_COLUMN, palette=PALETTE, corner=True)
    pair_grid.fig.suptitle("Pair Plot of Key Features", y=1.02, fontweight="bold")
    pair_grid.savefig(VISUALS_DIR / "05_pairplot.png", dpi=180)
    plt.close("all")
    print("EDA visuals saved inside visuals/ folder.")


def preprocess_data(df):
    """Steps 4 and 5: Clean data, encode target, engineer features, and scale."""
    print("\nSTEP 4 - DATA PREPROCESSING")
    clean_df = df.copy()
    for column in NUMERIC_COLUMNS:
        clean_df[column] = pd.to_numeric(clean_df[column], errors="coerce")
        if clean_df[column].isnull().any():
            median_value = clean_df[column].median()
            clean_df[column] = clean_df[column].fillna(median_value)
            print(f"Filled missing {column} values with median: {median_value:.2f}")

    label_encoder = LabelEncoder()
    clean_df["RiskLevel_enc"] = label_encoder.fit_transform(clean_df[TARGET_COLUMN])
    print("Label encoding mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

    print("\nSTEP 5 - FEATURE ENGINEERING")
    clean_df["PulsePressure"] = clean_df["SystolicBP"] - clean_df["DiastolicBP"]
    clean_df["Age_BS_Interaction"] = clean_df["Age"] * clean_df["BS"]
    clean_df["AnemiaFlag"] = (clean_df["Hemoglobin"] < 10.0).astype(int)

    x = clean_df[FEATURE_COLUMNS]
    y = clean_df["RiskLevel_enc"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(FEATURE_COLUMNS, MODEL_DIR / "feature_cols.pkl")
    return x_train_scaled, x_test_scaled, y_train, y_test, label_encoder


def balance_training_data(x_train_scaled, y_train):
    """Step 6: Use SMOTE to reduce class imbalance in training data only."""
    print("\nSTEP 6 - HANDLE IMBALANCED DATA USING SMOTE")
    print("Before SMOTE:", pd.Series(y_train).value_counts().to_dict())
    smote = SMOTE(random_state=42)
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train_scaled, y_train)
    print("After SMOTE:", pd.Series(y_train_balanced).value_counts().to_dict())
    return x_train_balanced, y_train_balanced


def get_models():
    """Step 7: Define three suitable ML models for the project."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=4, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss", random_state=42),
    }


def train_and_evaluate_models(models, x_train, y_train, x_test, y_test, label_encoder):
    """Steps 7 and 8: Train models and evaluate them using report metrics."""
    print("\nSTEP 7 & 8 - MODEL BUILDING AND EVALUATION")
    results = {}
    predictions = {}
    high_risk_label = "high risk"
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0)
        results[name] = {
            "Accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
            "Precision": round(report["weighted avg"]["precision"] * 100, 2),
            "Recall": round(report["weighted avg"]["recall"] * 100, 2),
            "F1 Score": round(report["weighted avg"]["f1-score"] * 100, 2),
            "High Risk Recall": round(report[high_risk_label]["recall"] * 100, 2),
        }
        predictions[name] = y_pred
        print(results[name])
    results_df = pd.DataFrame(results).T.sort_values(by=["High Risk Recall", "F1 Score", "Accuracy"], ascending=False)
    print("\nModel comparison:")
    print(results_df)
    return results_df, predictions


def create_model_visuals(models, predictions, results_df, y_test, label_encoder):
    """Step 9: Save confusion matrix, feature importance, and comparison charts."""
    print("\nSTEP 9 - MODEL VISUALIZATIONS")
    fig, axes = plt.subplots(1, len(predictions), figsize=(18, 5))
    for ax, (name, y_pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name, fontweight="bold")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "06_confusion_matrices.png", dpi=180)
    plt.close()

    rf_model = models["Random Forest"]
    feature_importance = pd.Series(rf_model.feature_importances_, index=FEATURE_COLUMNS).sort_values()
    plt.figure(figsize=(10, 6))
    feature_importance.plot(kind="barh", color="#2563eb")
    plt.title("Random Forest Feature Importance", fontweight="bold")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "07_feature_importance.png", dpi=180)
    plt.close()

    results_df[["Accuracy", "F1 Score", "High Risk Recall"]].plot(kind="bar", figsize=(10, 5))
    plt.title("Model Performance Comparison", fontweight="bold")
    plt.ylabel("Score (%)")
    plt.ylim(0, 110)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "08_model_comparison.png", dpi=180)
    plt.close()


def save_best_model(models, results_df):
    """Step 10: Save the best model using High Risk Recall as priority metric."""
    print("\nSTEP 10 - SAVE BEST MODEL")
    best_model_name = results_df.index[0]
    joblib.dump(models[best_model_name], MODEL_DIR / "best_model.pkl")
    print(f"Best model saved: {best_model_name} -> models/best_model.pkl")
    return best_model_name


def main():
    """Run the complete ML workflow from data loading to model saving."""
    setup_directories()
    df = load_dataset()
    run_eda(df)
    x_train, x_test, y_train, y_test, label_encoder = preprocess_data(df)
    x_train_balanced, y_train_balanced = balance_training_data(x_train, y_train)
    models = get_models()
    results_df, predictions = train_and_evaluate_models(models, x_train_balanced, y_train_balanced, x_test, y_test, label_encoder)
    create_model_visuals(models, predictions, results_df, y_test, label_encoder)
    best_model_name = save_best_model(models, results_df)
    results_df.to_csv(REPORTS_DIR / "model_metrics_summary.csv")
    print(f"Metrics saved: reports/model_metrics_summary.csv")
    print(f"Selected model: {best_model_name}")
    print("\nPipeline complete. Run: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
