# SafeBirth - Maternal Risk Score Predictor

SafeBirth is a B.Tech CSE machine learning project that predicts maternal pregnancy risk as Low Risk, Mid Risk, or High Risk using basic health readings.

## Features

- Single patient risk prediction using Streamlit sliders
- Clearly visible input names, units, and value ranges
- Color-coded prediction result: green, yellow, red
- Prediction confidence and class probability chart
- Engineered feature explanation for transparency
- Batch CSV upload and downloadable prediction results
- EDA visual gallery inside the app
- Modular code structure for easier viva explanation

## Folder Structure

```text
SafeBirth_Project/
├── data/
│   ├── maternal_health.csv
│   └── generate_dataset.py
├── notebooks/
│   └── train_pipeline.py
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── feature_cols.pkl
├── app/
│   ├── streamlit_app.py
│   ├── config.py
│   ├── model_service.py
│   ├── prediction_service.py
│   ├── ui_helpers.py
│   └── __init__.py
├── reports/
│   ├── project_report.md
│   └── model_metrics_summary.csv
├── visuals/
├── requirements.txt
└── README.md
```

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- imbalanced-learn SMOTE
- Joblib
- Streamlit

## Dataset Columns

| Column | Meaning | Unit |
|---|---|---|
| Age | Patient age | years |
| SystolicBP | Upper blood pressure | mmHg |
| DiastolicBP | Lower blood pressure | mmHg |
| BS | Blood sugar | mmol/L |
| BodyTemp | Body temperature | F |
| HeartRate | Heart rate | bpm |
| Hemoglobin | Hemoglobin level | g/dL |
| RiskLevel | Target class | low risk, mid risk, high risk |

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model and generate visuals:

```bash
python notebooks/train_pipeline.py
```

Run the web app:

```bash
streamlit run app/streamlit_app.py
```

## Methodology

1. Load maternal health CSV dataset.
2. Check shape, data types, null values, and class distribution.
3. Perform EDA using histograms, boxplots, heatmap, pair plot, and count plot.
4. Fill missing numerical values using median.
5. Encode target labels using LabelEncoder.
6. Create engineered features: PulsePressure, Age_BS_Interaction, AnemiaFlag.
7. Scale features using StandardScaler.
8. Balance the training data using SMOTE.
9. Train Logistic Regression, Random Forest, and XGBoost.
10. Evaluate using Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and High Risk Recall.
11. Save the best model and deploy with Streamlit.

## Viva Questions

**Q: Why did you use SMOTE?**
A: SMOTE balances the training data by creating synthetic minority-class samples. This helps the model learn High Risk cases better.

**Q: Why is High Risk Recall important?**
A: In healthcare, missing a high-risk patient is more dangerous than giving a false warning, so High Risk Recall is a key metric.

**Q: Why did you create PulsePressure?**
A: PulsePressure captures the difference between systolic and diastolic BP, which can indicate cardiovascular stress.

**Q: Why Streamlit?**
A: Streamlit is simple, fast, and suitable for college ML project deployment.

## Disclaimer

This project is for academic demonstration only. It is not a certified medical diagnosis system and should not replace professional medical advice.
