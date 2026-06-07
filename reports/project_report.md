# SafeBirth: Maternal Risk Score Predictor
## Pre Final Year Project Report - B.Tech Computer Science and Engineering

## Abstract

SafeBirth is a machine learning based maternal health screening system that predicts pregnancy risk level as Low Risk, Mid Risk, or High Risk. The project uses basic maternal health readings such as age, blood pressure, blood sugar, body temperature, heart rate, and hemoglobin. The objective is to support early risk identification using a simple and explainable ML pipeline suitable for a final year engineering project.

The system performs data loading, EDA, preprocessing, feature engineering, class balancing using SMOTE, model training, model evaluation, and deployment using Streamlit. Logistic Regression, Random Forest, and XGBoost are compared. The best model is selected with special focus on High Risk Recall because missing a high-risk pregnancy can be dangerous.

## Introduction

Maternal health monitoring is important during pregnancy because early detection of risk can help doctors and healthcare workers take timely action. In many places, expert medical support may not always be immediately available. Machine learning can assist by learning patterns from historical maternal health data and providing a quick risk category based on measurable readings.

SafeBirth is designed as an educational clinical decision-support prototype. It is not a replacement for doctors, but it demonstrates how machine learning can be applied to healthcare screening.

## Objectives

1. Build a supervised ML model to classify maternal risk into three classes.
2. Perform EDA to understand feature behavior and risk patterns.
3. Handle missing values and class imbalance properly.
4. Create simple medical features for better interpretability.
5. Compare Logistic Regression, Random Forest, and XGBoost.
6. Deploy the trained model using a modern Streamlit interface.
7. Make the project report-ready and easy to explain in viva.

## Dataset Description

| Feature | Description | Unit |
|---|---|---|
| Age | Patient age | years |
| SystolicBP | Upper blood pressure reading | mmHg |
| DiastolicBP | Lower blood pressure reading | mmHg |
| BS | Blood sugar level | mmol/L |
| BodyTemp | Body temperature | F |
| HeartRate | Resting heart rate | bpm |
| Hemoglobin | Hemoglobin level | g/dL |
| RiskLevel | Target class | low risk, mid risk, high risk |

## Methodology

### Data Loading

The dataset is loaded using Pandas. Shape, data types, missing values, first few records, and class distribution are checked. This helps confirm whether the dataset is ready for analysis.

### Exploratory Data Analysis

The EDA stage includes risk level distribution chart, histograms, boxplots, correlation heatmap, and pair plot. These charts show how each medical reading changes across Low, Mid, and High risk groups.

### EDA Findings

Blood pressure and blood sugar are usually strong indicators of maternal risk. Higher systolic and diastolic BP values are commonly associated with high-risk pregnancy. Blood sugar is important because high glucose may indicate gestational diabetes risk. Hemoglobin is useful because low hemoglobin may indicate anemia. Age can increase risk when combined with abnormal sugar or BP readings.

### Data Preprocessing

Missing numerical values are filled using the median because median is less affected by outliers than mean. The target variable RiskLevel is converted into numbers using LabelEncoder. Features are scaled using StandardScaler so that measurements with different units do not dominate the model.

### Feature Engineering

| Feature | Formula | Reason |
|---|---|---|
| PulsePressure | SystolicBP - DiastolicBP | Captures BP pressure difference |
| Age_BS_Interaction | Age x BS | Captures combined age and sugar risk |
| AnemiaFlag | 1 if Hemoglobin < 10 else 0 | Flags possible anemia |

### Handling Imbalanced Data

Class imbalance can make the model biased toward majority classes. In healthcare, this is dangerous because the model may miss High Risk patients. SMOTE is used only on the training data to create synthetic samples for minority classes.

### Model Building

Three models are trained: Logistic Regression, Random Forest, and XGBoost. Logistic Regression is a simple baseline, Random Forest is robust and interpretable, and XGBoost is a strong boosting model for tabular data.

### Model Evaluation

The models are evaluated using Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and High Risk Recall. High Risk Recall is the most important metric because it measures how many actual high-risk cases were correctly detected.

## Streamlit Application

The web app includes a sidebar, clear patient input sliders with field names and units, prediction button, color-coded result card, confidence score, probability chart, input summary, engineered feature explanation, batch CSV prediction, EDA visual gallery, and project explanation tab.

## Conclusion

SafeBirth demonstrates a practical machine learning workflow for maternal risk prediction. It combines EDA, preprocessing, feature engineering, SMOTE, model comparison, and Streamlit deployment. The project is suitable for final year presentation because it is practical, explainable, and connected to a real healthcare problem.

## Future Scope

1. Use a larger real-world hospital dataset.
2. Add SHAP explainability for individual predictions.
3. Add multilingual support for rural healthcare workers.
4. Convert the app into a mobile application.
5. Integrate patient history and follow-up visit data.
6. Add authentication for healthcare staff.
7. Deploy on Streamlit Community Cloud.

## Common Viva Questions

**Q: Why did you choose this project?**
A: Maternal health is an important real-world problem, and the dataset uses simple readings that are easy to collect.

**Q: Why did you use Random Forest?**
A: Random Forest handles non-linear patterns well, works strongly on tabular data, and provides feature importance.

**Q: Why did you use SMOTE?**
A: It balances the training data so the model does not ignore the minority High Risk class.

**Q: Why is High Risk Recall important?**
A: A false negative in High Risk prediction can delay treatment, so recall is critical.

**Q: Is this app a medical device?**
A: No. It is an academic project and should only be used as a demonstration or screening-support prototype.
