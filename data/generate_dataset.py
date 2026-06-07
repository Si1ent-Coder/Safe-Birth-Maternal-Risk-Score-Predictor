"""
SafeBirth Dataset Generator
Generates a realistic synthetic maternal health dataset
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1200  # total samples

rows = []

def make_patient(risk):
    if risk == "low risk":
        age        = np.random.randint(20, 33)
        sbp        = np.random.randint(100, 120)
        dbp        = np.random.randint(65, 80)
        bs         = round(np.random.uniform(6.0, 7.5), 1)
        temp       = round(np.random.uniform(98.0, 98.9), 1)
        hr         = np.random.randint(65, 80)
        hb         = round(np.random.uniform(11.5, 14.5), 1)
    elif risk == "mid risk":
        age        = np.random.randint(25, 40)
        sbp        = np.random.randint(120, 140)
        dbp        = np.random.randint(80, 90)
        bs         = round(np.random.uniform(7.5, 10.0), 1)
        temp       = round(np.random.uniform(99.0, 100.0), 1)
        hr         = np.random.randint(76, 90)
        hb         = round(np.random.uniform(9.5, 11.5), 1)
    else:  # high risk
        age        = np.random.randint(35, 55)
        sbp        = np.random.randint(140, 180)
        dbp        = np.random.randint(90, 110)
        bs         = round(np.random.uniform(10.0, 19.0), 1)
        temp       = round(np.random.uniform(100.0, 103.0), 1)
        hr         = np.random.randint(85, 110)
        hb         = round(np.random.uniform(6.5, 9.5), 1)
    return [age, sbp, dbp, bs, temp, hr, hb, risk]

# Class distribution: 40% low, 35% mid, 25% high
for _ in range(480): rows.append(make_patient("low risk"))
for _ in range(420): rows.append(make_patient("mid risk"))
for _ in range(300): rows.append(make_patient("high risk"))

df = pd.DataFrame(rows, columns=[
    "Age","SystolicBP","DiastolicBP","BS","BodyTemp","HeartRate","Hemoglobin","RiskLevel"
])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Introduce ~3% missing values for realism
for col in ["BS","Hemoglobin","BodyTemp"]:
    mask = np.random.rand(len(df)) < 0.03
    df.loc[mask, col] = np.nan

df.to_csv("data/maternal_health.csv", index=False)
print(f"Dataset saved  →  {df.shape[0]} rows × {df.shape[1]} cols")
print(df["RiskLevel"].value_counts())
