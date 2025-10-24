# train_model.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Simple demo dataset
data = pd.DataFrame({
    'value': [80, 120, 140, 160, 100, 90],
    'systolic': [120, 130, 140, 150, 125, 110],
    'diastolic': [80, 85, 90, 95, 82, 75],
    'risk': [0, 0, 1, 1, 0, 0]   # 0=low risk, 1=high risk
})

X = data[['value', 'systolic', 'diastolic']]
y = data['risk']

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "ml_model.pkl")
print("✅ Model trained and saved as ml_model.pkl")
