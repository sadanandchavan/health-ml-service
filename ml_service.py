# ml_service.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import joblib
import logging
import os


# --- Logging setup ---
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/ml_predictions.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Load ML model ---
model = joblib.load("ml_model.pkl")

# --- FastAPI app ---
app = FastAPI(title="Enhanced Health ML Service")

# --- Request schemas with examples ---
class Reading(BaseModel):
    value: Optional[int] = Field(None, example=150)
    systolic: Optional[int] = Field(None, example=140)
    diastolic: Optional[int] = Field(None, example=90)
    type: Optional[str] = Field("glucose", example="glucose")
    context: Optional[dict] = Field({}, example={"meal":"after","notes":"Feeling dizzy"})

class BatchRequest(BaseModel):
    readings: List[Reading] = Field(..., example=[
        {"value":150,"systolic":140,"diastolic":90,"type":"glucose","context":{"meal":"after","notes":"Feeling dizzy"}},
        {"value":120,"systolic":130,"diastolic":85,"type":"glucose","context":{"meal":"before","notes":"Fasting"}},
        {"value":95,"systolic":110,"diastolic":70,"type":"bp","context":{"position":"sitting","notes":"Normal"}}
    ])

# --- Helper functions ---
def categorize_risk(score: float) -> str:
    if score > 0.7:
        return "High Risk"
    elif score > 0.4:
        return "Moderate Risk"
    else:
        return "Low Risk"

def compute_alert(score: float) -> str:
    if score > 0.7:
        return "High risk — consult doctor immediately"
    elif score > 0.4:
        return "Moderate risk — monitor closely"
    else:
        return "Normal"

def log_request(endpoint: str, input_data, output_data):
    logging.info(f"Endpoint: {endpoint}, Input: {input_data}, Output: {output_data}")

# --- Endpoints ---
#@app.get("/")
#def root():
#    return {"message": "ML Service is running!"}

@app.post("/predict", summary="Predict risk for a single reading", response_description="Risk score and category")
def predict(reading: Reading):
    X = pd.DataFrame([{
        "value": reading.value or 0,
        "systolic": reading.systolic or 0,
        "diastolic": reading.diastolic or 0
    }])
    score = model.predict_proba(X)[:, 1][0]
    risk = categorize_risk(score)
    output = {"prediction": float(score), "risk_category": risk}
    log_request("/predict", reading.dict(), output)
    return output

@app.post("/predict_batch", summary="Predict risk for multiple readings", response_description="List of risk scores and categories")
def predict_batch(req: BatchRequest):
    df = pd.DataFrame([{
        "value": r.value or 0,
        "systolic": r.systolic or 0,
        "diastolic": r.diastolic or 0
    } for r in req.readings])
    scores = model.predict_proba(df)[:, 1].tolist()
    risks = [categorize_risk(s) for s in scores]
    output = {"predictions": [{"score": s, "risk_category": r} for s, r in zip(scores, risks)]}
    log_request("/predict_batch", [r.dict() for r in req.readings], output)
    return output

@app.post("/alert", summary="Generate health alert for a single reading", response_description="Risk score and alert message")
def generate_alert(reading: Reading):
    X = pd.DataFrame([{
        "value": reading.value or 0,
        "systolic": reading.systolic or 0,
        "diastolic": reading.diastolic or 0
    }])
    score = model.predict_proba(X)[:, 1][0]
    alert = compute_alert(score)
    output = {"risk_score": float(score), "alert": alert}
    log_request("/alert", reading.dict(), output)
    return output



from fastapi.staticfiles import StaticFiles
# Serve the static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Optional: root redirects to HTML page
from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/static/index.html")
