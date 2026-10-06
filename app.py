import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

model = joblib.load("best_model.pkl")

class Patientinput(BaseModel):
    Pclass: int = Field(..., example=1)
    Sex: str = Field(..., example="female")
    Age: float = Field(..., example=38.0)
    SibSp: int = Field(..., example=1)
    Parch: int = Field(..., example=0)
    Fare: float = Field(..., example=71.28)
    Embarked: str = Field(..., example="C")

app = FastAPI(title="Patient Readmission Prediction", version="1.0")

@app.get("/health")
def health():
    return {"status": "API is running", "model_loaded": True}

@app.post("/predict")
def predict_readmission(patient: Patientinput):
    df_patient = pd.DataFrame([patient.model_dump()])
    probability = float(model.predict_proba(df_patient)[:, 1][0])
    prediction = int(model.predict(df_patient)[0])

    return {
        "readmission_probability": round(probability, 4),
        "prediction": prediction,
        "risk_level": "High Risk" if prediction == 1 else "Low Risk",
    }
