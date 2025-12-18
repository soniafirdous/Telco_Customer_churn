import pandas as pd
from fastapi import FastAPI
import numpy as np
import joblib
from pydantic import BaseModel

app = FastAPI()
model = joblib.load('best_model.pkl')
preprocessor = joblib.load("preprocessor.pkl")


# Pydantic model for request
class InputData(BaseModel):
    features: list

feature_names = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
]


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is working!"}


@app.post("/predict")
def predict(data: InputData):

    # Convert to DataFrame
    input_df = pd.DataFrame([data.features], columns=feature_names)

    # Apply preprocessing
    X_processed = preprocessor.transform(input_df)

    # Predict
    prediction = model.predict(X_processed)[0]
    probabilities = model.predict_proba(X_processed).tolist()

    label_map = {"No": 0, "Yes": 1}

    return {
      "class_label": label_map[prediction],
      "prediction": prediction
      , "probabilities": probabilities
    }
