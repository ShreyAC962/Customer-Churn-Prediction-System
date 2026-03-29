from fastapi import FastAPI
import pandas as pd
from src.predict import predict

app = FastAPI()

@app.get("/")
def home():
    return {"message" : " Churn Prediction API"}

@app.post("/predict")
def make_prediction(data : dict):
    df = pd.DataFrame([data]) #creates a table and treat this as one row
    result = predict(df)
    return {"prediction" : int(result[0])}