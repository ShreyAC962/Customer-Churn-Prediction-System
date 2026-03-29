# Customer Churn Prediction (MLOps Project)

## Overview
This is an **end-to-end production-ready MLOps project** for predicting customer churn using the Telco dataset.  
It demonstrates the full ML lifecycle: data processing, model training, experiment tracking with **MLflow**, API deployment using **FastAPI**, and Docker containerization.

---

## Project Structure
mlops-churn/
│
├── data/ # Raw dataset
├── notebooks/ # Exploration notebooks
├── src/ # Core ML scripts
│ ├── data_processing.py
│ ├── train.py
│ ├── predict.py
├── models/ # Trained model artifacts
├── api/ # FastAPI service
│ └── app.py
├── requirements.txt
├── Dockerfile
├── .github/workflows/ci.yml


---

## Features

- **Data Processing**: Cleans and encodes categorical features  
- **Model Training**: Uses `RandomForestClassifier` with MLflow experiment tracking  
- **Experiment Tracking**: Logs parameters, metrics, and models using MLflow  
- **Model Serving**: REST API using FastAPI for predictions  
- **Dockerized**: Easy deployment as a container  
- **CI/CD Ready**: Optional GitHub Actions pipeline for automated testing/training

---

## Installation

Clone the repository:

```bash
git clone <repo-url>
cd mlops-churn
```

## Install dependencies:
pip install -r requirements.txt

## Usage
1. Train Model
python src/train.py

This will train the model and log it in MLflow under the experiment churn_prediction.

2. Run API
uvicorn api.app:app --reload

The API will be available at http://127.0.0.1:8000

Example Prediction
curl -X POST http://127.0.0.1:8000/predict \
-H "Content-Type: application/json" \
-d '{"tenure": 5, "MonthlyCharges": 70, "OtherFeature1": value1, "OtherFeature2": value2}'

3. MLflow UI
mlflow ui
Open http://127.0.0.1:5000 in your browser to track experiments, metrics, and models.

4. Dockerize
docker build -t churn-api .
docker run -p 8000:8000 churn-api
The API will be accessible at http://localhost:8000

5. CI/CD 

If using GitHub Actions, every push will:

Install dependencies
Run training script
Ensure reproducibility and no broken pipelines

## Notes
MLflow tracks experiments in the mlruns/ folder locally
Use models:/churn_prediction/latest for production-ready model loading
Customize data_processing.py for feature engineering and preprocessing

## Next Steps (Advanced)
Deploy API to AWS ECS / Kubernetes
Implement feature store and automated retraining
Add model monitoring (data drift, performance drift)