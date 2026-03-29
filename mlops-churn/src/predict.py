import mlflow.sklearn

def load_model():
    model = mlflow.sklearn.load_model("models:/churn_prediction/1")
    return model

def predict(data):
    model = load_model()
    return model.predict(data)