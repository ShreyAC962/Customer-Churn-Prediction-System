import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from data_processing import load_data, preprocess

mlflow.set_experiment("churn_prediction")

df = load_data("data/churn.csv")
X_train, X_test, y_train, y_test = preprocess(df)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=1000)

    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metrics("accuracy", accuracy)

