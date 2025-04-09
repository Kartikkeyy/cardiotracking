import requests
from fastapi import FastAPI
import pandas as pd
from src.mlflow.config import MLFLOW_SERVER_URL
import mlflow

mlflow.set_tracking_uri(uri=MLFLOW_SERVER_URL)
mlflow.set_experiment("Heart Disease Classifier")
app = FastAPI()

@app.post("/predict")
def predict():
    client = mlflow.tracking.MlflowClient()

    try:
        model = client.get_latest_versions("knn")[0]

        run_id = model.run_id
        model_path = f"runs:/{run_id}/heart_disease_model"

        model = mlflow.sklearn.load_model(model_path)

        # Sample input
        input_sample = {
            'age': 52, 'sex': 1, 'cp': 0, 'trestbps': 125, 'chol': 212, 'fbs': 0,
            'restecg': 1, 'thalach': 168, 'exang': 0, 'oldpeak': 1, 'slope': 2,
            'ca': 2, 'thal': 3
        }

        input_df = pd.DataFrame([input])
        prediction = model.predict(input_df)

        return {"prediction": int(prediction[0])}
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return {"error": "Model not found"}