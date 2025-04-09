import tempfile

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, confusion_matrix, \
    roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from src.config.database import DB_CONNECTION
from src.mlflow.config import MLFLOW_SERVER_URL, MLFLOW_EXPERIMENT_NAME

import mlflow
from mlflow.models import infer_signature

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    mlflow.set_tracking_uri(uri=MLFLOW_SERVER_URL)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = pd.read_sql_query('select * from heart_disease',con=DB_CONNECTION)

    # Preprocessing
    skewed_cols = ['ca', 'oldpeak', 'chol', 'trestbps', 'thalach']
    for col in skewed_cols:
        df[col] = np.log1p(df[col])

    X = df.drop('target', axis=1)
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    # Model Building
    model_args = {
        "n_neighbors": 2
    }

    knn = KNeighborsClassifier(**model_args)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    # Model Evaluation
    print(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    mlflow.start_run()
    # Log the hyperparameters
    mlflow.log_params(model_args)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("precision", precision)

    # Metrics artifacts
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    with tempfile.NamedTemporaryFile(suffix=".png") as tmpfile:
        plt.savefig(tmpfile.name)
        mlflow.log_artifact(tmpfile.name, artifact_path="plots")

    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, knn.predict_proba(X_test)[:, 1])
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label="ROC Curve", color="blue")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    with tempfile.NamedTemporaryFile(suffix=".png") as tmpfile:
        plt.savefig(tmpfile.name)
        mlflow.log_artifact(tmpfile.name, artifact_path="plots")

    plt.close()

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("category", "classifier")
    mlflow.set_tag("model", "knn")

    # Infer the model signature
    signature = infer_signature(X_train, knn.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=knn,
        artifact_path="heart_disease_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="knn",
    )
    mlflow.register_model(f"runs:/{model_info.run_id}/model", "knn")

    mlflow.end_run()
