"""
Trains the upgrade-prediction model, logs the experiment to MLflow, and
registers the trained model in the MLflow Model Registry as
"subscription-upgrade-predictor".

Run this script every time you'd retrain in a real MLOps pipeline
(e.g. triggered weekly by new production data). Each run creates a NEW
model version in the registry -- nothing is silently overwritten, so you
always keep a full audit trail of what was trained, when, and how well
it performed.
"""

import json
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from data_gen import generate_dataset, FEATURE_NAMES

MODEL_NAME = "subscription-upgrade-predictor"
MLFLOW_TRACKING_URI = "sqlite:///mlruns/mlflow.db"


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("subscription-upgrade")

    df = generate_dataset()
    X, y = df[FEATURE_NAMES], df["upgraded"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_leaf": 5,
        "random_state": 42,
    }

    with mlflow.start_run() as run:
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, proba),
        }

        # 1. Log params + metrics -> this is "experiment tracking"
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # 2. Log + register the model -> this is the "model registry" step
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        print(f"Run ID: {run.info.run_id}")
        print("Metrics:", json.dumps(metrics, indent=2))

        # 3. Save baseline feature statistics for DRIFT DETECTION later.
        #    This is the "snapshot" of what training data looked like,
        #    which production traffic will later be compared against.
        baseline_stats = {
            col: {
                "mean": float(X_train[col].mean()),
                "std": float(X_train[col].std()),
                "min": float(X_train[col].min()),
                "max": float(X_train[col].max()),
                "sample": X_train[col].sample(min(500, len(X_train)), random_state=1).tolist(),
            }
            for col in FEATURE_NAMES
        }
        with open("models/baseline_stats.json", "w") as f:
            json.dump(baseline_stats, f)
        mlflow.log_artifact("models/baseline_stats.json")

    # Promote the newest version straight to "Production" for this demo.
    # In a real pipeline this step would gate on the metrics above
    # (e.g. only promote if roc_auc > previous production model's roc_auc).
    client = mlflow.tracking.MlflowClient()
    latest_version = client.get_latest_versions(MODEL_NAME, stages=["None"])[0].version
    client.transition_model_version_stage(
        name=MODEL_NAME, version=latest_version, stage="Production", archive_existing_versions=True
    )
    print(f"Promoted version {latest_version} of '{MODEL_NAME}' to Production")


if __name__ == "__main__":
    main()
