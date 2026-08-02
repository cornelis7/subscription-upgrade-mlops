"""
Loads the current "Production" model straight from the MLflow Model Registry.

Why load by stage name ("Production") instead of a fixed file path?
---------------------------------------------------------------------
If we hardcoded a path like models/model_v1.pkl, promoting a new model
would mean editing code and redeploying the API. By asking the registry
"give me whatever is currently Production", retraining + promoting a new
version is enough -- the API picks it up on its next restart (or
immediately, if you call reload_model()).
"""

import mlflow
from functools import lru_cache

MODEL_NAME = "subscription-upgrade-predictor"
MLFLOW_TRACKING_URI = "sqlite:///mlruns/mlflow.db"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

_cached_model = None
_cached_version = None


def load_production_model():
    global _cached_model, _cached_version
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not versions:
        raise RuntimeError(
            f"No model version in 'Production' stage for '{MODEL_NAME}'. "
            "Run `python src/train.py` first."
        )
    version_info = versions[0]
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version_info.version}")
    _cached_model = model
    _cached_version = version_info.version
    return model, version_info.version


def get_model():
    """Returns the cached model, loading it on first call."""
    if _cached_model is None:
        load_production_model()
    return _cached_model, _cached_version


def reload_model():
    """Force-refresh the cached model -- call this after promoting a new version."""
    return load_production_model()
