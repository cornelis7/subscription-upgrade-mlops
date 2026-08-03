from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.schemas import PredictionRequest, PredictionResponse
from app.model_loader import get_model, reload_model
from app.monitoring import log_prediction, check_drift, get_predictions, FEATURE_NAMES


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the Production model once at startup instead of on every request
    try:
        get_model()
    except RuntimeError as e:
        print(f"WARNING at startup: {e}")
    yield


app = FastAPI(title="Subscription Upgrade Predictor", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    """Used by Docker/Kubernetes/load balancers to check the service is alive."""
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    try:
        _, version = get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"model_name": "subscription-upgrade-predictor", "production_version": version}


@app.post("/reload-model")
def reload():
    """Call this after promoting a new model version in MLflow, so the API
    picks it up without a full redeploy."""
    try:
        _, version = reload_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"reloaded": True, "production_version": version}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        model, version = get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    features = request.model_dump()
    row = [[features[f] for f in FEATURE_NAMES]]

    proba = model.predict_proba(row)[0][1]
    prediction = int(proba >= 0.5)

    # Log every prediction for monitoring, then check current drift status.
    log_prediction(features, prediction, float(proba))
    drift_result = check_drift()

    return PredictionResponse(
        will_upgrade=bool(prediction),
        upgrade_probability=round(float(proba), 4),
        model_version=str(version),
        drift_warning=drift_result["drifted"],
    )


@app.get("/monitoring/drift")
def drift_status():
    """Full drift report -- this is what the monitoring dashboard reads."""
    return check_drift()


@app.get("/monitoring/history")
def monitoring_history(limit: int = 500):
    """Recent logged predictions as JSON. The dashboard calls this over
    HTTP instead of reading the SQLite file directly, since in production
    it runs on a separate host from the API and doesn't share a disk."""
    return get_predictions(limit=limit)
