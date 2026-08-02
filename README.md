# Subscription Upgrade Predictor — MLOps Demo

An end-to-end **MLOps** project. Unlike a typical ML portfolio project, the
model itself (a RandomForestClassifier predicting whether a user will
upgrade to a premium subscription) is intentionally simple — the point of
this project is everything *around* the model:

- **Experiment tracking & model registry** — MLflow
- **Serving** — FastAPI, loading the current "Production" model version
  from the registry (no hardcoded model paths)
- **Containerization** — Docker + docker-compose (API + monitoring dashboard)
- **CI/CD** — GitHub Actions runs the full test suite on every push, and
  only builds the Docker image if tests pass
- **Monitoring & drift detection** — every prediction is logged, and a
  statistical test (Kolmogorov-Smirnov) checks whether production traffic
  has drifted from the training data distribution

## Project structure

```
mlops-churn-service/
├── app/
│   ├── main.py            # FastAPI app: /predict, /health, /monitoring/drift
│   ├── schemas.py         # Pydantic request/response validation
│   ├── model_loader.py    # Loads current Production model from MLflow registry
│   └── monitoring.py      # Prediction logging + KS-test drift detection
├── src/
│   ├── data_gen.py        # Synthetic dataset generator
│   └── train.py           # Trains model, logs to MLflow, promotes to Production
├── monitoring/
│   └── dashboard.py       # Streamlit dashboard: predictions + drift status
├── tests/
│   └── test_api.py        # Test suite run by CI/CD on every push
├── .github/workflows/
│   └── ci-cd.yml          # Test -> build Docker image pipeline
├── Dockerfile              # API service image
├── Dockerfile.dashboard    # Monitoring dashboard image
├── docker-compose.yml      # Runs both services together
└── requirements.txt
```

## How to run locally (without Docker)

```bash
pip install -r requirements.txt

# 1. Train the model (creates mlruns/mlflow.db + models/baseline_stats.json,
#    registers the model, and promotes version 1 to "Production")
python src/train.py

# 2. Start the API
uvicorn app.main:app --reload

# 3. In another terminal, start the monitoring dashboard
streamlit run monitoring/dashboard.py

# 4. Run the test suite
pytest tests/ -v
```

Try a prediction:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "monthly_usage_hours": 45.0,
    "days_since_signup": 180,
    "num_support_tickets": 2,
    "avg_session_minutes": 22.5,
    "num_referrals": 1,
    "discount_pct_used": 15.0
  }'
```

Explore the MLflow UI (experiment history + registered model versions):
```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

## How to run with Docker

```bash
docker compose up --build
```
- API → http://localhost:8000/docs (interactive Swagger UI)
- Dashboard → http://localhost:8501

## The retrain → promote → serve loop

This is the core MLOps loop this project demonstrates:

1. New data comes in → run `python src/train.py` again. This creates a
   **new model version** in the registry (nothing is overwritten) and
   promotes it to Production, archiving the old version.
2. Call `POST /reload-model` on the running API — it re-fetches whatever
   is currently marked Production, **without restarting the container**.
3. Meanwhile, `/monitoring/drift` and the dashboard tell you *when* a
   retrain is actually needed, instead of retraining on a blind schedule.

## Why the drift check works the way it does

Each feature's training-set distribution is snapshotted (500 sample
values) into `models/baseline_stats.json` at training time. Once at least
30 predictions have been logged, `/monitoring/drift` runs a two-sample
Kolmogorov-Smirnov test between that snapshot and the most recent 200
production requests, per feature. A p-value below 0.05 flags that
feature as drifted. This mirrors the statistical method real monitoring
tools (e.g. Evidently AI) use under the hood for numerical features.

## What's intentionally left out (portfolio talking points)

Mentioning these in an interview shows you know the boundaries of this
demo vs. a real production system:
- No authentication/rate-limiting on the API
- MLflow uses local SQLite instead of a hosted tracking server + S3/GCS
  artifact store
- CI/CD builds the image but doesn't push it to a registry or deploy it
  (would need cloud credentials)
- Drift detection only checks feature drift, not label/concept drift
  (would require ground-truth labels arriving later)
