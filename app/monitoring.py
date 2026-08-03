"""
Logs every prediction request to a database table, and checks for data
drift using the two-sample Kolmogorov-Smirnov test.

Storage backend
----------------
Controlled by the DATABASE_URL environment variable:
  - Not set (local dev)   -> falls back to a local SQLite file at
                             models/monitoring.db. Good enough for running
                             the API on your own machine or in Docker.
  - Set to a Postgres URL (production) -> e.g.
                             "postgresql://user:pass@host/dbname"
                             from a free host like Neon or Supabase.
                             This is what makes logged predictions survive
                             server restarts and redeploys, since the data
                             lives outside the container's disk entirely.

SQLAlchemy is the layer that lets the exact same SQL-ish code below work
against either database -- we don't have to write separate SQLite and
Postgres versions of every query.

How the drift check works
--------------------------
For each feature, we have a `sample` of ~500 values from the TRAINING set
(saved in models/baseline_stats.json). Once enough production requests
have come in, we run scipy's `ks_2samp` between that training sample and
the last N production values for the same feature.

The KS test returns a p-value: how likely is it that both samples came
from the same distribution? A low p-value (< 0.05 here) means "unlikely"
-- i.e. the production data has likely drifted from what the model was
trained on. This is a standard, lightweight drift-detection method used
in real MLOps monitoring stacks (e.g. Evidently AI uses the same test
under the hood for numerical features).
"""

import json
import os
import time
from pathlib import Path

from scipy.stats import ks_2samp
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///models/monitoring.db")
engine = create_engine(DATABASE_URL)

BASELINE_PATH = Path("models/baseline_stats.json")
DRIFT_CHECK_MIN_SAMPLES = 30
DRIFT_P_VALUE_THRESHOLD = 0.05

FEATURE_NAMES = [
    "monthly_usage_hours",
    "days_since_signup",
    "num_support_tickets",
    "avg_session_minutes",
    "num_referrals",
    "discount_pct_used",
]

# SQLite and Postgres spell "auto-incrementing integer primary key"
# differently -- this picks the right one for whichever DATABASE_URL is set.
_PK_COLUMN = (
    "id INTEGER PRIMARY KEY AUTOINCREMENT"
    if engine.dialect.name == "sqlite"
    else "id SERIAL PRIMARY KEY"
)


def _ensure_table():
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS predictions (
                {_PK_COLUMN},
                timestamp DOUBLE PRECISION,
                {", ".join(f"{f} DOUBLE PRECISION" for f in FEATURE_NAMES)},
                prediction INTEGER,
                probability DOUBLE PRECISION
            )
        """))


_ensure_table()


def log_prediction(features: dict, prediction: int, probability: float):
    cols = ["timestamp"] + FEATURE_NAMES + ["prediction", "probability"]
    params = {
        "timestamp": time.time(),
        **{f: features[f] for f in FEATURE_NAMES},
        "prediction": prediction,
        "probability": probability,
    }
    placeholders = ", ".join(f":{c}" for c in cols)
    with engine.begin() as conn:
        conn.execute(
            text(f"INSERT INTO predictions ({', '.join(cols)}) VALUES ({placeholders})"),
            params,
        )


def get_predictions(limit: int = 500) -> list[dict]:
    """Returns the most recent logged predictions as a list of dicts, oldest
    first, so the dashboard can plot them in chronological order. This is
    what makes the data readable over HTTP from a dashboard running on a
    completely different host than the API."""
    cols = ["id", "timestamp"] + FEATURE_NAMES + ["prediction", "probability"]
    with engine.connect() as conn:
        rows = conn.execute(
            text(f"SELECT {', '.join(cols)} FROM predictions ORDER BY id DESC LIMIT :limit"),
            {"limit": limit},
        ).fetchall()
    rows = list(reversed(rows))
    return [dict(zip(cols, row)) for row in rows]


def check_drift(window: int = 200) -> dict:
    """Runs a KS-test per feature against the training baseline.

    Returns a dict like:
        {"drifted": bool, "details": {feature: {"p_value": .., "drifted": bool}}}
    """
    if not BASELINE_PATH.exists():
        return {"drifted": False, "details": {}, "reason": "no baseline stats found"}

    baseline = json.loads(BASELINE_PATH.read_text())

    with engine.connect() as conn:
        row_count = conn.execute(text("SELECT COUNT(*) FROM predictions")).scalar()
        if row_count < DRIFT_CHECK_MIN_SAMPLES:
            return {
                "drifted": False,
                "details": {},
                "reason": f"only {row_count} logged predictions, need {DRIFT_CHECK_MIN_SAMPLES}",
            }

        details = {}
        any_drift = False
        for feature in FEATURE_NAMES:
            rows = conn.execute(
                text(f"SELECT {feature} FROM predictions ORDER BY id DESC LIMIT :window"),
                {"window": window},
            ).fetchall()
            recent_values = [r[0] for r in rows]
            baseline_sample = baseline[feature]["sample"]

            stat, p_value = ks_2samp(baseline_sample, recent_values)
            drifted = bool(p_value < DRIFT_P_VALUE_THRESHOLD)
            any_drift = any_drift or drifted
            details[feature] = {"p_value": round(float(p_value), 4), "drifted": drifted}

    return {"drifted": bool(any_drift), "details": details}
