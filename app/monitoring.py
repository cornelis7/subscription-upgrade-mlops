"""
Logs every prediction request to a local SQLite table (a stand-in for a
real monitoring store like Postgres/BigQuery in production), and checks
for data drift using the two-sample Kolmogorov-Smirnov test.

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
import sqlite3
import time
from pathlib import Path

from scipy.stats import ks_2samp

DB_PATH = Path("models/monitoring.db")
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


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            {", ".join(f"{f} REAL" for f in FEATURE_NAMES)},
            prediction INTEGER,
            probability REAL
        )
        """
    )
    return conn


def log_prediction(features: dict, prediction: int, probability: float):
    conn = _get_conn()
    cols = ["timestamp"] + FEATURE_NAMES + ["prediction", "probability"]
    values = [time.time()] + [features[f] for f in FEATURE_NAMES] + [prediction, probability]
    placeholders = ", ".join(["?"] * len(cols))
    conn.execute(f"INSERT INTO predictions ({', '.join(cols)}) VALUES ({placeholders})", values)
    conn.commit()
    conn.close()


def check_drift(window: int = 200) -> dict:
    """Runs a KS-test per feature against the training baseline.

    Returns a dict like:
        {"drifted": bool, "details": {feature: {"p_value": .., "drifted": bool}}}
    """
    if not BASELINE_PATH.exists():
        return {"drifted": False, "details": {}, "reason": "no baseline stats found"}

    baseline = json.loads(BASELINE_PATH.read_text())

    conn = _get_conn()
    row_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    if row_count < DRIFT_CHECK_MIN_SAMPLES:
        conn.close()
        return {"drifted": False, "details": {}, "reason": f"only {row_count} logged predictions, need {DRIFT_CHECK_MIN_SAMPLES}"}

    details = {}
    any_drift = False
    for feature in FEATURE_NAMES:
        rows = conn.execute(
            f"SELECT {feature} FROM predictions ORDER BY id DESC LIMIT ?", (window,)
        ).fetchall()
        recent_values = [r[0] for r in rows]
        baseline_sample = baseline[feature]["sample"]

        stat, p_value = ks_2samp(baseline_sample, recent_values)
        drifted = bool(p_value < DRIFT_P_VALUE_THRESHOLD)
        any_drift = any_drift or drifted
        details[feature] = {"p_value": round(float(p_value), 4), "drifted": drifted}

    conn.close()
    return {"drifted": bool(any_drift), "details": details}
