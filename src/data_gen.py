"""
Generates a synthetic "premium upgrade prediction" dataset.

Why synthetic data for an MLOps project?
-----------------------------------------
The point of this project is NOT to prove we can build an accurate model —
it's to prove we can track, serve, deploy, and monitor one properly.
Using make_classification keeps the modeling side intentionally simple and
fully reproducible (fixed random_state), so all the engineering work below
is the actual deliverable.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

FEATURE_NAMES = [
    "monthly_usage_hours",
    "days_since_signup",
    "num_support_tickets",
    "avg_session_minutes",
    "num_referrals",
    "discount_pct_used",
]


def generate_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=len(FEATURE_NAMES),
        n_informative=4,
        n_redundant=1,
        n_clusters_per_class=2,
        weights=[0.7, 0.3],  # 30% upgrade rate, mirrors realistic conversion rates
        flip_y=0.03,
        random_state=random_state,
    )

    df = pd.DataFrame(X, columns=FEATURE_NAMES)

    # Rescale into human-readable ranges instead of raw standardized floats
    df["monthly_usage_hours"] = (df["monthly_usage_hours"] * 10 + 40).clip(lower=0).round(1)
    df["days_since_signup"] = (df["days_since_signup"] * 60 + 180).clip(lower=1).round(0)
    df["num_support_tickets"] = (df["num_support_tickets"] * 1.5 + 2).clip(lower=0).round(0)
    df["avg_session_minutes"] = (df["avg_session_minutes"] * 8 + 20).clip(lower=1).round(1)
    df["num_referrals"] = (df["num_referrals"] * 1.2 + 1).clip(lower=0).round(0)
    df["discount_pct_used"] = (df["discount_pct_used"] * 10 + 15).clip(lower=0, upper=100).round(1)

    df["upgraded"] = y
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/subscription_data.csv", index=False)
    print(f"Generated {len(df)} rows -> data/subscription_data.csv")
    print(df["upgraded"].value_counts(normalize=True))
