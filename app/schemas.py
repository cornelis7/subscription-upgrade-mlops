from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    monthly_usage_hours: float = Field(..., ge=0, le=200)
    days_since_signup: float = Field(..., ge=0)
    num_support_tickets: float = Field(..., ge=0)
    avg_session_minutes: float = Field(..., ge=0)
    num_referrals: float = Field(..., ge=0)
    discount_pct_used: float = Field(..., ge=0, le=100)

    class Config:
        json_schema_extra = {
            "example": {
                "monthly_usage_hours": 45.0,
                "days_since_signup": 180,
                "num_support_tickets": 2,
                "avg_session_minutes": 22.5,
                "num_referrals": 1,
                "discount_pct_used": 15.0,
            }
        }


class PredictionResponse(BaseModel):
    will_upgrade: bool
    upgrade_probability: float
    model_version: str
    drift_warning: bool
