"""Pydantic schemas for the credit risk scoring API."""
from typing import List, Optional
from pydantic import BaseModel, Field


class ApplicantInput(BaseModel):
    """Input: a dictionary of raw applicant features.

    Keys should match Home Credit raw column names (e.g., AMT_INCOME_TOTAL,
    NAME_EDUCATION_TYPE). Engineered features are computed internally.
    """
    features: dict = Field(
        ...,
        description="Raw applicant features as key-value pairs."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": {
                    "AMT_INCOME_TOTAL": 180000.0,
                    "AMT_CREDIT": 500000.0,
                    "AMT_ANNUITY": 25000.0,
                    "AMT_GOODS_PRICE": 450000.0,
                    "DAYS_BIRTH": -12000,
                    "DAYS_EMPLOYED": -2000,
                    "NAME_EDUCATION_TYPE": "Higher education",
                    "NAME_CONTRACT_TYPE": "Cash loans"
                }
            }
        }
    }


class ReasonCode(BaseModel):
    """A single SHAP-based reason code."""
    feature: str
    value: Optional[str] = None
    impact: float
    direction: str  # "increases_risk" or "decreases_risk"


class PredictionResponse(BaseModel):
    """Output of the scoring endpoint."""
    default_probability: float = Field(..., ge=0, le=1)
    decision: str  # approve / review / reject
    expected_loss: float
    reason_codes: List[ReasonCode]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool