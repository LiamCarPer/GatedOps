"""Request and response contracts for the scoring API.

The request schema is the model contract: a typed feature vector that the API
validates before the model ever sees it. The response carries the lineage of
the model that produced the prediction, so every score can be audited back to
code, data, and training run.
"""

from pydantic import BaseModel, Field

FEATURE_ORDER = [
    "tenure_years",
    "monthly_spend",
    "support_tickets",
    "usage_frequency",
    "engagement_score",
    "has_contract",
    "payment_delay",
]


class ScoreRequest(BaseModel):
    """Feature vector for the churn-classifier demo model."""

    tenure_years: float
    monthly_spend: float
    support_tickets: int
    usage_frequency: float
    engagement_score: float = Field(ge=0.0, le=1.0)
    has_contract: int = Field(ge=0, le=1)
    payment_delay: float


class Lineage(BaseModel):
    """Trace of the served model back to code, data, and run."""

    model_name: str
    model_version: str
    artifact_hash: str
    git_sha: str
    run_id: str
    data_hash: str


class ScoreResponse(BaseModel):
    """Prediction together with the lineage of the model that produced it."""

    prediction: int
    probability: float
    lineage: Lineage
