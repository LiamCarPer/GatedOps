"""Schema for the lineage manifest emitted by every model release."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from gatedops.gate.report import GateReport

PromoteStage = Literal["None", "Staging", "Production"]


class ModelManifest(BaseModel):
    """Auditable record of one model release.

    The combination of ``run_id``, ``artifact_hash``, ``git_sha`` and
    ``data_hash`` lets any served prediction be traced back to the exact code,
    dataset, training run and model bytes that produced it.
    """

    model_name: str
    model_version: str
    artifact_hash: str
    run_id: str
    git_sha: str
    data_hash: str
    metrics: dict[str, float] = Field(default_factory=dict)
    gate: GateReport | None = None
    promote_stage: PromoteStage = "None"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
