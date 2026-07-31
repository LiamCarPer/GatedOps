"""Structured output of a gate evaluation."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

RuleKind = Literal["threshold", "champion"]


class GateCheck(BaseModel):
    """Evidence for a single rule evaluation."""

    rule: RuleKind
    metric: str
    actual: float | None = None
    threshold: float | None = None
    champion: float | None = None
    passed: bool
    detail: str = ""


class GateReport(BaseModel):
    """The verdict of a gate run, with per-check evidence for auditability."""

    model_name: str
    status: Literal["PASS", "FAIL"]
    checks: list[GateCheck]
    decided_at: datetime
    summary: str = ""
