"""Structured output of a gate evaluation."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

RuleKind = Literal["threshold", "champion"]
Verdict = Literal["PASS", "FAIL", "ERROR"]


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
    """The verdict of a gate run, with per-check evidence for auditability.

    ``ERROR`` is distinct from ``FAIL``: it means the gate itself was
    misconfigured (for example, no rules declared) rather than the model
    failing to meet a declared rule.
    """

    model_name: str
    status: Verdict
    checks: list[GateCheck]
    decided_at: datetime
    summary: str = ""
