"""Declarative gate rules.

A gate is a set of rules a model must satisfy before it is allowed to reach
production. Rules are expressed in a ``gate.yaml`` and validated against the
metrics produced by an evaluation run.

Two kinds of rules are supported:

* ``ThresholdRule`` -- a single metric must compare favourably with a fixed value.
* ``ChampionRule`` -- the challenger must not regress by more than ``tolerance``
  below the current champion on a given metric.
"""

from typing import Literal

from pydantic import BaseModel, Field

ComparisonOp = Literal[">=", "<=", ">", "<"]


class ThresholdRule(BaseModel):
    """Compare a metric against a fixed threshold."""

    metric: str
    op: ComparisonOp
    value: float
    description: str = ""


class ChampionRule(BaseModel):
    """Guard against regressions relative to the model currently in production."""

    metric: str
    tolerance: float = 0.0
    description: str = ""


class GateConfig(BaseModel):
    """The full set of rules evaluated for a model release."""

    thresholds: list[ThresholdRule] = Field(default_factory=list)
    champion: ChampionRule | None = None
