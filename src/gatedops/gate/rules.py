"""Declarative gate rules.

A gate is a set of rules a model must satisfy before it is allowed to reach
production. Rules are expressed in a ``gate.yaml`` and validated against the
metrics produced by an evaluation run.

Two kinds of rules are supported:

* ``ThresholdRule`` -- a single metric must compare favourably with a fixed value.
* ``ChampionRule`` -- the challenger must beat, or at least not regress beyond
  a tolerance from, the model currently in production.

Metric direction matters for champion comparison: ``f1`` is higher-is-better
while ``false_alarm_rate`` is lower-is-better. ``ChampionRule`` carries an
explicit ``higher_is_better`` flag so a regression on a lower-is-better metric
(increase) is detected correctly.
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
    """Guard against regressions relative to the model currently in production.

    For higher-is-better metrics the challenger must score at least
    ``champion + min_delta - tolerance``. For lower-is-better metrics it must
    score at most ``champion - min_delta + tolerance``. With defaults this
    means "do not regress by more than ``tolerance``".
    """

    metric: str
    tolerance: float = 0.0
    min_delta: float = 0.0
    higher_is_better: bool = True
    description: str = ""


class GateConfig(BaseModel):
    """The full set of rules evaluated for a model release."""

    thresholds: list[ThresholdRule] = Field(default_factory=list)
    champion: ChampionRule | None = None
