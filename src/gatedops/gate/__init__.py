"""Contract-driven quality gates that decide whether a model may be promoted."""

from gatedops.gate.engine import evaluate_gate
from gatedops.gate.report import GateCheck, GateReport
from gatedops.gate.rules import ChampionRule, GateConfig, ThresholdRule

__all__ = [
    "ChampionRule",
    "GateCheck",
    "GateConfig",
    "GateReport",
    "ThresholdRule",
    "evaluate_gate",
]
