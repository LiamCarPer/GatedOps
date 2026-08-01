"""Pure gate evaluation: metrics in, verdict out.

The engine is intentionally free of any MLflow or framework dependency so the
same policy can gate any model, in any repository.
"""

from datetime import UTC, datetime
from typing import Literal

from gatedops.gate.report import GateCheck, GateReport
from gatedops.gate.rules import ComparisonOp, GateConfig

# Metrics are floats; a challenger that ties the required margin within float
# noise must be treated as meeting the rule, not rejected at the boundary.
_EPSILON = 1e-9


def _compare(actual: float, op: ComparisonOp, value: float) -> bool:
    if op in (">=", ">"):
        return actual >= value if op == ">=" else actual > value
    return actual <= value if op == "<=" else actual < value


def evaluate_gate(
    config: GateConfig,
    metrics: dict[str, float],
    model_name: str = "unnamed",
    champion_metrics: dict[str, float] | None = None,
) -> GateReport:
    """Evaluate ``metrics`` (and optionally the champion's) against ``config``.

    A check fails if its metric is missing from the evaluation, which makes a
    malformed run fail closed rather than pass by omission. The champion guard
    is the one exception: before a first release there is no baseline, so it
    passes vacuously until the registry holds a production model.
    """
    checks: list[GateCheck] = []

    for rule in config.thresholds:
        actual = metrics.get(rule.metric)
        if actual is None:
            checks.append(
                GateCheck(
                    rule="threshold",
                    metric=rule.metric,
                    passed=False,
                    detail=f"metric {rule.metric!r} missing from evaluation metrics",
                )
            )
            continue
        passed = _compare(actual, rule.op, rule.value)
        checks.append(
            GateCheck(
                rule="threshold",
                metric=rule.metric,
                actual=actual,
                threshold=rule.value,
                passed=passed,
                detail=rule.description,
            )
        )

    if config.champion is not None:
        if champion_metrics is None:
            checks.append(
                GateCheck(
                    rule="champion",
                    metric=config.champion.metric,
                    passed=True,
                    detail="no champion in registry; champion guard is vacuous",
                )
            )
        else:
            actual = metrics.get(config.champion.metric)
            champion = champion_metrics.get(config.champion.metric)
            if actual is None or champion is None:
                checks.append(
                    GateCheck(
                        rule="champion",
                        metric=config.champion.metric,
                        passed=False,
                        detail=(
                            f"metric {config.champion.metric!r} missing from "
                            "challenger or champion metrics"
                        ),
                    )
                )
            else:
                delta = actual - champion
                if config.champion.higher_is_better:
                    required = config.champion.min_delta - config.champion.tolerance
                    passed = delta >= required - _EPSILON
                else:
                    required = config.champion.tolerance - config.champion.min_delta
                    passed = delta <= required + _EPSILON
                checks.append(
                    GateCheck(
                        rule="champion",
                        metric=config.champion.metric,
                        actual=actual,
                        champion=champion,
                        threshold=required,
                        passed=passed,
                        detail=(
                            f"challenger delta {delta:+.4f} vs required "
                            f"<= {required:+.4f}"
                            if not config.champion.higher_is_better
                            else f"challenger delta {delta:+.4f} vs required "
                            f">= {required:+.4f}"
                        ),
                    )
                )

    status: Literal["PASS", "FAIL", "ERROR"]
    if not checks:
        status = "ERROR"
        summary = "no rules configured: the gate is misconfigured"
    else:
        passed_checks = sum(1 for check in checks if check.passed)
        status = "PASS" if passed_checks == len(checks) else "FAIL"
        summary = f"{passed_checks}/{len(checks)} checks passed"

    return GateReport(
        model_name=model_name,
        status=status,
        checks=checks,
        decided_at=datetime.now(UTC),
        summary=summary,
    )
