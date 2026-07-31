"""Pure gate evaluation: metrics in, verdict out.

The engine is intentionally free of any MLflow or framework dependency so the
same policy can gate any model, in any repository.
"""

from datetime import UTC, datetime
from typing import Literal

from gatedops.gate.report import GateCheck, GateReport
from gatedops.gate.rules import ComparisonOp, GateConfig


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
    malformed run fail closed rather than pass by omission.
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
                    passed=False,
                    detail="champion rule enabled but no champion metrics provided",
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
                passed = delta >= -config.champion.tolerance
                checks.append(
                    GateCheck(
                        rule="champion",
                        metric=config.champion.metric,
                        actual=actual,
                        champion=champion,
                        threshold=-config.champion.tolerance,
                        passed=passed,
                        detail=(
                            f"challenger delta {delta:+.4f} vs tolerance "
                            f"{config.champion.tolerance:.4f}"
                        ),
                    )
                )

    status: Literal["PASS", "FAIL"]
    if not checks:
        status = "FAIL"
        summary = "no rules configured: gate cannot pass"
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
