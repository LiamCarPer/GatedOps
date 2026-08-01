"""Unit tests for the gate engine and its rules."""

from gatedops.gate.engine import evaluate_gate
from gatedops.gate.report import GateReport
from gatedops.gate.rules import ChampionRule, GateConfig, ThresholdRule


def test_threshold_rules_pass() -> None:
    config = GateConfig(
        thresholds=[
            ThresholdRule(metric="f1", op=">=", value=0.75),
            ThresholdRule(metric="false_alarm_rate", op="<=", value=0.05),
        ]
    )
    report = evaluate_gate(config, {"f1": 0.86, "false_alarm_rate": 0.02}, model_name="churn-v1")

    assert report.status == "PASS"
    assert len(report.checks) == 2
    assert all(check.passed for check in report.checks)


def test_threshold_rule_blocks_bad_model() -> None:
    config = GateConfig(thresholds=[ThresholdRule(metric="f1", op=">=", value=0.75)])
    report = evaluate_gate(config, {"f1": 0.51}, model_name="churn-v1")

    assert report.status == "FAIL"
    assert report.checks[0].passed is False


def test_missing_metric_fails_closed() -> None:
    config = GateConfig(thresholds=[ThresholdRule(metric="f1", op=">=", value=0.75)])
    report = evaluate_gate(config, {"precision": 0.9}, model_name="churn-v1")

    assert report.status == "FAIL"
    assert "missing" in report.checks[0].detail


def test_champion_within_tolerance_passes() -> None:
    config = GateConfig(champion=ChampionRule(metric="f1", tolerance=0.01))
    report = evaluate_gate(
        config, {"f1": 0.80}, champion_metrics={"f1": 0.808}, model_name="churn-v1"
    )

    assert report.status == "PASS"


def test_champion_regression_beyond_tolerance_fails() -> None:
    config = GateConfig(champion=ChampionRule(metric="f1", tolerance=0.01))
    report = evaluate_gate(
        config, {"f1": 0.79}, champion_metrics={"f1": 0.81}, model_name="churn-v1"
    )

    assert report.status == "FAIL"
    assert report.checks[0].rule == "champion"


def test_champion_rule_vacuous_without_champion() -> None:
    config = GateConfig(champion=ChampionRule(metric="f1", tolerance=0.01))
    report = evaluate_gate(config, {"f1": 0.80}, model_name="churn-v1")

    assert report.status == "PASS"
    assert "vacuous" in report.checks[0].detail


def test_empty_config_never_passes() -> None:
    report = evaluate_gate(GateConfig(), {"f1": 0.99}, model_name="churn-v1")

    assert report.status == "FAIL"
    assert report.checks == []


def test_report_roundtrips_through_json() -> None:
    config = GateConfig(thresholds=[ThresholdRule(metric="f1", op=">=", value=0.75)])
    report = evaluate_gate(config, {"f1": 0.86}, model_name="churn-v1")

    restored = GateReport.model_validate_json(report.model_dump_json())

    assert restored == report
