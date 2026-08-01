"""End-to-end tests for the pipeline: gate pass, gate block, champion guard.

The tests share a single module-scoped MLflow registry so database
initialization happens once, and each scenario uses its own model name to stay
independent.
"""

import os

import mlflow
import pytest

from gatedops.config import DataConfig, ModelConfig, RunConfig
from gatedops.gate.rules import ChampionRule, GateConfig, ThresholdRule
from gatedops.pipelines.run import run_pipeline


@pytest.fixture(scope="module", autouse=True)
def _registry(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("gatedops")
    os.chdir(workdir)
    tracking_uri = f"sqlite:///{workdir / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    yield tracking_uri


def _config(tracking_uri: str, model_name: str) -> RunConfig:
    return RunConfig(
        model_name=model_name,
        tracking_uri=tracking_uri,
        gate=GateConfig(
            thresholds=[
                ThresholdRule(metric="f1", op=">=", value=0.70),
                ThresholdRule(metric="false_alarm_rate", op="<=", value=0.10),
            ],
            champion=ChampionRule(metric="f1", tolerance=0.02),
        ),
        data=DataConfig(n_rows=4000, signal_strength=0.9, seed=1),
        model=ModelConfig(),
    )


def test_good_model_passes_and_promotes(_registry) -> None:
    result = run_pipeline(_config(_registry, "churn-good"))

    assert result.gate.status == "PASS"
    assert result.promoted is True
    assert result.receipt is not None
    assert result.manifest.promote_stage == "Production"
    assert result.manifest.gate is not None


def test_bad_model_is_blocked(_registry) -> None:
    config = _config(_registry, "churn-bad")
    config.data.signal_strength = 0.1

    result = run_pipeline(config)

    assert result.gate.status == "FAIL"
    assert result.promoted is False
    assert result.receipt is None
    assert result.manifest.promote_stage == "Staging"


def test_champion_regression_is_blocked(_registry) -> None:
    champion = run_pipeline(_config(_registry, "churn-champion"))
    assert champion.promoted is True

    challenger = _config(_registry, "churn-champion")
    challenger.data.signal_strength = 0.5
    result = run_pipeline(challenger)

    assert result.gate.status == "FAIL"
    assert any(check.rule == "champion" and not check.passed for check in result.gate.checks)
    assert result.promoted is False
