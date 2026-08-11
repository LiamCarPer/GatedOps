"""Prefect flow tests: the gated release promotes on pass, skips on fail.

Both tests share a single module-scoped MLflow registry (same hermetic pattern
as ``test_pipeline.py``) and use distinct model names so the pass and fail
scenarios never see each other's registrations.
"""

import os

import mlflow
import pytest

pytest.importorskip("prefect")

from gatedops.config import DataConfig, ModelConfig, RunConfig
from gatedops.gate.rules import ChampionRule, GateConfig, ThresholdRule
from gatedops.manifest.schema import ModelManifest
from gatedops.pipelines.prefect_flow import gated_flow
from gatedops.registry.mlflow_ import MlflowRegistry


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


def test_gated_flow_promotes_on_pass(_registry) -> None:
    config = _config(_registry, "flow-churn-good")

    outcome = gated_flow(config)

    assert outcome["promoted"] is True
    assert outcome["gate_status"] == "PASS"

    production = MlflowRegistry(tracking_uri=_registry).current_production("flow-churn-good")
    assert production is not None
    assert str(production.version) == outcome["model_version"]

    raw = MlflowRegistry(tracking_uri=_registry).model_version_tag(
        "flow-churn-good", outcome["model_version"], "gatedops.manifest"
    )
    assert raw is not None
    manifest = ModelManifest.model_validate_json(raw)
    assert manifest.promote_stage == "Production"


def test_gated_flow_skips_promotion_on_fail(_registry) -> None:
    config = _config(_registry, "flow-churn-bad")
    config.data.signal_strength = 0.1

    outcome = gated_flow(config)

    assert outcome["promoted"] is False
    assert outcome["gate_status"] == "FAIL"
    assert MlflowRegistry(tracking_uri=_registry).current_production("flow-churn-bad") is None
