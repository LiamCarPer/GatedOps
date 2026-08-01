"""Integration tests for the scoring API against a real promoted model."""

import os

import mlflow
import pytest
from fastapi.testclient import TestClient

from gatedops.config import DataConfig, ModelConfig, RunConfig
from gatedops.gate.rules import ChampionRule, GateConfig, ThresholdRule
from gatedops.pipelines.run import run_pipeline
from gatedops.serve.app import create_app
from gatedops.serve.config import ServeConfig

_VALID_REQUEST = {
    "tenure_years": 2.5,
    "monthly_spend": 49.9,
    "support_tickets": 2,
    "usage_frequency": 41.0,
    "engagement_score": 0.4,
    "has_contract": 0,
    "payment_delay": 3.2,
}


@pytest.fixture(scope="module", autouse=True)
def _env(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("gatedops-serve")
    os.chdir(workdir)
    tracking_uri = f"sqlite:///{workdir / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)

    config = RunConfig(
        model_name="churn-api",
        tracking_uri=tracking_uri,
        gate=GateConfig(
            thresholds=[ThresholdRule(metric="f1", op=">=", value=0.70)],
            champion=ChampionRule(metric="f1", tolerance=0.02),
        ),
        data=DataConfig(n_rows=4000, signal_strength=0.9, seed=1),
        model=ModelConfig(),
    )
    result = run_pipeline(config)
    assert result.promoted is True
    yield tracking_uri, result


def _client(tracking_uri: str, model_name: str = "churn-api") -> TestClient:
    app = create_app(
        ServeConfig(model_name=model_name, tracking_uri=tracking_uri, autoreload=False)
    )
    return TestClient(app)


def test_score_returns_matching_lineage(_env) -> None:
    tracking_uri, result = _env
    client = _client(tracking_uri)

    response = client.post("/score", json=_VALID_REQUEST)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0

    lineage = body["lineage"]
    manifest = result.manifest
    assert lineage["model_version"] == manifest.model_version
    assert lineage["artifact_hash"] == manifest.artifact_hash
    assert lineage["run_id"] == manifest.run_id
    assert lineage["git_sha"] == manifest.git_sha
    assert lineage["data_hash"] == manifest.data_hash


def test_health_and_manifest(_env) -> None:
    tracking_uri, result = _env
    client = _client(tracking_uri)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["model_loaded"] is True
    assert health.json()["model_version"] == result.manifest.model_version

    manifest_response = client.get("/manifest")
    assert manifest_response.status_code == 200
    assert manifest_response.json()["artifact_hash"] == result.manifest.artifact_hash


def test_score_rejects_invalid_request(_env) -> None:
    tracking_uri, _ = _env
    client = _client(tracking_uri)

    response = client.post("/score", json={"tenure_years": "not-a-number"})

    assert response.status_code == 422


def test_score_fails_closed_without_production(_env) -> None:
    tracking_uri, _ = _env
    client = _client(tracking_uri, model_name="never-promoted")

    response = client.post("/score", json=_VALID_REQUEST)

    assert response.status_code == 503
