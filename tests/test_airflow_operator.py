"""Airflow operator tests: promote on pass, fail on reject, XCom push.

The tests share a single module-scoped MLflow registry (same hermetic pattern
as ``test_pipeline.py``) and use distinct model names so the pass and fail
scenarios never see each other's registrations. ``GatedOpsOperator`` is
instantiated with an explicit ``task_id`` because ``BaseOperator`` requires it.

On Windows, Airflow 3.x fails to lazily import its models because
``airflow.dag_processing.bundles.base`` imports ``fcntl``, which only exists on
POSIX. Importing Airflow also registers a SQLAlchemy ``before_configured`` hook
that fires while MLflow configures its mappers, so the missing module breaks
MLflow. A tiny ``fcntl`` stub installed before any Airflow import lets the
lazy model loading complete, and forcing ``import_all_models`` up front makes
the hook a no-op by the time MLflow configures mappers.
"""

import os
import sys
import types

try:
    import fcntl
except ImportError:
    fcntl = types.ModuleType("fcntl")

    def _noop(*args, **kwargs):
        return 0

    fcntl.fcntl = _noop
    fcntl.flock = _noop
    fcntl.lockf = _noop
    fcntl.LOCK_SH = 1
    fcntl.LOCK_UN = 8
    fcntl.LOCK_EX = 2
    fcntl.LOCK_NB = 4
    sys.modules["fcntl"] = fcntl

import mlflow
import pytest

pytest.importorskip("airflow")
from airflow.exceptions import AirflowFailException
from airflow.models import import_all_models

from gatedops.airflow.operator import GatedOpsOperator
from gatedops.config import DataConfig, ModelConfig, RunConfig
from gatedops.gate.rules import ChampionRule, GateConfig, ThresholdRule
from gatedops.registry.mlflow_ import MlflowRegistry

import_all_models()


@pytest.fixture(scope="module", autouse=True)
def _registry(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("gatedops")
    os.chdir(workdir)
    tracking_uri = f"sqlite:///{workdir / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    yield tracking_uri


class _FakeTi:
    """Stand-in for the Airflow task instance that records XCom pushes."""

    def __init__(self) -> None:
        self.pushes: list[tuple[str, object]] = []

    def xcom_push(self, key: str, value: object) -> None:
        self.pushes.append((key, value))


def _context() -> dict:
    return {"ti": _FakeTi()}


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


@pytest.mark.airflow
def test_operator_promotes_on_pass(_registry) -> None:
    config = _config(_registry, "op-churn-good")
    operator = GatedOpsOperator(task_id="test-gate", config=config)

    outcome = operator.execute(_context())

    assert outcome["gate_status"] == "PASS"
    assert outcome["promoted"] is True
    assert outcome["run_id"]
    assert outcome["model_version"]

    production = MlflowRegistry(tracking_uri=_registry).current_production("op-churn-good")
    assert production is not None
    assert str(production.version) == outcome["model_version"]


@pytest.mark.airflow
def test_operator_fails_on_reject(_registry) -> None:
    config = _config(_registry, "op-churn-bad")
    config.data.signal_strength = 0.1
    operator = GatedOpsOperator(task_id="test-gate", config=config)

    with pytest.raises(AirflowFailException, match="gated release rejected"):
        operator.execute(_context())

    assert MlflowRegistry(tracking_uri=_registry).current_production("op-churn-bad") is None


@pytest.mark.airflow
def test_operator_pushes_xcom(_registry) -> None:
    config = _config(_registry, "op-churn-xcom")
    operator = GatedOpsOperator(task_id="test-gate", config=config)
    ctx = _context()

    operator.execute(ctx)

    assert ctx["ti"].pushes == [("gate_status", "PASS")]
