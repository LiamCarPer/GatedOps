"""Integration tests for the MLflow registry adapter."""

import cloudpickle
import mlflow
from sklearn.linear_model import LogisticRegression

from gatedops.registry.mlflow_ import MlflowRegistry


def test_registry_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.set_experiment("registry-test").experiment_id

    model = LogisticRegression().fit([[0.0], [1.0]], [0, 1])
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(cloudpickle.dumps(model))

    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        mlflow.log_metrics({"f1": 0.9})
        mlflow.log_artifact(artifact, artifact_path="model")

    registry = MlflowRegistry(tracking_uri=tracking_uri)

    version = registry.register_version("churn", run_id)
    assert version == "1"

    downloaded = registry.version_artifact("churn", version)
    assert downloaded.name == "model.pkl"
    assert downloaded.read_bytes() == artifact.read_bytes()

    registry.set_production("churn", version)

    production = registry.current_production("churn")
    assert production is not None
    assert str(production.version) == version

    assert registry.metrics_for("churn", version)["f1"] == 0.9


def test_find_model_file_prefers_known_names(tmp_path) -> None:
    artifact_dir = tmp_path / "model"
    artifact_dir.mkdir()
    (artifact_dir / "MLmodel").write_text("{}")
    (artifact_dir / "requirements.txt").write_text("")
    skops = artifact_dir / "model.skops"
    skops.write_bytes(b"skops")

    assert MlflowRegistry._find_model_file(artifact_dir) == skops


def test_find_model_file_falls_back_to_serialized_file(tmp_path) -> None:
    artifact_dir = tmp_path / "model"
    artifact_dir.mkdir()
    (artifact_dir / "MLmodel").write_text("{}")
    (artifact_dir / "conda.yaml").write_text("")
    binary = artifact_dir / "custom.bin"
    binary.write_bytes(b"bytes")

    assert MlflowRegistry._find_model_file(artifact_dir) == binary


def test_find_model_file_missing_returns_none(tmp_path) -> None:
    artifact_dir = tmp_path / "model"
    artifact_dir.mkdir()
    (artifact_dir / "MLmodel").write_text("{}")

    assert MlflowRegistry._find_model_file(artifact_dir) is None
