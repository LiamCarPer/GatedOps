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
