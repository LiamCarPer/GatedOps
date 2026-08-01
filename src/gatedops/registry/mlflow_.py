"""MLflow-backed implementation of the model registry protocol.

The pipeline stores the model as a plain, byte-exact artifact (``model.pkl``)
rather than an MLflow model flavor. That keeps the artifact hash meaningful:
what the server loads is exactly what the gate saw, with no flavor machinery
between training and serving.
"""

from pathlib import Path

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from gatedops.promote.promote import PromoteBlockedError

_ARTIFACT_DIR = "model"
_ARTIFACT_FILE = "model.pkl"
_PRODUCTION_ALIAS = "production"


class MlflowRegistry:
    """Adapter that lets the promotion policy drive an MLflow registry."""

    def __init__(self, tracking_uri: str | None = None) -> None:
        self._client = MlflowClient(tracking_uri=tracking_uri or mlflow.get_tracking_uri())

    def register_version(self, model_name: str, run_id: str) -> str:
        """Register the ``model`` artifact of ``run_id`` and return the version."""
        try:
            self._client.create_registered_model(model_name)
        except MlflowException as exc:
            if exc.error_code != "RESOURCE_ALREADY_EXISTS":
                raise
        version = self._client.create_model_version(
            name=model_name,
            source=f"runs:/{run_id}/{_ARTIFACT_DIR}",
            run_id=run_id,
        )
        return str(version.version)

    def version_artifact(self, model_name: str, model_version: str) -> Path:
        version = self._client.get_model_version(model_name, model_version)
        run_id = version.run_id or self._require_run_id(model_name, model_version)
        artifact_dir = Path(self._client.download_artifacts(run_id, _ARTIFACT_DIR))
        model_file = artifact_dir / _ARTIFACT_FILE
        if not model_file.is_file():
            raise PromoteBlockedError(
                f"no {_ARTIFACT_FILE} artifact for {model_name} version {model_version}"
            )
        return model_file

    def set_production(self, model_name: str, model_version: str) -> None:
        self._client.set_registered_model_alias(model_name, _PRODUCTION_ALIAS, model_version)

    def current_production(self, model_name: str) -> ModelVersion | None:
        try:
            return self._client.get_model_version_by_alias(model_name, _PRODUCTION_ALIAS)
        except MlflowException:
            return None

    def metrics_for(self, model_name: str, model_version: str) -> dict[str, float]:
        version = self._client.get_model_version(model_name, model_version)
        run_id = version.run_id or self._require_run_id(model_name, model_version)
        run = self._client.get_run(run_id)
        return {name: float(value) for name, value in run.data.metrics.items()}

    @staticmethod
    def _require_run_id(model_name: str, model_version: str) -> str:
        raise PromoteBlockedError(
            f"model version {model_name} v{model_version} has no tracking run"
        )
