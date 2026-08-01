"""Load and verify the production model from the registry.

The loader is the enforcement point at serving time. It only ever serves the
version the ``production`` alias points at, and only after proving the bytes in
the registry match the artifact hash recorded in the lineage manifest. A
background poller hot-swaps to a newly promoted version without a restart.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from pathlib import Path

import cloudpickle
import pandas as pd

from gatedops.manifest.hashing import sha256_file
from gatedops.manifest.schema import ModelManifest
from gatedops.registry.mlflow_ import MlflowRegistry

logger = logging.getLogger(__name__)

_MANIFEST_TAG = "gatedops.manifest"
_THRESHOLD_PARAM = "threshold"
_DEFAULT_THRESHOLD = 0.6


class ModelNotLoadedError(RuntimeError):
    """Raised when there is no production model to serve yet."""


class ModelVerificationError(RuntimeError):
    """Raised when a model fails its lineage or artifact-hash verification."""


class ModelLoader:
    """Resolves, verifies, caches, and hot-swaps the production model."""

    def __init__(
        self,
        registry: MlflowRegistry,
        model_name: str,
        poll_seconds: float = 15.0,
    ) -> None:
        self._registry = registry
        self._model_name = model_name
        self._poll_seconds = poll_seconds
        self._lock = threading.Lock()
        self._model = None
        self._threshold = _DEFAULT_THRESHOLD
        self._manifest: ModelManifest | None = None
        self._loaded_at: datetime | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="gatedops-poller",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # -- loading -----------------------------------------------------------

    def load(self) -> None:
        """Resolve the production version, verify it, and swap it in."""
        version = self._registry.current_production(self._model_name)
        if version is None:
            with self._lock:
                self._model = None
                self._manifest = None
            raise ModelNotLoadedError(
                f"no production version registered for {self._model_name!r}"
            )

        version_str = str(version.version)
        manifest = self._read_manifest(version_str)
        artifact = self._registry.version_artifact(self._model_name, version_str)
        self._verify(artifact, manifest)

        model = cloudpickle.loads(artifact.read_bytes())
        params = self._registry.params_for(self._model_name, version_str)
        threshold = float(params.get(_THRESHOLD_PARAM, _DEFAULT_THRESHOLD))

        with self._lock:
            self._model = model
            self._threshold = threshold
            self._manifest = manifest
            self._loaded_at = datetime.now(UTC)
        logger.info(
            "serving %s v%s (artifact %s)",
            self._model_name,
            version_str,
            manifest.artifact_hash[:12],
        )

    def _read_manifest(self, model_version: str) -> ModelManifest:
        raw = self._registry.model_version_tag(self._model_name, model_version, _MANIFEST_TAG)
        if raw is None:
            raise ModelVerificationError(
                f"no {_MANIFEST_TAG} tag on {self._model_name} v{model_version}"
            )
        return ModelManifest.model_validate_json(raw)

    def _verify(self, artifact: Path, manifest: ModelManifest) -> None:
        actual = sha256_file(artifact)
        if actual != manifest.artifact_hash:
            raise ModelVerificationError(
                f"artifact hash mismatch for {self._model_name} v{manifest.model_version}: "
                f"manifest {manifest.artifact_hash}, registry {actual}"
            )

    # -- polling -----------------------------------------------------------

    def _poll_loop(self) -> None:
        while not self._stop.wait(self._poll_seconds):
            try:
                if self.reload_if_changed():
                    logger.info("production alias changed; reloaded model")
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("model reload poll failed: %s", exc)

    def reload_if_changed(self) -> bool:
        """Reload when the production alias moved, returning whether it did."""
        version = self._registry.current_production(self._model_name)
        current = None if version is None else str(version.version)
        with self._lock:
            loaded = None if self._manifest is None else self._manifest.model_version
        if current == loaded:
            return False
        try:
            self.load()
        except (ModelNotLoadedError, ModelVerificationError) as exc:
            logger.warning("could not reload production model: %s", exc)
        return True

    # -- serving -----------------------------------------------------------

    def predict(self, features: dict[str, float]) -> tuple[int, float]:
        """Score one feature vector, returning ``(prediction, probability)``."""
        with self._lock:
            model = self._model
            threshold = self._threshold
        if model is None:
            raise ModelNotLoadedError(
                f"no production model loaded for {self._model_name!r}"
            )

        frame = pd.DataFrame([features])
        names = getattr(model, "feature_names_in_", None)
        if names is not None:
            frame = frame[list(names)]
        probability = float(model.predict_proba(frame)[0, 1])
        prediction = int(probability >= threshold)
        return prediction, probability

    # -- introspection -----------------------------------------------------

    @property
    def manifest(self) -> ModelManifest | None:
        with self._lock:
            return self._manifest

    def status(self) -> dict[str, object]:
        with self._lock:
            manifest = self._manifest
            loaded_at = self._loaded_at
        if manifest is None:
            return {
                "model_loaded": False,
                "model_version": None,
                "artifact_hash": None,
                "loaded_at": None,
            }
        return {
            "model_loaded": True,
            "model_version": manifest.model_version,
            "artifact_hash": manifest.artifact_hash,
            "loaded_at": loaded_at.isoformat() if loaded_at is not None else None,
        }
