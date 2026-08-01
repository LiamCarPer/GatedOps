"""Unit tests for the model loader: verification, caching, and hot reload."""

from pathlib import Path
from types import SimpleNamespace

import cloudpickle
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gatedops.gate.engine import evaluate_gate
from gatedops.gate.rules import GateConfig, ThresholdRule
from gatedops.manifest.builder import build_manifest
from gatedops.manifest.schema import ModelManifest
from gatedops.serve.loader import (
    ModelLoader,
    ModelNotLoadedError,
    ModelVerificationError,
)


def _make_model(feature_order: list[str] | None = None) -> object:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 7))
    y = (x[:, 0] + x[:, 1] > 0.0).astype(int)
    if feature_order is not None:
        frame = pd.DataFrame(x, columns=feature_order)
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(frame, y)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(x, y)


def _write_artifact(tmp_path: Path, model: object) -> Path:
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(cloudpickle.dumps(model))
    return artifact


def _manifest_json(artifact: Path, version: str = "3") -> str:
    report = evaluate_gate(
        GateConfig(thresholds=[ThresholdRule(metric="f1", op=">=", value=0.7)]),
        {"f1": 0.9},
        model_name="churn",
    )
    manifest = build_manifest(
        model_name="churn",
        model_version=version,
        artifact=artifact,
        run_id="run-1",
        data_hash="d1",
        metrics={"f1": 0.9},
        gate=report,
    )
    return manifest.model_dump_json()


class FakeRegistry:
    """Minimal in-memory registry with the surface the loader uses."""

    def __init__(
        self,
        artifact: Path,
        manifest_json: str,
        params: dict[str, str],
        version: str = "3",
    ) -> None:
        self._artifact = artifact
        self._manifest = manifest_json
        self._params = params
        self.production_version = version

    def current_production(self, model_name: str) -> SimpleNamespace | None:
        if self.production_version is None:
            return None
        return SimpleNamespace(version=self.production_version)

    def version_artifact(self, model_name: str, model_version: str) -> Path:
        return self._artifact

    def params_for(self, model_name: str, model_version: str) -> dict[str, str]:
        return self._params

    def model_version_tag(self, model_name: str, model_version: str, key: str) -> str | None:
        return self._manifest


def _loader(tmp_path: Path, registry: FakeRegistry) -> ModelLoader:
    return ModelLoader(registry, "churn", poll_seconds=3600)


def test_load_and_predict(tmp_path: Path) -> None:
    model = _make_model()
    artifact = _write_artifact(tmp_path, model)
    manifest = _manifest_json(artifact)
    registry = FakeRegistry(artifact, manifest, {"threshold": "0.6"})

    loader = _loader(tmp_path, registry)
    loader.load()

    assert loader.manifest is not None
    assert isinstance(loader.manifest, ModelManifest)
    assert loader.status()["model_loaded"] is True

    prediction, probability = loader.predict(
        {"f1": 0.5, "f2": 0.5, "f3": 0.5, "f4": 0.5, "f5": 0.5, "f6": 0.5, "f7": 0.5}
    )
    assert prediction in (0, 1)
    assert 0.0 <= probability <= 1.0


def test_load_fails_without_production(tmp_path: Path) -> None:
    registry = FakeRegistry(tmp_path / "x.pkl", "{}", {}, version=None)

    loader = _loader(tmp_path, registry)
    with pytest.raises(ModelNotLoadedError):
        loader.load()


def test_load_rejects_tampered_artifact(tmp_path: Path) -> None:
    model = _make_model()
    artifact = _write_artifact(tmp_path, model)
    manifest = _manifest_json(artifact)

    artifact.write_bytes(artifact.read_bytes() + b"\x00")

    registry = FakeRegistry(artifact, manifest, {"threshold": "0.6"})
    loader = _loader(tmp_path, registry)
    with pytest.raises(ModelVerificationError, match="hash mismatch"):
        loader.load()


def test_load_rejects_missing_manifest_tag(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path, _make_model())
    registry = FakeRegistry(artifact, None, {"threshold": "0.6"})

    loader = _loader(tmp_path, registry)
    with pytest.raises(ModelVerificationError, match="no gatedops.manifest tag"):
        loader.load()


def test_predict_reorders_features(tmp_path: Path) -> None:
    order = [
        "tenure_years",
        "monthly_spend",
        "support_tickets",
        "usage_frequency",
        "engagement_score",
        "has_contract",
        "payment_delay",
    ]
    model = _make_model(feature_order=order)
    artifact = _write_artifact(tmp_path, model)
    registry = FakeRegistry(artifact, _manifest_json(artifact), {"threshold": "0.6"})

    loader = _loader(tmp_path, registry)
    loader.load()

    features = {name: 0.5 for name in order}
    reordered = {name: features[name] for name in reversed(order)}
    assert loader.predict(features) == loader.predict(reordered)


def test_reload_swaps_on_promotion(tmp_path: Path) -> None:
    model = _make_model()
    artifact = _write_artifact(tmp_path, model)
    registry = FakeRegistry(artifact, _manifest_json(artifact), {"threshold": "0.6"})

    loader = _loader(tmp_path, registry)
    loader.load()
    assert loader.manifest.model_version == "3"
    assert loader.reload_if_changed() is False

    new_artifact = _write_artifact(tmp_path, _make_model())
    registry.production_version = "4"
    registry._manifest = _manifest_json(new_artifact, version="4")

    assert loader.reload_if_changed() is True
    assert loader.manifest.model_version == "4"


def test_threshold_comes_from_registry_params(tmp_path: Path) -> None:
    model = _make_model()
    artifact = _write_artifact(tmp_path, model)
    manifest = _manifest_json(artifact)
    features = {f"f{i}": 0.5 for i in range(1, 8)}

    low = _loader(tmp_path, FakeRegistry(artifact, manifest, {"threshold": "0.0"}))
    low.load()
    high = _loader(tmp_path, FakeRegistry(artifact, manifest, {"threshold": "1.0"}))
    high.load()

    prediction_low, _ = low.predict(features)
    prediction_high, _ = high.predict(features)
    assert prediction_low == 1
    assert prediction_high == 0
