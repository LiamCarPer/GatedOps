"""Unit tests for the promotion guard: gate pass and artifact-hash verification."""

from pathlib import Path

import pytest

from gatedops.gate.engine import evaluate_gate
from gatedops.gate.rules import GateConfig, ThresholdRule
from gatedops.manifest.builder import build_manifest
from gatedops.manifest.schema import ModelManifest
from gatedops.promote.promote import (
    PromoteBlockedError,
    PromoteReceipt,
    promote,
)


class FakeRegistry:
    """In-memory registry recording production promotions."""

    def __init__(self, artifact: Path) -> None:
        self._artifact = artifact
        self.calls: list[tuple[str, str]] = []

    def version_artifact(self, model_name: str, model_version: str) -> Path:
        return self._artifact

    def set_production(self, model_name: str, model_version: str) -> None:
        self.calls.append((model_name, model_version))


def _passed_manifest(tmp_path: Path) -> tuple[ModelManifest, FakeRegistry]:
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"good-model")
    report = evaluate_gate(
        GateConfig(thresholds=[ThresholdRule(metric="f1", op=">=", value=0.75)]),
        {"f1": 0.9},
        model_name="churn",
    )
    manifest = build_manifest(
        model_name="churn",
        model_version="v2",
        artifact=artifact,
        run_id="r",
        data_hash="d",
        metrics={"f1": 0.9},
        gate=report,
        promote_stage="Staging",
    )
    return manifest, FakeRegistry(artifact)


def test_promote_rejects_failed_gate(tmp_path: Path) -> None:
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"bad-model")
    report = evaluate_gate(
        GateConfig(thresholds=[ThresholdRule(metric="f1", op=">=", value=0.75)]),
        {"f1": 0.3},
        model_name="churn",
    )
    manifest = build_manifest(
        model_name="churn",
        model_version="v1",
        artifact=artifact,
        run_id="r",
        data_hash="d",
        metrics={"f1": 0.3},
        gate=report,
    )

    with pytest.raises(PromoteBlockedError, match="gate status"):
        promote(manifest, FakeRegistry(artifact))


def test_promote_rejects_missing_gate(tmp_path: Path) -> None:
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"x")
    manifest = build_manifest(
        model_name="churn",
        model_version="v1",
        artifact=artifact,
        run_id="r",
        data_hash="d",
        metrics={},
    )

    with pytest.raises(PromoteBlockedError, match="no gate report"):
        promote(manifest, FakeRegistry(artifact))


def test_promote_rejects_artifact_tampering(tmp_path: Path) -> None:
    manifest, registry = _passed_manifest(tmp_path)
    tampered = tmp_path / "tampered.pkl"
    tampered.write_bytes(b"good-model-mutated")
    registry._artifact = tampered

    with pytest.raises(PromoteBlockedError, match="hash mismatch"):
        promote(manifest, registry)


def test_promote_success(tmp_path: Path) -> None:
    manifest, registry = _passed_manifest(tmp_path)

    receipt = promote(manifest, registry)

    assert isinstance(receipt, PromoteReceipt)
    assert receipt.to_stage == "Production"
    assert receipt.from_stage == "Staging"
    assert registry.calls == [("churn", "v2")]
