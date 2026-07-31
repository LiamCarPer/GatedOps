"""Unit tests for the lineage manifest schema, hashing, and builder."""

from pathlib import Path

from gatedops.gate.engine import evaluate_gate
from gatedops.gate.rules import GateConfig, ThresholdRule
from gatedops.manifest.builder import build_manifest
from gatedops.manifest.hashing import sha256_bytes, sha256_file
from gatedops.manifest.schema import ModelManifest


def _passed_manifest(tmp_path: Path) -> ModelManifest:
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"pickle-bytes")
    report = evaluate_gate(
        GateConfig(thresholds=[ThresholdRule(metric="f1", op=">=", value=0.75)]),
        {"f1": 0.90},
        model_name="churn",
    )
    return build_manifest(
        model_name="churn",
        model_version="v3",
        artifact=artifact,
        run_id="run-abc",
        data_hash="d1",
        metrics={"f1": 0.90},
        gate=report,
    )


def test_sha256_file_matches_bytes(tmp_path: Path) -> None:
    payload = tmp_path / "bin"
    payload.write_bytes(b"abc")

    assert sha256_file(payload) == sha256_bytes(b"abc")


def test_build_manifest_hashes_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"pickle-bytes")
    manifest = _passed_manifest(tmp_path)

    assert manifest.artifact_hash == sha256_bytes(b"pickle-bytes")
    assert manifest.git_sha
    assert manifest.promote_stage == "None"
    assert isinstance(manifest, ModelManifest)


def test_manifest_json_roundtrip(tmp_path: Path) -> None:
    manifest = _passed_manifest(tmp_path)

    restored = ModelManifest.model_validate_json(manifest.model_dump_json())

    assert restored == manifest
