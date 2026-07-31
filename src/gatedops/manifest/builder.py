"""Build a ``ModelManifest`` from the facts of a training run."""

import subprocess
from pathlib import Path

from gatedops.gate.report import GateReport
from gatedops.manifest.hashing import sha256_file
from gatedops.manifest.schema import ModelManifest, PromoteStage


def git_rev() -> str:
    """Short HEAD SHA, suffixed ``-dirty`` when the working tree has changes."""
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return f"{sha}-dirty" if dirty else sha


def build_manifest(
    *,
    model_name: str,
    model_version: str,
    artifact: Path,
    run_id: str,
    data_hash: str,
    metrics: dict[str, float],
    gate: GateReport | None = None,
    promote_stage: PromoteStage = "None",
) -> ModelManifest:
    """Assemble a manifest, hashing the model artifact in the process."""
    return ModelManifest(
        model_name=model_name,
        model_version=model_version,
        artifact_hash=sha256_file(artifact),
        run_id=run_id,
        git_sha=git_rev(),
        data_hash=data_hash,
        metrics=metrics,
        gate=gate,
        promote_stage=promote_stage,
    )
