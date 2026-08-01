"""Enforce the gate and the artifact hash before a model reaches production."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from gatedops.manifest.hashing import sha256_file
from gatedops.manifest.schema import ModelManifest, PromoteStage


class PromoteBlockedError(RuntimeError):
    """Raised when a model is not allowed to be promoted to production."""


class ModelRegistry(Protocol):
    """Minimal interface a model registry must expose for promotion."""

    def version_artifact(self, model_name: str, model_version: str) -> Path:
        """Return a local copy of the model artifact for a registered version."""
        ...

    def set_production(self, model_name: str, model_version: str) -> None:
        """Point the production alias of ``model_name`` at ``model_version``."""
        ...


@dataclass(frozen=True)
class PromoteReceipt:
    """Proof that a promotion happened, for logging and lineage."""

    model_name: str
    model_version: str
    from_stage: PromoteStage
    to_stage: PromoteStage
    artifact_hash: str
    gate_status: str
    promoted_at: str


def promote(manifest: ModelManifest, registry: ModelRegistry) -> PromoteReceipt:
    """Promote ``manifest`` to production, or refuse to.

    Refusal reasons are explicit so failures are auditable:

    * the manifest carries no gate report, or its gate did not pass;
    * the bytes the registry holds for the version differ from the artifact
      hash recorded at training time.
    """
    if manifest.gate is None:
        raise PromoteBlockedError("cannot promote: no gate report attached to the manifest")

    if manifest.gate.status != "PASS":
        raise PromoteBlockedError(
            f"cannot promote: gate status is {manifest.gate.status!r}"
        )

    artifact = registry.version_artifact(manifest.model_name, manifest.model_version)
    actual_hash = sha256_file(artifact)
    if actual_hash != manifest.artifact_hash:
        raise PromoteBlockedError(
            "cannot promote: artifact hash mismatch "
            f"(manifest {manifest.artifact_hash}, registry {actual_hash})"
        )

    registry.set_production(manifest.model_name, manifest.model_version)
    return PromoteReceipt(
        model_name=manifest.model_name,
        model_version=manifest.model_version,
        from_stage=manifest.promote_stage,
        to_stage="Production",
        artifact_hash=actual_hash,
        gate_status=manifest.gate.status,
        promoted_at=manifest.created_at.isoformat(),
    )
