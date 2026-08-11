"""Prefect 3 flow for the scheduled gated release loop.

The pipeline itself runs with ``promote_on_pass=False``: train, evaluate and
the gate all happen inside the ``_train`` task, but promotion is owned by the
flow. Only a run whose gate report reads ``PASS`` is handed to ``_promote``,
so a failed gate can never reach production through this schedule.
"""

from __future__ import annotations

from mlflow.tracking import MlflowClient
from prefect import flow, task

from gatedops.config import RunConfig, resolve_tracking_uri
from gatedops.pipelines.run import PipelineResult, run_pipeline
from gatedops.promote.promote import promote
from gatedops.registry.mlflow_ import MlflowRegistry

_MANIFEST_TAG = "gatedops.manifest"


@task
def _train(
    config: RunConfig,
    tracking_uri: str,
    signal_strength: float | None,
    seed: int | None,
) -> PipelineResult:
    """Train, evaluate and gate a model without promoting it."""
    return run_pipeline(
        config,
        tracking_uri=tracking_uri,
        signal_strength=signal_strength,
        seed=seed,
        promote_on_pass=False,
    )


@task
def _promote(result: PipelineResult, tracking_uri: str) -> dict:
    """Promote a gated, registered model and record it in the lineage.

    ``run_pipeline`` runs with ``promote_on_pass=False``, so the model version
    tag still says ``Staging`` after this task flips the production alias. The
    manifest is re-tagged as ``Production`` here so the audit trail stays
    accurate.
    """
    registry = MlflowRegistry(tracking_uri=tracking_uri)
    receipt = promote(result.manifest, registry)

    manifest = result.manifest.model_copy(update={"promote_stage": "Production"})
    client = MlflowClient(tracking_uri=tracking_uri)
    client.set_model_version_tag(
        result.manifest.model_name,
        result.manifest.model_version,
        _MANIFEST_TAG,
        manifest.model_dump_json(),
    )
    return {
        "from_stage": receipt.from_stage,
        "to_stage": receipt.to_stage,
        "model_version": receipt.model_version,
        "artifact_hash": receipt.artifact_hash,
    }


@flow(name="gatedops-gated-release")
def gated_flow(
    config: RunConfig,
    tracking_uri: str | None = None,
    signal_strength: float | None = None,
    seed: int | None = None,
) -> dict:
    """Run the gated release loop: promote only when the gate passed.

    Example
    -------
    Serve the flow as a daily cron deployment (Prefect 3.8+)::

        from gatedops.config import RunConfig

        gated_flow.serve(
            name="gatedops-daily",
            cron="0 3 * * *",
            parameters={"config": RunConfig(model_name="churn")},
        )
    """
    resolved = resolve_tracking_uri(tracking_uri or config.tracking_uri)
    result = _train(
        config,
        tracking_uri=resolved,
        signal_strength=signal_strength,
        seed=seed,
    )

    promoted = False
    if result.gate.status == "PASS":
        _promote(result, tracking_uri=resolved)
        promoted = True

    return {
        "run_id": result.run_id,
        "model_version": result.model_version,
        "gate_status": result.gate.status,
        "promoted": promoted,
    }
