"""End-to-end pipeline: train, evaluate, gate, register, promote.

The canonical loop of the project. A single function that, given a
``RunConfig``, produces a trained, evaluated, gated, registered model and -- if
the gate passes -- a promotion to production, along with a lineage manifest
that ties every step together.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cloudpickle
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from gatedops.config import RunConfig, resolve_tracking_uri
from gatedops.data.synthetic import data_hash, generate_churn
from gatedops.eval.metrics import classification_metrics
from gatedops.gate.engine import evaluate_gate
from gatedops.gate.report import GateReport
from gatedops.manifest.builder import build_manifest
from gatedops.manifest.schema import ModelManifest
from gatedops.promote.promote import PromoteReceipt, promote
from gatedops.registry.mlflow_ import MlflowRegistry


@dataclass(frozen=True)
class PipelineResult:
    run_id: str
    model_version: str
    metrics: dict[str, float]
    gate: GateReport
    manifest: ModelManifest
    promoted: bool
    receipt: PromoteReceipt | None


def _build_model(config: RunConfig):
    if config.model.algorithm == "logistic":
        return LogisticRegression(max_iter=2000, random_state=config.model.random_state)
    raise ValueError(f"unsupported algorithm: {config.model.algorithm!r}")


def run_pipeline(
    config: RunConfig,
    *,
    tracking_uri: str | None = None,
    signal_strength: float | None = None,
    seed: int | None = None,
    promote_on_pass: bool = True,
) -> PipelineResult:
    """Execute the train -> evaluate -> gate -> register -> promote loop."""
    tracking_uri = resolve_tracking_uri(tracking_uri or config.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("gatedops")

    signal = config.data.signal_strength if signal_strength is None else signal_strength
    rseed = config.data.seed if seed is None else seed

    frame = generate_churn(
        n_rows=config.data.n_rows,
        signal_strength=signal,
        seed=rseed,
        target=config.data.target,
    )
    features = frame.drop(columns=[config.data.target])
    labels = frame[config.data.target]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=config.model.test_size,
        random_state=config.model.random_state,
        stratify=labels,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = _build_model(config)
    model.fit(x_train, y_train)
    y_score = model.predict_proba(x_test)[:, 1]
    y_pred = (y_score >= config.model.threshold).astype(int)
    metrics = classification_metrics(y_test, y_pred, y_score)

    with mlflow.start_run(run_name=f"{config.model_name}-seed{rseed}") as run:
        run_id = run.info.run_id
        mlflow.log_params(
            {
                "algorithm": config.model.algorithm,
                "signal_strength": signal,
                "seed": rseed,
                "n_rows": config.data.n_rows,
                "test_size": config.model.test_size,
                "threshold": config.model.threshold,
            }
        )
        mlflow.log_metrics(metrics)

        model_dir = Path("artifacts") / run_id
        model_dir.mkdir(parents=True, exist_ok=True)
        model_file = model_dir / "model.pkl"
        model_file.write_bytes(cloudpickle.dumps(model))
        mlflow.log_artifact(model_file, artifact_path="model")

    client = MlflowClient(tracking_uri=tracking_uri)
    registry = MlflowRegistry(tracking_uri=tracking_uri)
    champion = registry.current_production(config.model_name)
    champion_metrics = (
        registry.metrics_for(config.model_name, champion.version) if champion else None
    )

    gate = evaluate_gate(
        config.gate,
        metrics,
        model_name=config.model_name,
        champion_metrics=champion_metrics,
    )

    version = registry.register_version(config.model_name, run_id)
    client.set_registered_model_alias(config.model_name, "staging", version)

    manifest = build_manifest(
        model_name=config.model_name,
        model_version=version,
        artifact=model_file,
        run_id=run_id,
        data_hash=data_hash(frame),
        metrics=metrics,
        gate=gate,
        promote_stage="Staging",
    )

    receipt: PromoteReceipt | None = None
    promoted = False
    if gate.status == "PASS" and promote_on_pass:
        receipt = promote(manifest, registry)
        promoted = True
        manifest = manifest.model_copy(update={"promote_stage": "Production"})

    client.set_model_version_tag(
        config.model_name, version, "gatedops.manifest", manifest.model_dump_json()
    )
    client.set_model_version_tag(config.model_name, version, "gatedops.status", gate.status)

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    manifest_path = artifacts_dir / f"{run_id}.manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    client.log_artifact(run_id, manifest_path, artifact_path="manifests")

    return PipelineResult(
        run_id=run_id,
        model_version=version,
        metrics=metrics,
        gate=gate,
        manifest=manifest,
        promoted=promoted,
        receipt=receipt,
    )
