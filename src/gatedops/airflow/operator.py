"""Airflow operator that runs a GatedOps gated release inside a DAG.

``GatedOpsOperator`` wraps :func:`gatedops.pipelines.run.run_pipeline` so a
scheduled DAG can train, evaluate, gate and promote a model with the same
policy as the CLI and the Prefect flow. The configuration comes from either an
explicit :class:`gatedops.config.RunConfig` or a ``pipeline.yaml`` file whose
path may be templated (``config_path`` is in ``template_fields``).

The gate verdict is pushed to XCom under the key ``gate_status`` and the full
outcome is returned as ``return_value`` (``run_id``, ``model_version``,
``gate_status``, ``promoted``). By default a rejected release fails the task
with :class:`airflow.exceptions.AirflowFailException`, so downstream alerting
or remediation can hook in through failure callbacks.

Example DAG::

    from airflow import DAG
    from gatedops.airflow.operator import GatedOpsOperator

    with DAG("gatedops_dag", schedule="@daily") as dag:
        GatedOpsOperator(
            task_id="gatedops_train_gate_promote",
            config_path="/path/to/configs/pipeline.yaml",
        )
"""

from __future__ import annotations

from pathlib import Path

from airflow.exceptions import AirflowFailException
from airflow.models import BaseOperator

from gatedops.config import RunConfig, load_run_config
from gatedops.pipelines.run import run_pipeline


class GatedOpsOperator(BaseOperator):
    """Train, evaluate, gate, register and promote a model in one task.

    Args:
        config_path: Path to a ``pipeline.yaml`` that resolves to a
            ``RunConfig``. Mutually exclusive with ``config``.
        config: An already-validated ``RunConfig``. Mutually exclusive with
            ``config_path``.
        tracking_uri: Override for the MLflow tracking URI.
        signal_strength: Override for the data generator's signal strength.
        seed: Override for the data generator's seed.
        promote_on_pass: Whether a passing gate automatically promotes to
            production.
        fail_on_reject: When True, raise ``AirflowFailException`` if the gate
            does not pass.
    """

    template_fields = ("config_path",)

    def __init__(
        self,
        *,
        config_path: str | None = None,
        config: RunConfig | None = None,
        tracking_uri: str | None = None,
        signal_strength: float | None = None,
        seed: int | None = None,
        promote_on_pass: bool = True,
        fail_on_reject: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if config is not None and config_path is not None:
            raise ValueError("pass either config_path or config, not both")
        if config is None and config_path is None:
            raise ValueError("pass either config_path or config")
        self.config_path = config_path
        self.config = config
        self.tracking_uri = tracking_uri
        self.signal_strength = signal_strength
        self.seed = seed
        self.promote_on_pass = promote_on_pass
        self.fail_on_reject = fail_on_reject

    def execute(self, context) -> dict:
        """Run the gated release and report the outcome."""
        if self.config is not None:
            config = self.config
        elif self.config_path is not None:
            config = load_run_config(Path(self.config_path))
        else:
            raise ValueError("GatedOpsOperator requires either config or config_path")

        result = run_pipeline(
            config,
            tracking_uri=self.tracking_uri,
            signal_strength=self.signal_strength,
            seed=self.seed,
            promote_on_pass=self.promote_on_pass,
        )

        ti = context.get("ti")
        if ti is not None:
            ti.xcom_push(key="gate_status", value=result.gate.status)

        if self.fail_on_reject and result.gate.status != "PASS":
            raise AirflowFailException(f"gated release rejected: {result.gate.summary}")

        return {
            "run_id": result.run_id,
            "model_version": result.model_version,
            "gate_status": result.gate.status,
            "promoted": result.promoted,
        }
