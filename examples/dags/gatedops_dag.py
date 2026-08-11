"""Daily gated release schedule for GatedOps.

One task: train, evaluate, gate, register and -- on a passing gate -- promote
the model to production. A rejected gate fails the task via
``AirflowFailException`` so the run shows up as failed and alerts fire.
"""

from datetime import datetime

from airflow import DAG

from gatedops.airflow.operator import GatedOpsOperator

with DAG(
    "gatedops_dag",
    default_args={"retries": 0},
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "gated-release"],
) as dag:
    # The config path must point at the deployment's pipeline.yaml.
    gated_release = GatedOpsOperator(
        task_id="gatedops_train_gate_promote",
        config_path="/path/to/configs/pipeline.yaml",
    )
