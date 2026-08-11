"""Airflow integration: run the gated release loop from a DAG."""

from gatedops.airflow.operator import GatedOpsOperator

__all__ = ["GatedOpsOperator"]
