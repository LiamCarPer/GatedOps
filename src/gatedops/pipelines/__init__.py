"""The end-to-end training pipeline: train, evaluate, gate, register, promote."""

from gatedops.pipelines.run import PipelineResult, run_pipeline

__all__ = ["PipelineResult", "run_pipeline"]

try:
    # The Prefect flow lives behind the optional ``orchestration`` extra; the
    # base package must import cleanly without it.
    from gatedops.pipelines.prefect_flow import gated_flow

    __all__ = ["PipelineResult", "gated_flow", "run_pipeline"]
except ImportError:  # pragma: no cover - prefect extra not installed
    pass
