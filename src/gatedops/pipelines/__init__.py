"""The end-to-end training pipeline: train, evaluate, gate, register, promote."""

from gatedops.pipelines.run import PipelineResult, run_pipeline

__all__ = ["PipelineResult", "run_pipeline"]
