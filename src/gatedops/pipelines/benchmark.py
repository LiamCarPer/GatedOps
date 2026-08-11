"""ONNX export and serving-latency benchmarking for trained pipelines.

The serving-quality gate needs numbers the runtime can actually hit, not
training-time estimates. This module converts a fitted sklearn ``Pipeline`` to
an ONNX graph with ``skl2onnx``, then measures single-row inference latency and
throughput with ``onnxruntime`` so the results can flow through the same
gate/manifest/lineage machinery as the classification metrics.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Stable, widely supported opset; onnxruntime 1.28 handles it without warning.
_TARGET_OPSET = 15


def export_to_onnx(
    pipeline,
    sample_features: NDArray[np.float64],
    path: Path,
) -> Path:
    """Convert a fitted sklearn pipeline to a serialized ONNX model.

    Args:
        pipeline: Fitted sklearn estimator exposing ``predict_proba`` (e.g. the
            ``StandardScaler`` + ``LogisticRegression`` pipeline of GatedOps).
        sample_features: Array of shape ``(n_features,)`` or ``(1, n_features)``
            used to determine the width of the model input.
        path: Destination path for the ``.onnx`` file.

    Returns:
        The ``path`` the model was written to.
    """
    sample = np.asarray(sample_features, dtype=np.float64)
    initial_types = [("features", FloatTensorType([None, sample.shape[1]]))]
    model_onnx = convert_sklearn(
        pipeline,
        initial_types=initial_types,
        target_opset=_TARGET_OPSET,
        # Classifiers default to a ZipMap output (a list of per-row dicts);
        # disable it so the probability output stays a (N, C) float matrix.
        options={id(pipeline): {"zipmap": False}},
    )
    path = Path(path)
    path.write_bytes(model_onnx.SerializeToString())
    return path


def benchmark_onnx(
    onnx_path: Path,
    features: NDArray[np.float32],
    n_warmup: int = 10,
    n_iterations: int = 100,
) -> dict[str, float]:
    """Benchmark single-row inference latency of an ONNX model.

    Args:
        onnx_path: Path to a serialized ONNX model with a float32 ``features``
            input and a probability-matrix output.
        features: Array of shape ``(N, n_features)`` of candidate rows; every
            row is scored as a separate single-row inference.
        n_warmup: Number of single-row inferences to run before timing begins
            (lets onnxruntime initialize its providers).
        n_iterations: Number of single-row inferences to time.

    Returns:
        Latency percentiles per row in milliseconds and aggregate throughput:
        keys ``latency_p50_ms``, ``latency_p95_ms``, ``latency_p99_ms`` and
        ``throughput_rps`` (``n_iterations`` divided by total timed seconds).
    """
    import onnxruntime as ort

    rows = np.asarray(features, dtype=np.float32)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def _infer(row: NDArray[np.float32]) -> None:
        session.run(None, {input_name: row[None, :]})

    for row in rows[:n_warmup]:
        _infer(row)

    timings_ms: list[float] = []
    for index in range(n_iterations):
        row = rows[index % rows.shape[0]]
        start = time.perf_counter()
        _infer(row)
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    total_seconds = sum(timings_ms) / 1000.0
    return {
        "latency_p50_ms": float(np.percentile(timings_ms, 50.0)),
        "latency_p95_ms": float(np.percentile(timings_ms, 95.0)),
        "latency_p99_ms": float(np.percentile(timings_ms, 99.0)),
        "throughput_rps": float(
            n_iterations / total_seconds if total_seconds > 0.0 else 0.0
        ),
    }
