"""Tests for ONNX export and serving-latency benchmarking."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gatedops.pipelines.benchmark import benchmark_onnx, export_to_onnx

_N_FEATURES = 7

_BENCHMARK_KEYS = {
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "throughput_rps",
}


def _fitted_pipeline() -> object:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(500, _N_FEATURES))
    y = (x[:, 0] + x[:, 2] - x[:, 5] > 0.0).astype(int)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(x, y)


def test_export_to_onnx_writes_file(tmp_path) -> None:
    model = _fitted_pipeline()
    sample = np.random.default_rng(1).normal(size=(1, _N_FEATURES))

    onnx_path = export_to_onnx(model, sample, tmp_path / "model.onnx")

    assert onnx_path.is_file()
    assert onnx_path.suffix == ".onnx"
    assert onnx_path.read_bytes().startswith(b"\x08")  # protobuf header


def test_benchmark_onnx_returns_positive_metrics(tmp_path) -> None:
    model = _fitted_pipeline()
    x_test = np.random.default_rng(2).normal(size=(200, _N_FEATURES))
    onnx_path = export_to_onnx(model, x_test, tmp_path / "model.onnx")

    results = benchmark_onnx(
        onnx_path, x_test.astype(np.float32), n_warmup=5, n_iterations=50
    )

    assert set(results) == _BENCHMARK_KEYS
    assert all(value > 0.0 for value in results.values())


def test_benchmark_latency_percentiles_are_ordered(tmp_path) -> None:
    model = _fitted_pipeline()
    x_test = np.random.default_rng(3).normal(size=(200, _N_FEATURES))
    onnx_path = export_to_onnx(model, x_test, tmp_path / "model.onnx")

    results = benchmark_onnx(
        onnx_path, x_test.astype(np.float32), n_warmup=5, n_iterations=50
    )

    assert results["latency_p50_ms"] <= results["latency_p95_ms"] + 1e-9
    assert results["latency_p95_ms"] <= results["latency_p99_ms"] + 1e-9
