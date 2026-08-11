"""Unit tests for CLI helpers and default configuration resolution."""

from types import SimpleNamespace

from gatedops.cli import _default_config
from gatedops.config import load_run_config


def test_default_config_resolves_from_working_directory() -> None:
    config_path = _default_config()

    assert config_path.name == "pipeline.yaml"
    assert config_path.is_file()


def test_default_config_falls_back_to_source_layout(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    config_path = _default_config()

    assert config_path.name == "pipeline.yaml"
    assert config_path.is_file()


def test_default_config_loads_valid_run_config() -> None:
    config = load_run_config(_default_config())

    assert config.model_name == "churn-classifier"
    assert len(config.gate.thresholds) >= 1


def _fake_pipeline_result() -> SimpleNamespace:
    return SimpleNamespace(
        run_id="r",
        model_version="1",
        metrics={},
        gate=SimpleNamespace(status="PASS", summary="", checks=[]),
        receipt=None,
    )


def test_run_benchmark_flag_wires_onnx_and_benchmark(monkeypatch) -> None:
    import gatedops.cli as cli_mod

    captured: dict = {}

    def _fake_run_pipeline(config, **kwargs):
        captured.update(kwargs)
        return _fake_pipeline_result()

    monkeypatch.setattr(cli_mod, "run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr(cli_mod, "_print_result", lambda *args, **kwargs: None)

    assert cli_mod.main(["run", "--benchmark", "--config", str(_default_config())]) == 0

    assert captured["export_onnx"] is True
    assert captured["benchmark"] is True


def test_run_without_benchmark_flag_keeps_defaults(monkeypatch) -> None:
    import gatedops.cli as cli_mod

    captured: dict = {}

    def _fake_run_pipeline(config, **kwargs):
        captured.update(kwargs)
        return _fake_pipeline_result()

    monkeypatch.setattr(cli_mod, "run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr(cli_mod, "_print_result", lambda *args, **kwargs: None)

    assert cli_mod.main(["run", "--config", str(_default_config())]) == 0

    assert captured["export_onnx"] is False
    assert captured["benchmark"] is False
