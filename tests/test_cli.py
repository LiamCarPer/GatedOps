"""Unit tests for CLI helpers and default configuration resolution."""

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
