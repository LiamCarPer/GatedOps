"""Runtime configuration for a GatedOps pipeline run."""

import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from gatedops.gate.rules import GateConfig

_ENV_VAR = re.compile(r"\$\{([A-Z0-9_]+)\}")


def expand_env(text: str) -> str:
    """Expand ``${VAR}`` placeholders from the environment."""

    def _replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ValueError(f"environment variable {name!r} used in configuration but not set")
        return os.environ[name]

    return _ENV_VAR.sub(_replace, text)


def default_tracking_uri() -> str:
    """A registry location that is not inside a cloud-synced folder.

    Synced folders (OneDrive, Dropbox) hold file locks that make the first
    MLflow database initialization pathologically slow, so the default lives
    under the user's local app-data directory instead of next to the code.
    """
    local = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
    base = Path(local) if local else Path.cwd()
    return f"sqlite:///{base / 'gatedops' / 'mlflow.db'}"


def resolve_tracking_uri(uri: str) -> str:
    """Expand and prepare a tracking URI for use.

    Ensures the parent directory of a sqlite database exists.
    """
    uri = expand_env(uri)
    if uri.startswith("sqlite:///"):
        db_path = Path(uri.removeprefix("sqlite:///"))
        db_path.parent.mkdir(parents=True, exist_ok=True)
    return uri


class DataConfig(BaseModel):
    n_rows: int = 20_000
    signal_strength: float = 0.8
    seed: int = 42
    target: str = "churn"


class ModelConfig(BaseModel):
    algorithm: str = "logistic"
    random_state: int = 42
    test_size: float = 0.2
    threshold: float = 0.6


class RunConfig(BaseModel):
    model_name: str
    tracking_uri: str = Field(default_factory=default_tracking_uri)
    gate: GateConfig = Field(default_factory=GateConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)


def load_run_config(path: str | Path) -> RunConfig:
    """Load a ``pipeline.yaml`` into a validated ``RunConfig``."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if isinstance(raw, dict) and isinstance(raw.get("tracking_uri"), str):
        raw["tracking_uri"] = expand_env(raw["tracking_uri"])
    return RunConfig.model_validate(raw)
