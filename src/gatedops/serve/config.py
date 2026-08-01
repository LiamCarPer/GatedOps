"""Runtime configuration for the scoring service."""

from pydantic_settings import BaseSettings, SettingsConfigDict

from gatedops.config import default_tracking_uri


class ServeConfig(BaseSettings):
    model_name: str = "churn-classifier"
    tracking_uri: str = default_tracking_uri()
    poll_seconds: float = 15.0
    autoreload: bool = True

    model_config = SettingsConfigDict(env_prefix="GATEDOPS_", extra="ignore")
