"""Production serving layer: a hash-verified, alias-served scoring API."""

from gatedops.serve.app import create_app
from gatedops.serve.config import ServeConfig
from gatedops.serve.loader import (
    ModelLoader,
    ModelNotLoadedError,
    ModelVerificationError,
)
from gatedops.serve.schemas import ScoreRequest, ScoreResponse

__all__ = [
    "ModelLoader",
    "ModelNotLoadedError",
    "ModelVerificationError",
    "ScoreRequest",
    "ScoreResponse",
    "ServeConfig",
    "create_app",
]
