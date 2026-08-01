"""FastAPI application exposing the gated, lineage-annotated scoring API."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, status

from gatedops.config import resolve_tracking_uri
from gatedops.registry.mlflow_ import MlflowRegistry
from gatedops.serve.config import ServeConfig
from gatedops.serve.loader import (
    ModelLoader,
    ModelNotLoadedError,
    ModelVerificationError,
)
from gatedops.serve.schemas import Lineage, ScoreRequest, ScoreResponse

logger = logging.getLogger(__name__)


def create_app(config: ServeConfig) -> FastAPI:
    """Build the scoring application around a hash-verified model loader."""
    tracking_uri = resolve_tracking_uri(config.tracking_uri)
    registry = MlflowRegistry(tracking_uri=tracking_uri)
    loader = ModelLoader(registry, config.model_name, poll_seconds=config.poll_seconds)

    if config.autoreload:
        loader.start()

    try:
        loader.load()
    except (ModelNotLoadedError, ModelVerificationError) as exc:
        logger.warning("scoring API started without a production model: %s", exc)

    app = FastAPI(title="GatedOps Scoring API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, object]:
        current = loader.status()
        return {"status": "ok" if current["model_loaded"] else "degraded", **current}

    @app.post("/score", response_model=ScoreResponse)
    def score(request: ScoreRequest) -> ScoreResponse:
        try:
            prediction, probability = loader.predict(request.model_dump())
        except ModelNotLoadedError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

        manifest = loader.manifest
        if manifest is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "no production model loaded")
        return ScoreResponse(
            prediction=prediction,
            probability=probability,
            lineage=Lineage(
                model_name=manifest.model_name,
                model_version=manifest.model_version,
                artifact_hash=manifest.artifact_hash,
                git_sha=manifest.git_sha,
                run_id=manifest.run_id,
                data_hash=manifest.data_hash,
            ),
        )

    @app.get("/manifest")
    def get_manifest() -> dict[str, object]:
        manifest = loader.manifest
        if manifest is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "no production model loaded")
        return manifest.model_dump(mode="json")

    @app.post("/reload")
    def reload() -> dict[str, str]:
        try:
            loader.load()
        except (ModelNotLoadedError, ModelVerificationError) as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
        return {"status": "ok"}

    return app
