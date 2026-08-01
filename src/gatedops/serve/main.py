"""Module-level application for uvicorn.

Run with: ``uvicorn gatedops.serve.main:app``
"""

from gatedops.serve.app import create_app
from gatedops.serve.config import ServeConfig

app = create_app(ServeConfig())
