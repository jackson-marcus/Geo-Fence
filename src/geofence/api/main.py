"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from geofence import __version__
from geofence.api.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app() -> FastAPI:
    app = FastAPI(
        title="geofence",
        description="Location intelligence: gravity-model store-placement scoring, drive-time trade areas on a synthetic city grid, and cannibalization analysis for new-site decisions.",
        version=__version__,
    )
    app.include_router(router)
    return app


app = create_app()
