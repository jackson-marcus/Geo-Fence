"""API routes: /stores, /score-site, /best-sites, /heatmap, /health."""

from __future__ import annotations

import functools
import logging

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from geofence.models.huff import best_sites, score_site, store_summary
from geofence.settings import get_config, resolve_path

logger = logging.getLogger(__name__)
router = APIRouter()


class SiteRequest(BaseModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    size_sqm: float = Field(gt=0, le=20000, default=1200)


@functools.lru_cache(maxsize=1)
def _city():
    processed = resolve_path(get_config()["data"]["processed_dir"])
    pop_path, stores_path = processed / "population.npy", processed / "stores.parquet"
    if not (pop_path.exists() and stores_path.exists()):
        raise FileNotFoundError("No city; run scripts/make_city.py")
    return np.load(pop_path), pd.read_parquet(stores_path)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/stores")
def stores() -> list[dict]:
    try:
        population, store_df = _city()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return store_summary(population, store_df).to_dict(orient="records")


@router.get("/heatmap")
def heatmap() -> dict:
    try:
        population, store_df = _city()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "population": population.tolist(),
        "stores": store_df.to_dict(orient="records"),
    }


@router.post("/score-site")
def score(request: SiteRequest) -> dict:
    try:
        population, store_df = _city()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    grid = population.shape[0]
    if not (0 <= request.x < grid and 0 <= request.y < grid):
        raise HTTPException(status_code=422, detail=f"coordinates outside {grid}x{grid} grid")
    return score_site(population, store_df, request.x, request.y, request.size_sqm)


@router.get("/best-sites")
def best(top_k: int = 8) -> list[dict]:
    try:
        population, store_df = _city()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return best_sites(population, store_df, top_k=top_k)
