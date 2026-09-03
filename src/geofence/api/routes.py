"""API routes: gravity scoring against a live store network plus the network change log."""

from __future__ import annotations

import functools
import logging

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from geofence.models.huff import isochrone, trade_area
from geofence.settings import get_config, resolve_path
from geofence.streams.consumer import LiveNetwork
from geofence.streams.schemas import EVENT_KINDS, InvalidEventError, NetworkEvent
from geofence.workers.processor import plan_rollout

logger = logging.getLogger(__name__)
router = APIRouter()


class SiteRequest(BaseModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    size_sqm: float = Field(gt=0, le=20000, default=1200)


class EventRequest(BaseModel):
    kind: str = Field(description=f"one of {EVENT_KINDS}")
    store_id: int | None = Field(default=None, description="omit on open to auto-assign")
    x: float | None = None
    y: float | None = None
    size_sqm: float | None = None
    note: str = ""


class RolloutRequest(BaseModel):
    n_stores: int = Field(ge=1, le=10, default=3)
    size_sqm: float | None = Field(default=None, gt=0, le=20000)
    commit: bool = False


@functools.lru_cache(maxsize=1)
def _network() -> LiveNetwork:
    cfg = get_config()
    processed = resolve_path(cfg["data"]["processed_dir"])
    pop_path, stores_path = processed / "population.npy", processed / "stores.parquet"
    if not (pop_path.exists() and stores_path.exists()):
        raise FileNotFoundError("No city; run scripts/make_city.py")
    return LiveNetwork(np.load(pop_path), pd.read_parquet(stores_path), cfg["gravity"])


def _live() -> LiveNetwork:
    try:
        return _network()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/stores")
def stores() -> list[dict]:
    net = _live()
    out = net.worker.stores_df()
    demand = net.worker.demand()
    out["captured_demand"] = np.round(demand, 1)
    out["demand_share"] = np.round(demand / net.population.sum(), 4)
    return out.to_dict(orient="records")


@router.get("/heatmap")
def heatmap() -> dict:
    net = _live()
    return {
        "population": net.population.tolist(),
        "stores": net.worker.stores_df().to_dict(orient="records"),
    }


@router.post("/score-site")
def score(request: SiteRequest) -> dict:
    net = _live()
    grid = net.worker.grid
    if not (0 <= request.x < grid and 0 <= request.y < grid):
        raise HTTPException(status_code=422, detail=f"coordinates outside {grid}x{grid} grid")
    return net.worker.score(request.x, request.y, request.size_sqm)


@router.get("/best-sites")
def best(top_k: int = 8) -> list[dict]:
    """Top-k of ONE sweep. Sites here compete for the same uncovered pockets; do not
    read the list as a batch to open together - use /network/rollout for that."""
    placement = get_config()["placement"]
    return _live().worker.best(top_k, placement["candidate_step"], placement["new_store_size"])


@router.get("/trade-area/{store_id}")
def trade(store_id: int, minutes: float = 15.0, threshold: float = 0.5) -> dict:
    net = _live()
    store_df = net.worker.stores_df()
    if store_id not in set(store_df["store_id"]):
        raise HTTPException(status_code=404, detail=f"unknown store_id {store_id}")
    population = net.population
    iso = isochrone(population, store_df, store_id, minutes=minutes)
    capture = trade_area(population, store_df, store_id, threshold=threshold)
    return {
        "store_id": store_id,
        "minutes": minutes,
        "isochrone_cells": int(iso.sum()),
        "isochrone_population": round(float(population[iso].sum()), 1),
        "capture_cells": int(capture.sum()),
        "capture_population": round(float(population[capture].sum()), 1),
    }


@router.get("/network")
def network() -> dict:
    return _live().summary()


@router.get("/network/events")
def network_events(after_seq: int = 0) -> list[dict]:
    """The change log with each event's outcome (applied + effect, or rejected + reason)."""
    net = _live()
    return [net.projection.outcomes[e.seq] for e in net.log.read(after_seq)]


@router.post("/network/events")
def submit_event(request: EventRequest):
    """Open, close or resize a store. The event is journaled even if the projection
    rejects it (409); an accepted one comes back with what it did to demand."""
    net = _live()
    body = request.model_dump()
    if body["store_id"] is None:
        if body["kind"] != "open":
            raise HTTPException(status_code=422, detail="store_id is required for close/resize")
        body["store_id"] = net.next_store_id()
    try:
        event = NetworkEvent.from_payload(body)
    except InvalidEventError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    outcome = net.submit(event)
    logger.info("event %s -> %s", event.as_dict(), outcome["status"])
    if outcome["status"] == "rejected":
        return JSONResponse(status_code=409, content=outcome)
    return outcome


@router.post("/network/reset")
def reset_network() -> dict:
    net = _live()
    dropped = net.reset()
    return {"dropped_events": dropped, **net.summary()}


@router.post("/network/rollout")
def rollout(request: RolloutRequest) -> dict:
    """Plan an n-store expansion by greedy sequential re-sweeps; `commit` opens them."""
    net = _live()
    placement = get_config()["placement"]
    size = request.size_sqm or placement["new_store_size"]
    result = plan_rollout(
        net.worker, request.n_stores, size, placement["candidate_step"], net.next_store_id()
    )
    if request.commit:
        for step in result["plan"]:
            net.submit(
                NetworkEvent.open(
                    step["store_id"],
                    step["x"],
                    step["y"],
                    size,
                    note=f"rollout step {step['step']}",
                )
            )
    result["committed"] = request.commit
    result["network"] = net.summary()
    return result
