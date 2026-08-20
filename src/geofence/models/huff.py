"""Huff gravity model: patronage probabilities, trade areas, site scoring.

P(cell -> store) = (size^alpha / d^beta) / sum_over_stores(...), capped at a
max distance. Site scoring = captured demand of a hypothetical new store;
cannibalization = demand it steals from existing stores vs net-new capture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from geofence.settings import get_config


def _distances(grid: int, x: float, y: float) -> np.ndarray:
    yy, xx = np.mgrid[0:grid, 0:grid]
    return np.sqrt((xx - x) ** 2 + (yy - y) ** 2) + 0.3  # 0.3 = intra-cell distance


def attraction_maps(population: np.ndarray, stores: pd.DataFrame) -> np.ndarray:
    """(n_stores, grid, grid) attraction values, distance-capped."""
    cfg = get_config()["gravity"]
    grid = population.shape[0]
    maps = []
    for _, store in stores.iterrows():
        d = _distances(grid, store["x"], store["y"])
        a = (store["size_sqm"] ** cfg["alpha"]) / (d ** cfg["beta"])
        a[d > cfg["max_distance_km"]] = 0.0
        maps.append(a)
    return np.array(maps)


def patronage(population: np.ndarray, stores: pd.DataFrame) -> np.ndarray:
    """(n_stores, grid, grid) demand captured per store per cell."""
    maps = attraction_maps(population, stores)
    total = maps.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(total > 0, maps / total, 0.0)
    return share * population[None, :, :]


def store_summary(population: np.ndarray, stores: pd.DataFrame) -> pd.DataFrame:
    demand = patronage(population, stores).sum(axis=(1, 2))
    out = stores.copy()
    out["captured_demand"] = np.round(demand, 1)
    out["demand_share"] = np.round(demand / population.sum(), 4)
    return out


def trade_area(population: np.ndarray, stores: pd.DataFrame, store_id: int, threshold: float = 0.5):
    """Cells where this store's capture probability exceeds `threshold`."""
    maps = attraction_maps(population, stores)
    total = maps.sum(axis=0)
    idx = stores.index[stores["store_id"] == store_id][0]
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(total > 0, maps[idx] / total, 0.0)
    return share >= threshold


def score_site(
    population: np.ndarray, stores: pd.DataFrame, x: float, y: float, size: float
) -> dict:
    """Add a hypothetical store; report net-new vs cannibalized demand."""
    before = patronage(population, stores).sum(axis=(1, 2))
    candidate = pd.concat(
        [stores, pd.DataFrame([{"store_id": -1, "x": x, "y": y, "size_sqm": size}])],
        ignore_index=True,
    )
    after_all = patronage(population, candidate)
    after = after_all[:-1].sum(axis=(1, 2))
    new_capture = float(after_all[-1].sum())
    cannibalized = float((before - after).sum())
    return {
        "x": x,
        "y": y,
        "captured_demand": round(new_capture, 1),
        "cannibalized_from_existing": round(cannibalized, 1),
        "net_new_demand": round(new_capture - cannibalized, 1),
        "cannibalization_rate": round(cannibalized / max(new_capture, 1e-9), 4),
    }


def sweep_sites(population: np.ndarray, stores: pd.DataFrame) -> list[dict]:
    """Score every candidate cell on the placement grid."""
    cfg = get_config()["placement"]
    grid = population.shape[0]
    results = []
    for y in range(2, grid - 2, cfg["candidate_step"]):
        for x in range(2, grid - 2, cfg["candidate_step"]):
            results.append(
                score_site(population, stores, float(x), float(y), cfg["new_store_size"])
            )
    return results


def best_sites(population: np.ndarray, stores: pd.DataFrame, top_k: int = 10) -> list[dict]:
    results = sweep_sites(population, stores)
    results.sort(key=lambda r: -r["net_new_demand"])
    return results[:top_k]


def isochrone(
    population: np.ndarray,
    stores: pd.DataFrame,
    store_id: int,
    minutes: float = 15.0,
    speed_kmh: float = 30.0,
) -> np.ndarray:
    """Drive-time reach: cells within `minutes` of the store at average city speed."""
    grid = population.shape[0]
    row = stores.loc[stores["store_id"] == store_id].iloc[0]
    d = _distances(grid, row["x"], row["y"])
    return d <= speed_kmh * minutes / 60.0
