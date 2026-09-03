"""Gravity worker: keeps the per-store attraction stack current as the network changes.

`models/huff.py` is the reference implementation and rebuilds every store's
attraction map on each call. That is fine for one score, but the placement
sweep scores hundreds of candidates against the same network, and every
network event would otherwise mean a full rebuild. The worker keeps the
(stores, H, W) stack and patches exactly one slice per open/close/resize.

Scoring a candidate then needs one new map. With `T = stack.sum(0)` before and
`T' = T + c` after adding candidate map `c`, the candidate captures
`sum(pop * c / T')`, and existing stores lose `sum(pop * c / T')` over the cells
they already covered (`T > 0`). The difference - net-new demand - is therefore
exactly the population in cells the candidate covers and nobody else did.
Tests pin the worker to the reference model to 1e-9.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from geofence.models.huff import _distances


class GravityWorker:
    def __init__(
        self,
        population: np.ndarray,
        stores: pd.DataFrame,
        alpha: float,
        beta: float,
        max_distance: float,
    ) -> None:
        self.population = np.asarray(population, dtype=float)
        self.alpha, self.beta, self.max_distance = float(alpha), float(beta), float(max_distance)
        self.ids: list[int] = [int(s) for s in stores["store_id"]]
        self.xs: list[float] = [float(v) for v in stores["x"]]
        self.ys: list[float] = [float(v) for v in stores["y"]]
        self.sizes: list[float] = [float(v) for v in stores["size_sqm"]]
        maps = [self._map(x, y, s) for x, y, s in zip(self.xs, self.ys, self.sizes, strict=True)]
        self._stack = np.stack(maps) if maps else np.zeros((0, *self.population.shape))
        self._total: np.ndarray | None = None

    # -- attraction stack maintenance ------------------------------------------------------
    @property
    def grid(self) -> int:
        return self.population.shape[0]

    def _map(self, x: float, y: float, size: float) -> np.ndarray:
        d = _distances(self.grid, x, y)
        a = (size**self.alpha) / (d**self.beta)
        a[d > self.max_distance] = 0.0
        return a

    def _index(self, store_id: int) -> int:
        try:
            return self.ids.index(int(store_id))
        except ValueError:
            raise KeyError(f"store {store_id} is not in the network") from None

    def total_attraction(self) -> np.ndarray:
        if self._total is None:
            self._total = self._stack.sum(axis=0)
        return self._total

    def open(self, store_id: int, x: float, y: float, size: float) -> None:
        if int(store_id) in self.ids:
            raise KeyError(f"store {store_id} already exists")
        self.ids.append(int(store_id))
        self.xs.append(float(x))
        self.ys.append(float(y))
        self.sizes.append(float(size))
        self._stack = np.concatenate([self._stack, self._map(x, y, size)[None]])
        self._total = None

    def close(self, store_id: int) -> None:
        i = self._index(store_id)
        for seq in (self.ids, self.xs, self.ys, self.sizes):
            del seq[i]
        self._stack = np.delete(self._stack, i, axis=0)
        self._total = None

    def resize(self, store_id: int, size: float) -> None:
        i = self._index(store_id)
        self.sizes[i] = float(size)
        self._stack[i] = self._map(self.xs[i], self.ys[i], size)
        self._total = None

    def copy(self) -> GravityWorker:
        clone = GravityWorker.__new__(GravityWorker)
        clone.__dict__.update(self.__dict__)
        clone.ids, clone.xs, clone.ys = list(self.ids), list(self.xs), list(self.ys)
        clone.sizes, clone._stack = list(self.sizes), self._stack.copy()
        return clone

    # -- read side -------------------------------------------------------------------------
    def stores_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"store_id": self.ids, "x": self.xs, "y": self.ys, "size_sqm": self.sizes}
        )

    def demand(self) -> np.ndarray:
        """Captured demand per store, in `self.ids` order."""
        total = self.total_attraction()
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(total > 0, self._stack / total, 0.0)
        return (share * self.population).sum(axis=(1, 2))

    def total_captured(self) -> float:
        return float(self.population[self.total_attraction() > 0].sum())

    def coverage_fraction(self) -> float:
        return self.total_captured() / float(self.population.sum())

    def score(self, x: float, y: float, size: float) -> dict[str, Any]:
        """Same contract as `huff.score_site`, against the live network, without a rebuild."""
        c = self._map(x, y, size)
        before = self.total_attraction()
        after = before + c
        with np.errstate(divide="ignore", invalid="ignore"):
            taken = np.where(after > 0, c / after, 0.0) * self.population
        captured = float(taken.sum())
        cannibalized = float(taken[before > 0].sum())
        return {
            "x": float(x),
            "y": float(y),
            "captured_demand": round(captured, 1),
            "cannibalized_from_existing": round(cannibalized, 1),
            "net_new_demand": round(captured - cannibalized, 1),
            "cannibalization_rate": round(cannibalized / max(captured, 1e-9), 4),
        }

    def sweep(self, step: int, size: float) -> list[dict[str, Any]]:
        grid = self.grid
        return [
            self.score(float(x), float(y), size)
            for y in range(2, grid - 2, step)
            for x in range(2, grid - 2, step)
        ]

    def best(self, top_k: int, step: int, size: float) -> list[dict[str, Any]]:
        results = self.sweep(step, size)
        results.sort(key=lambda r: -r["net_new_demand"])
        return results[:top_k]


def plan_rollout(
    worker: GravityWorker, n_stores: int, size: float, step: int, first_id: int
) -> dict[str, Any]:
    """Greedy sequential expansion: open the best net-new site, re-sweep, repeat.

    Also evaluates the naive alternative - taking the top-n of a single sweep as
    a batch - so the caller can see how much that list double-counts. Sites in
    one sweep are scored against the same network, so several of them can be
    "the best" for the same uncovered pocket; opened together they split it.
    Works on copies; nothing is committed to the network here.
    """
    if n_stores < 1:
        raise ValueError("n_stores must be >= 1")
    sim = worker.copy()
    baseline = sim.total_captured()
    plan: list[dict[str, Any]] = []
    for i in range(n_stores):
        candidates = sim.best(1, step, size)
        if not candidates or candidates[0]["net_new_demand"] <= 0:
            break  # nothing left to cover: every further store would be pure cannibalization
        pick = candidates[0]
        store_id = first_id + i
        sim.open(store_id, pick["x"], pick["y"], size)
        plan.append(
            {
                **pick,
                "store_id": store_id,
                "step": i + 1,
                "cumulative_gain": round(sim.total_captured() - baseline, 1),
            }
        )

    static = worker.best(n_stores, step, size)
    batch = worker.copy()
    for i, site in enumerate(static):
        batch.open(first_id + i, site["x"], site["y"], size)
    claimed = sum(s["net_new_demand"] for s in static)
    realized = batch.total_captured() - baseline
    return {
        "n_requested": n_stores,
        "size_sqm": float(size),
        "plan": plan,
        "sequential_gain": round(sim.total_captured() - baseline, 1),
        "static_top_k": {
            "sites": static,
            "claimed_gain": round(claimed, 1),
            "realized_gain": round(realized, 1),
            "overcount": round(claimed / realized - 1, 4) if realized > 0 else None,
        },
    }
