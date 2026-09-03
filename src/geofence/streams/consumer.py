"""Projection: folds the change log into the current network state.

Schema validation (streams/schemas.py) checks an event on its own. The
projection checks it against the network - you cannot close store 42 twice or
open a store off the grid - and applies it to the gravity worker. Rejected
events stay in the log with their reason; they just do not change anything.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from geofence.streams.producer import NetworkLog
from geofence.streams.schemas import NetworkEvent
from geofence.workers.processor import GravityWorker


class EventRejectedError(ValueError):
    """Well-formed event that does not make sense for the network as it stands."""


class NetworkProjection:
    def __init__(self, worker: GravityWorker) -> None:
        self.worker = worker
        self.applied_seq = 0
        self.outcomes: dict[int, dict[str, Any]] = {}

    def check(self, event: NetworkEvent) -> None:
        ids = self.worker.ids
        if event.kind == "open":
            if event.store_id in ids:
                raise EventRejectedError(f"store {event.store_id} already exists")
            grid = self.worker.grid
            if not (event.x < grid and event.y < grid):
                raise EventRejectedError(
                    f"({event.x}, {event.y}) is outside the {grid}x{grid} grid"
                )
        elif event.store_id not in ids:
            raise EventRejectedError(f"store {event.store_id} is not in the network")

    def apply(self, event: NetworkEvent) -> dict[str, Any]:
        """Validate, apply, and describe what the event did to the network's demand."""
        self.check(event)
        w = self.worker
        network_before = w.total_captured()
        if event.kind == "open":
            effect = w.score(event.x, event.y, event.size_sqm)
            w.open(event.store_id, event.x, event.y, event.size_sqm)
        elif event.kind == "close":
            released = float(w.demand()[w._index(event.store_id)])
            w.close(event.store_id)
            lost = network_before - w.total_captured()
            effect = {
                "released_demand": round(released, 1),
                "reabsorbed_by_others": round(released - lost, 1),
                "left_uncovered": round(lost, 1),
            }
        else:
            i = w._index(event.store_id)
            own_before = float(w.demand()[i])
            w.resize(event.store_id, event.size_sqm)
            own_after = float(w.demand()[i])
            effect = {
                "store_demand_before": round(own_before, 1),
                "store_demand_after": round(own_after, 1),
                "taken_from_others": round(own_after - own_before, 1),
            }
        effect["network_demand_before"] = round(network_before, 1)
        effect["network_demand_after"] = round(w.total_captured(), 1)
        return effect

    def catch_up(self, log: NetworkLog) -> list[dict[str, Any]]:
        """Apply every event after `applied_seq`; returns the outcome of each."""
        outcomes = []
        for event in log.read(self.applied_seq):
            try:
                outcome = {"status": "applied", "effect": self.apply(event)}
            except EventRejectedError as exc:
                outcome = {"status": "rejected", "reason": str(exc)}
            outcome["event"] = event.as_dict()
            self.outcomes[event.seq] = outcome
            self.applied_seq = event.seq
            outcomes.append(outcome)
        return outcomes

    @property
    def n_rejected(self) -> int:
        return sum(1 for o in self.outcomes.values() if o["status"] == "rejected")


class LiveNetwork:
    """Log + projection + worker, wired together; what the API holds in memory."""

    def __init__(
        self, population: np.ndarray, stores: pd.DataFrame, gravity: dict[str, float]
    ) -> None:
        self._baseline = (population, stores.reset_index(drop=True))
        self._gravity = gravity
        self.log = NetworkLog()
        self.projection = self._fresh_projection()

    def _fresh_projection(self) -> NetworkProjection:
        population, stores = self._baseline
        worker = GravityWorker(
            population,
            stores,
            self._gravity["alpha"],
            self._gravity["beta"],
            self._gravity["max_distance_km"],
        )
        return NetworkProjection(worker)

    @property
    def worker(self) -> GravityWorker:
        return self.projection.worker

    @property
    def population(self) -> np.ndarray:
        return self._baseline[0]

    def submit(self, event: NetworkEvent) -> dict[str, Any]:
        """Append to the log, fold it in, and return that event's outcome."""
        stamped = self.log.append(event)
        self.projection.catch_up(self.log)
        return self.projection.outcomes[stamped.seq]

    def reset(self) -> int:
        dropped = self.log.truncate()
        self.projection = self._fresh_projection()
        return dropped

    def next_store_id(self) -> int:
        """One above every id that ever existed. Rejected events do not count: a
        mistyped `close 9999` must not push the next opening to 10000."""
        applied = {
            o["event"]["store_id"]
            for o in self.projection.outcomes.values()
            if o["status"] == "applied"
        }
        return max(set(self.worker.ids) | applied, default=0) + 1

    def summary(self) -> dict[str, Any]:
        w = self.worker
        return {
            "seq": self.log.head,
            "n_stores": len(w.ids),
            "events_applied": self.projection.applied_seq - self.projection.n_rejected,
            "events_rejected": self.projection.n_rejected,
            "network_demand": round(w.total_captured(), 1),
            "coverage_fraction": round(w.coverage_fraction(), 4),
        }
