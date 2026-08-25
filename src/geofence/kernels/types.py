"""Data-Oriented Vectorised Kernels - Types and Data Layouts.

Array-first Structure-of-Arrays (SoA) containers and optimal site results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SpatialStoreGrid:
    """Structure-of-Arrays (SoA) layout for high-throughput spatial vectorization."""

    store_ids: np.ndarray  # int64 [S]
    x_coords: np.ndarray  # float64 [S]
    y_coords: np.ndarray  # float64 [S]
    size_sqm: np.ndarray  # float64 [S]

    def __post_init__(self) -> None:
        n = len(self.store_ids)
        if not (len(self.x_coords) == len(self.y_coords) == len(self.size_sqm) == n):
            raise ValueError(f"Array dimension mismatch in SpatialStoreGrid: size={n}")


@dataclass(frozen=True)
class OptimalSiteResult:
    """Outcome of vectorized candidate site search across spatial grid."""

    best_x: float
    best_y: float
    candidate_size_sqm: float
    total_captured_demand: float
    cannibalized_demand: float
    net_new_captured_demand: float
    cannibalization_rate_pct: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "best_coord": (round(self.best_x, 2), round(self.best_y, 2)),
            "size_sqm": round(self.candidate_size_sqm, 1),
            "total_captured": round(self.total_captured_demand, 1),
            "cannibalized": round(self.cannibalized_demand, 1),
            "net_new_demand": round(self.net_new_captured_demand, 1),
            "cannibalization_rate": f"{self.cannibalization_rate_pct:.1f}%",
        }
