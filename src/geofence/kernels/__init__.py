"""Data-Oriented Vectorised Kernels Package for GeoFence.

Provides zero-OOP array-first kernels for spatial gravity modeling.
"""

from geofence.kernels.spatial import (
    captured_demand_kernel,
    distance_tensor_kernel,
    evaluate_site_placement_kernel,
    huff_attraction_kernel,
    patronage_share_kernel,
)
from geofence.kernels.types import OptimalSiteResult, SpatialStoreGrid

__all__ = [
    "OptimalSiteResult",
    "SpatialStoreGrid",
    "captured_demand_kernel",
    "distance_tensor_kernel",
    "evaluate_site_placement_kernel",
    "huff_attraction_kernel",
    "patronage_share_kernel",
]
