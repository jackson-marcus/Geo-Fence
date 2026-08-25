"""Data-Oriented Vectorised Spatial Kernels.

Pure NumPy vectorized kernels operating on contiguous memory arrays.
Zero OOP overhead, cache-friendly SIMD broadcasting for spatial gravity models.
"""

from __future__ import annotations

import numpy as np

from geofence.kernels.types import OptimalSiteResult, SpatialStoreGrid


def distance_tensor_kernel(
    grid_h: int,
    grid_w: int,
    store_x: np.ndarray,
    store_y: np.ndarray,
    intra_cell_distance: float = 0.3,
) -> np.ndarray:
    """Computes (n_stores, grid_h, grid_w) Euclidean distance tensor via broadcasting."""
    yy, xx = np.mgrid[0:grid_h, 0:grid_w]
    # xx, yy shape: (grid_h, grid_w)
    # store_x, store_y shape: (S,) -> reshape to (S, 1, 1)
    dx = xx[None, :, :] - store_x[:, None, None]
    dy = yy[None, :, :] - store_y[:, None, None]
    return np.sqrt(dx**2 + dy**2) + intra_cell_distance


def huff_attraction_kernel(
    store_sizes: np.ndarray,
    dist_tensor: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.5,
    max_distance_km: float = 25.0,
) -> np.ndarray:
    """Computes (S, H, W) Huff attraction tensor: size^alpha / distance^beta."""
    sizes = store_sizes[:, None, None]
    attraction = (sizes**alpha) / (dist_tensor**beta)
    # Zero out beyond max service radius
    attraction = np.where(dist_tensor <= max_distance_km, attraction, 0.0)
    return attraction


def patronage_share_kernel(attraction_tensor: np.ndarray) -> np.ndarray:
    """Computes (S, H, W) patronage probability shares per grid cell."""
    total_attraction = attraction_tensor.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = np.where(total_attraction > 0.0, attraction_tensor / total_attraction, 0.0)
    return shares


def captured_demand_kernel(
    patronage_shares: np.ndarray,
    population_grid: np.ndarray,
) -> np.ndarray:
    """Computes (S, H, W) absolute demand captured per store per cell."""
    return patronage_shares * population_grid[None, :, :]


def evaluate_site_placement_kernel(
    candidate_x: float,
    candidate_y: float,
    candidate_size: float,
    pop_grid: np.ndarray,
    existing_stores: SpatialStoreGrid,
    alpha: float = 1.0,
    beta: float = 1.5,
    max_dist: float = 25.0,
) -> OptimalSiteResult:
    """Vectorized calculation of candidate site capture and cannibalization."""
    grid_h, grid_w = pop_grid.shape

    # 1. Baseline demand without candidate
    d_base = distance_tensor_kernel(
        grid_h, grid_w, existing_stores.x_coords, existing_stores.y_coords
    )
    a_base = huff_attraction_kernel(existing_stores.size_sqm, d_base, alpha, beta, max_dist)
    p_base = patronage_share_kernel(a_base)
    dem_base = captured_demand_kernel(p_base, pop_grid).sum(axis=(1, 2))

    # 2. Augmented store grid with candidate appended
    all_x = np.append(existing_stores.x_coords, candidate_x)
    all_y = np.append(existing_stores.y_coords, candidate_y)
    all_sizes = np.append(existing_stores.size_sqm, candidate_size)

    d_aug = distance_tensor_kernel(grid_h, grid_w, all_x, all_y)
    a_aug = huff_attraction_kernel(all_sizes, d_aug, alpha, beta, max_dist)
    p_aug = patronage_share_kernel(a_aug)
    dem_aug = captured_demand_kernel(p_aug, pop_grid).sum(axis=(1, 2))

    candidate_captured = float(dem_aug[-1])
    existing_post_capture = dem_aug[:-1]
    cannibalized = float(np.sum(np.maximum(0.0, dem_base - existing_post_capture)))
    net_new = max(0.0, candidate_captured - cannibalized)
    cannibalization_rate = (cannibalized / max(candidate_captured, 1e-6)) * 100.0

    return OptimalSiteResult(
        best_x=candidate_x,
        best_y=candidate_y,
        candidate_size_sqm=candidate_size,
        total_captured_demand=candidate_captured,
        cannibalized_demand=cannibalized,
        net_new_captured_demand=net_new,
        cannibalization_rate_pct=cannibalization_rate,
    )
