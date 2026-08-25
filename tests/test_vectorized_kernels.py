"""Unit tests for the Data-Oriented Vectorised Kernels in GeoFence."""

from __future__ import annotations

import numpy as np
import pytest

from geofence.kernels import (
    OptimalSiteResult,
    SpatialStoreGrid,
    captured_demand_kernel,
    distance_tensor_kernel,
    evaluate_site_placement_kernel,
    huff_attraction_kernel,
    patronage_share_kernel,
)


@pytest.fixture
def synthetic_pop_grid() -> np.ndarray:
    """30x30 spatial population grid with uniform demand."""
    return np.ones((30, 30), dtype=np.float64) * 100.0  # 90,000 total population


@pytest.fixture
def synthetic_stores() -> SpatialStoreGrid:
    """3 existing retail stores in SoA layout."""
    return SpatialStoreGrid(
        store_ids=np.array([1, 2, 3], dtype=np.int64),
        x_coords=np.array([5.0, 15.0, 25.0], dtype=np.float64),
        y_coords=np.array([5.0, 15.0, 25.0], dtype=np.float64),
        size_sqm=np.array([1200.0, 2500.0, 1800.0], dtype=np.float64),
    )


def test_distance_tensor_kernel_shape(synthetic_stores):
    d_tensor = distance_tensor_kernel(30, 30, synthetic_stores.x_coords, synthetic_stores.y_coords)
    assert d_tensor.shape == (3, 30, 30)
    # Check minimum distance at store 1's coordinate (5, 5)
    assert d_tensor[0, 5, 5] == pytest.approx(0.3)  # intra-cell distance
    assert np.all(d_tensor > 0.0)


def test_huff_attraction_and_patronage_kernel(synthetic_pop_grid, synthetic_stores):
    d_tensor = distance_tensor_kernel(30, 30, synthetic_stores.x_coords, synthetic_stores.y_coords)
    attraction = huff_attraction_kernel(synthetic_stores.size_sqm, d_tensor, alpha=1.0, beta=1.5)
    assert attraction.shape == (3, 30, 30)

    shares = patronage_share_kernel(attraction)
    assert shares.shape == (3, 30, 30)

    # Invariant: Sum of patronage probabilities across stores for each cell <= 1.0
    sum_shares = shares.sum(axis=0)
    assert np.all(sum_shares <= 1.0 + 1e-6)

    demand = captured_demand_kernel(shares, synthetic_pop_grid)
    assert demand.shape == (3, 30, 30)
    total_captured = demand.sum()
    assert total_captured > 0.0
    assert total_captured <= synthetic_pop_grid.sum()


def test_evaluate_site_placement_kernel(synthetic_pop_grid, synthetic_stores):
    res = evaluate_site_placement_kernel(
        candidate_x=10.0,
        candidate_y=20.0,
        candidate_size=2000.0,
        pop_grid=synthetic_pop_grid,
        existing_stores=synthetic_stores,
        alpha=1.0,
        beta=1.5,
    )

    assert isinstance(res, OptimalSiteResult)
    assert res.total_captured_demand > 0.0
    assert res.cannibalized_demand >= 0.0
    assert res.net_new_captured_demand >= 0.0
    assert res.cannibalized_demand + res.net_new_captured_demand == pytest.approx(
        res.total_captured_demand
    )
    assert "cannibalization_rate" in res.as_dict()
