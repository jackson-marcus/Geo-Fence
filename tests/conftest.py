"""Offline fixtures: a tiny deterministic city written to a tmp processed dir."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geofence.settings import get_config


@pytest.fixture
def tiny_city():
    """Uniform 24x24 city, three clustered-ish stores; the far corners sit
    beyond the 12km gravity cap so uncovered (net-new) demand exists."""
    population = np.full((24, 24), 100.0)
    stores = pd.DataFrame(
        {
            "store_id": [1, 2, 3],
            "x": [4.0, 5.0, 12.0],
            "y": [4.0, 5.0, 12.0],
            "size_sqm": [1200, 900, 1200],
        }
    )
    return population, stores


@pytest.fixture
def live_network(tiny_city):
    from geofence.streams.consumer import LiveNetwork

    population, stores = tiny_city
    return LiveNetwork(population, stores, get_config()["gravity"])


@pytest.fixture
def api_client(tmp_path, tiny_city):
    """TestClient wired to tmp artifacts; restores config + the network cache afterwards."""
    from fastapi.testclient import TestClient

    from geofence.api import routes
    from geofence.api.main import app

    population, stores = tiny_city
    np.save(tmp_path / "population.npy", population)
    stores.to_parquet(tmp_path / "stores.parquet", index=False)

    cfg = get_config()
    original = cfg["data"]["processed_dir"]
    cfg["data"]["processed_dir"] = str(tmp_path)
    routes._network.cache_clear()
    try:
        yield TestClient(app)
    finally:
        cfg["data"]["processed_dir"] = original
        routes._network.cache_clear()
