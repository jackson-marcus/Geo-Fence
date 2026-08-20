"""Huff model invariants + placement logic + API contract."""

from __future__ import annotations

import numpy as np

from geofence.models.huff import (
    attraction_maps,
    best_sites,
    isochrone,
    patronage,
    score_site,
    trade_area,
)


def test_patronage_conserves_covered_population(tiny_city):
    population, stores = tiny_city
    shares = patronage(population, stores)
    covered = attraction_maps(population, stores).sum(axis=0) > 0
    assert np.isclose(shares.sum(), population[covered].sum(), rtol=1e-6)
    assert (shares >= 0).all()


def test_cannibalization_orders_by_proximity(tiny_city):
    population, stores = tiny_city
    near = score_site(population, stores, 5.0, 4.0, 1200)
    far = score_site(population, stores, 20.0, 20.0, 1200)
    assert near["cannibalization_rate"] > far["cannibalization_rate"]
    assert far["net_new_demand"] > near["net_new_demand"]


def test_best_sites_sorted_with_expected_fields(tiny_city):
    population, stores = tiny_city
    sites = best_sites(population, stores, top_k=5)
    assert len(sites) == 5
    net_new = [s["net_new_demand"] for s in sites]
    assert net_new == sorted(net_new, reverse=True)
    assert {"captured_demand", "cannibalized_from_existing", "cannibalization_rate"} <= set(
        sites[0]
    )


def test_isochrone_grows_and_trade_area_contains_home_cell(tiny_city):
    population, stores = tiny_city
    short = isochrone(population, stores, store_id=1, minutes=10)
    long = isochrone(population, stores, store_id=1, minutes=30)
    assert long.sum() > short.sum()
    capture = trade_area(population, stores, store_id=3, threshold=0.5)
    assert capture[12, 12]  # store 3 dominates its own cell (no rival nearby)


def test_api_contract(api_client):
    assert api_client.get("/health").json() == {"status": "ok"}

    stores = api_client.get("/stores").json()
    assert len(stores) == 3 and "captured_demand" in stores[0]

    scored = api_client.post("/score-site", json={"x": 8, "y": 8, "size_sqm": 1200}).json()
    assert "net_new_demand" in scored and scored["captured_demand"] >= scored["net_new_demand"]
    assert (
        api_client.post("/score-site", json={"x": 999, "y": 1, "size_sqm": 900}).status_code == 422
    )

    trade = api_client.get("/trade-area/1", params={"minutes": 12}).json()
    assert trade["isochrone_cells"] > 0 and trade["capture_population"] >= 0
    assert api_client.get("/trade-area/99").status_code == 404
