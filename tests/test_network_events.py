"""Store-network change log: contracts, projection, the incremental worker, rollout."""

from __future__ import annotations

import numpy as np
import pytest

from geofence.models.huff import best_sites, patronage, score_site, trade_area
from geofence.settings import get_config
from geofence.streams.consumer import EventRejectedError, NetworkProjection
from geofence.streams.producer import NetworkLog
from geofence.streams.schemas import InvalidEventError, NetworkEvent
from geofence.workers.processor import GravityWorker, plan_rollout

STEP = 2
SIZE = 1200.0


def _worker(tiny_city) -> GravityWorker:
    population, stores = tiny_city
    g = get_config()["gravity"]
    return GravityWorker(population, stores, g["alpha"], g["beta"], g["max_distance_km"])


# -- schemas ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "open", "store_id": 4, "size_sqm": 900},  # no coordinates
        {"kind": "open", "store_id": 4, "x": 1, "y": 1},  # no size
        {"kind": "resize", "store_id": 1, "size_sqm": 0},  # zero size
        {"kind": "resize", "store_id": 1, "size_sqm": 50000},  # above the cap
        {"kind": "close", "store_id": 1, "size_sqm": 900},  # close carries no geometry
        {"kind": "relocate", "store_id": 1},  # not a thing
        {"kind": "close", "store_id": 0},
    ],
)
def test_malformed_events_are_refused_before_touching_the_network(payload):
    with pytest.raises(InvalidEventError):
        NetworkEvent.from_payload(payload)


def test_log_assigns_sequence_numbers_and_reads_after_a_cursor():
    log = NetworkLog()
    first = log.append(NetworkEvent.close(1))
    second = log.append(NetworkEvent.resize(1, 800))
    assert (first.seq, second.seq) == (1, 2)
    assert [e.seq for e in log.read(after_seq=1)] == [2]
    assert log.read(after_seq=2) == []
    assert log.truncate() == 2 and log.head == 0


# -- worker vs the reference model -----------------------------------------------------------


def test_worker_matches_reference_model_after_a_burst_of_changes(tiny_city):
    population, _ = tiny_city
    w = _worker(tiny_city)
    w.open(4, 18.0, 18.0, 900.0)
    w.resize(2, 1500.0)
    w.close(1)
    w.open(5, 3.0, 20.0, 600.0)
    stores = w.stores_df()
    assert list(stores.index) == [0, 1, 2, 3]  # RangeIndex: huff.trade_area indexes maps by label
    reference = patronage(population, stores).sum(axis=(1, 2))
    np.testing.assert_allclose(w.demand(), reference, rtol=1e-9)
    for x, y, size in [(18.0, 18.0, 900.0), (2.0, 2.0, 1200.0), (22.0, 2.0, 1800.0)]:
        assert w.score(x, y, size) == score_site(population, stores, x, y, size)
    assert w.best(4, STEP, SIZE) == best_sites(population, stores, top_k=4)
    assert trade_area(population, stores, store_id=5)[20, 3]


def test_net_new_is_exactly_the_newly_covered_population(tiny_city):
    """Under a hard service radius, share shifts inside covered cells are zero-sum:
    the only demand a candidate adds is population nobody reached before."""
    population, _ = tiny_city
    w = _worker(tiny_city)
    covered_before = w.total_attraction() > 0
    for x, y, size in [(20.0, 20.0, 600.0), (20.0, 20.0, 5000.0), (6.0, 6.0, 1200.0)]:
        reach = w._map(x, y, size) > 0
        newly = float(population[reach & ~covered_before].sum())
        assert w.score(x, y, size)["net_new_demand"] == pytest.approx(newly, abs=0.05)
    # so the candidate's size moves capture and cannibalization, never net-new
    small, big = w.score(20.0, 20.0, 600.0), w.score(20.0, 20.0, 5000.0)
    assert big["captured_demand"] > small["captured_demand"]
    assert big["net_new_demand"] == small["net_new_demand"]


def test_closing_a_store_that_was_the_only_coverage_loses_that_population(tiny_city):
    w = _worker(tiny_city)
    total = w.total_captured()
    w.close(3)  # store 3 at (12,12) is the only one reaching the far side of the grid
    assert w.total_captured() < total
    w.open(3, 12.0, 12.0, 1200.0)
    assert w.total_captured() == pytest.approx(total)


# -- projection -------------------------------------------------------------------------------


def test_rejected_events_stay_in_the_log_but_leave_the_network_alone(live_network):
    before = live_network.worker.demand().copy()
    for event in (
        NetworkEvent.close(99),
        NetworkEvent.open(1, 2.0, 2.0, 900.0),  # id already taken
        NetworkEvent.open(9, 30.0, 2.0, 900.0),  # off the 24x24 grid
        NetworkEvent.resize(42, 900.0),
    ):
        outcome = live_network.submit(event)
        assert outcome["status"] == "rejected", event
    assert live_network.log.head == 4
    assert live_network.projection.n_rejected == 4
    np.testing.assert_array_equal(live_network.worker.demand(), before)
    with pytest.raises(EventRejectedError):
        NetworkProjection(live_network.worker).check(NetworkEvent.close(99))


def test_open_effect_is_the_score_and_close_reports_where_demand_goes(live_network):
    opened = live_network.submit(NetworkEvent.open(7, 20.0, 20.0, 1200.0))
    eff = opened["effect"]
    assert eff["network_demand_after"] - eff["network_demand_before"] == pytest.approx(
        eff["net_new_demand"], abs=0.1
    )
    # store 2 at (5,5) sits next to store 1 at (4,4), yet two rim cells - (0,15) and
    # (15,0), 12.005 from store 1 - were reached by store 2 alone; closing it strands them
    w = live_network.worker
    others = np.zeros_like(w.population, dtype=bool)
    for sid in (1, 3, 7):
        others |= w._stack[w._index(sid)] > 0
    stranded = float(w.population[(w._stack[w._index(2)] > 0) & ~others].sum())
    assert stranded == 200.0
    closed = live_network.submit(NetworkEvent.close(2))["effect"]
    assert closed["left_uncovered"] == pytest.approx(stranded)
    assert closed["reabsorbed_by_others"] == pytest.approx(closed["released_demand"] - stranded)
    closed7 = live_network.submit(NetworkEvent.close(7))["effect"]
    assert closed7["left_uncovered"] == pytest.approx(eff["net_new_demand"], abs=0.1)
    assert live_network.summary()["n_stores"] == 2


def test_reset_rewinds_to_the_generated_city_and_ids_are_never_reused(live_network):
    live_network.submit(NetworkEvent.open(4, 20.0, 20.0, 1200.0))
    live_network.submit(NetworkEvent.close(4))
    live_network.submit(NetworkEvent.close(9999))  # rejected: must not bump the counter
    assert live_network.next_store_id() == 5  # 4 is closed but its id is never reused
    assert live_network.reset() == 3
    assert live_network.summary() == {
        "seq": 0,
        "n_stores": 3,
        "events_applied": 0,
        "events_rejected": 0,
        "network_demand": live_network.summary()["network_demand"],
        "coverage_fraction": live_network.summary()["coverage_fraction"],
    }
    assert live_network.next_store_id() == 4


# -- rollout ----------------------------------------------------------------------------------


def test_static_top_k_double_counts_a_pocket_that_sequential_planning_does_not(live_network):
    w = live_network.worker
    result = plan_rollout(w, 3, SIZE, STEP, live_network.next_store_id())
    plan, static = result["plan"], result["static_top_k"]
    assert [p["step"] for p in plan] == [1, 2, 3]
    # per-step marginals are exact, so they sum to the realised network gain
    assert plan[-1]["cumulative_gain"] == pytest.approx(result["sequential_gain"])
    assert sum(p["net_new_demand"] for p in plan) == pytest.approx(
        result["sequential_gain"], abs=0.2
    )
    # the single-sweep list scores every site against the same network: its sum is a fiction
    assert static["claimed_gain"] > static["realized_gain"]
    assert static["overcount"] > 0
    assert result["sequential_gain"] > static["realized_gain"]
    # planning must not touch the live network
    assert live_network.log.head == 0 and len(w.ids) == 3


def test_rollout_stops_when_nothing_is_left_to_cover(tiny_city):
    population, stores = tiny_city
    g = get_config()["gravity"]
    w = GravityWorker(population, stores, g["alpha"], g["beta"], max_distance=100.0)
    assert w.coverage_fraction() == 1.0
    assert plan_rollout(w, 3, SIZE, STEP, 4)["plan"] == []
