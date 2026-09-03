"""Static top-k vs greedy sequential rollout, plus sweep timing, on the generated city.

Usage:
    uv run python scripts/rollout_benchmark.py [--k 3]

Everything printed here comes from data/processed/ as written by make_city.py
with configs/config.yaml; regenerate the city and the numbers change.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geofence.models.huff import best_sites
from geofence.settings import get_config, resolve_path
from geofence.streams.consumer import LiveNetwork
from geofence.workers.processor import plan_rollout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    cfg = get_config()
    processed = resolve_path(cfg["data"]["processed_dir"])
    population = np.load(processed / "population.npy")
    stores = pd.read_parquet(processed / "stores.parquet")
    placement = cfg["placement"]
    step, size = placement["candidate_step"], placement["new_store_size"]
    net = LiveNetwork(population, stores, cfg["gravity"])

    t0 = time.perf_counter()
    reference = best_sites(population, stores, top_k=args.k)
    t_ref = time.perf_counter() - t0
    t0 = time.perf_counter()
    worker_best = net.worker.best(args.k, step, size)
    t_worker = time.perf_counter() - t0
    assert worker_best == reference, "worker diverged from models/huff.py"
    n_candidates = len(net.worker.sweep(step, size))
    print(
        f"sweep of {n_candidates} candidates: huff.py {t_ref:.3f}s, worker {t_worker:.3f}s "
        f"({t_ref / t_worker:.0f}x), identical top-{args.k}"
    )

    result = plan_rollout(net.worker, args.k, size, step, net.next_store_id())
    static = result["static_top_k"]
    print(f"\nstatic top-{args.k} of one sweep:")
    for s in static["sites"]:
        print(f"  ({s['x']:>4.0f},{s['y']:>4.0f}) net-new {s['net_new_demand']:>10,.1f}")
    print(f"  claimed (sum of scores) {static['claimed_gain']:>12,.1f}")
    print(
        f"  realised opened together {static['realized_gain']:>11,.1f}  "
        f"overcount {static['overcount']:.1%}"
    )
    print(f"\ngreedy sequential rollout ({args.k} re-sweeps):")
    for p in result["plan"]:
        print(
            f"  {p['step']}. ({p['x']:>4.0f},{p['y']:>4.0f}) net-new {p['net_new_demand']:>10,.1f}"
            f"  cannibalization {p['cannibalization_rate']:.0%}  cumulative {p['cumulative_gain']:,.1f}"
        )
    print(
        f"  realised {result['sequential_gain']:>12,.1f}  "
        f"({result['sequential_gain'] / static['realized_gain'] - 1:+.1%} vs static batch)"
    )


if __name__ == "__main__":
    main()
