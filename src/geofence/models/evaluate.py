"""Honest placement benchmark: capture-greedy vs cannibalization-aware site picks.

The naive strategy maximizes raw captured demand; the aware strategy maximizes
net-new demand (capture minus what it steals from existing stores). On a city
whose downtown is already over-served, the two diverge — that gap is the metric.

Usage:
    python -m geofence.models.evaluate
"""

from __future__ import annotations

import logging

import mlflow
import numpy as np
import pandas as pd

from geofence.models.huff import attraction_maps, sweep_sites
from geofence.settings import get_config, get_settings, resolve_path

logger = logging.getLogger(__name__)


def evaluate() -> dict:
    cfg = get_config()
    processed = resolve_path(cfg["data"]["processed_dir"])
    population = np.load(processed / "population.npy")
    stores = pd.read_parquet(processed / "stores.parquet")

    results = sweep_sites(population, stores)
    naive = max(results, key=lambda r: r["captured_demand"])
    aware = max(results, key=lambda r: r["net_new_demand"])

    covered = attraction_maps(population, stores).sum(axis=0) > 0
    metrics = {
        "naive_net_new": naive["net_new_demand"],
        "naive_cannibalization_rate": naive["cannibalization_rate"],
        "aware_net_new": aware["net_new_demand"],
        "aware_cannibalization_rate": aware["cannibalization_rate"],
        "net_new_uplift": round(
            aware["net_new_demand"] / max(naive["net_new_demand"], 1e-9) - 1, 4
        ),
        "coverage_fraction": round(float(population[covered].sum() / population.sum()), 4),
        "n_candidates": len(results),
    }

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(cfg["eval"]["experiment_name"])
    with mlflow.start_run(run_name="site-search"):
        mlflow.log_params(
            {
                "grid": population.shape[0],
                "n_stores": len(stores),
                "alpha": cfg["gravity"]["alpha"],
                "beta": cfg["gravity"]["beta"],
            }
        )
        mlflow.log_metrics(metrics)
    logger.info("placement benchmark %s", metrics)
    logger.info(
        "naive site (%s,%s) vs aware site (%s,%s)", naive["x"], naive["y"], aware["x"], aware["y"]
    )
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    evaluate()
