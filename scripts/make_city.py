"""Synthetic city: population grid with density clusters + existing stores.

Usage:
    uv run python scripts/make_city.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geofence.settings import get_config, resolve_path


def generate(grid: int, n_stores: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:grid, 0:grid]
    population = np.full((grid, grid), 40.0)
    for _ in range(6):  # density hotspots (downtown, suburbs)
        cx, cy = rng.uniform(4, grid - 4, 2)
        strength = rng.uniform(400, 1600)
        spread = rng.uniform(3, 7)
        population += strength * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * spread**2)))
    population *= rng.lognormal(0, 0.12, population.shape)

    stores = pd.DataFrame(
        {
            "store_id": np.arange(1, n_stores + 1),
            "x": rng.uniform(3, grid - 3, n_stores).round(1),
            "y": rng.uniform(3, grid - 3, n_stores).round(1),
            "size_sqm": rng.choice([600, 900, 1200, 1800], n_stores),
        }
    )
    return population.round(1), stores


def main() -> None:
    cfg = get_config()["data"]
    population, stores = generate(cfg["grid_size"], cfg["n_stores"], cfg["seed"])
    out = resolve_path(cfg["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "population.npy", population)
    stores.to_parquet(out / "stores.parquet", index=False)
    print(f"Wrote {cfg['grid_size']}x{cfg['grid_size']} grid (pop {population.sum():,.0f}) + {len(stores)} stores -> {out}")


if __name__ == "__main__":
    main()
