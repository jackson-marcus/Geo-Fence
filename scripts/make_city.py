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
    """City with a deliberately over-served downtown and underserved secondary hotspots.

    Retail crowds the densest center (half the stores land there), so a
    capture-greedy site pick cannibalizes heavily while a net-new-aware pick
    finds the underserved hotspots. That planted structure is what
    models/evaluate.py measures.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:grid, 0:grid]
    population = np.full((grid, grid), 40.0)
    centers = [(grid / 2 + rng.uniform(-3, 3), grid / 2 + rng.uniform(-3, 3), 2000.0, 5.0)]
    centers += [
        (
            rng.uniform(5, grid - 5),
            rng.uniform(5, grid - 5),
            rng.uniform(300, 700),
            rng.uniform(3, 6),
        )
        for _ in range(5)
    ]
    for cx, cy, strength, spread in centers:
        population += strength * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * spread**2)))
    population *= rng.lognormal(0, 0.12, population.shape)

    cx0, cy0 = centers[0][0], centers[0][1]
    n_downtown = n_stores // 2 + 1
    xs, ys = [], []
    for i in range(n_stores):
        if i < n_downtown:
            xs.append(float(np.clip(cx0 + rng.uniform(-4, 4), 2, grid - 3)))
            ys.append(float(np.clip(cy0 + rng.uniform(-4, 4), 2, grid - 3)))
        else:
            xs.append(float(rng.uniform(3, grid - 3)))
            ys.append(float(rng.uniform(3, grid - 3)))
    stores = pd.DataFrame(
        {
            "store_id": np.arange(1, n_stores + 1),
            "x": np.round(xs, 1),
            "y": np.round(ys, 1),
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
    print(
        f"Wrote {cfg['grid_size']}x{cfg['grid_size']} grid (pop {population.sum():,.0f}) + {len(stores)} stores -> {out}"
    )


if __name__ == "__main__":
    main()
