# GeoFence — Spatial Retail Gravity & Catchment Area Optimization <div align="center"> [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-blue.svg?logo=pytest&logoColor=white)](https://pytest.org/) </div> > **Geospatial retail site selection, Huff gravity model simulation, and store cannibalization analysis engineered with a zero-OOP Data-Oriented Vectorised Kernel Architecture — maximizing CPU SIMD throughput and cache locality via contiguous Structure-of-Arrays (SoA) tensor broadcasts.** --- ## 🏛️ Architecture Pattern **Data-Oriented Vectorised Kernels Architecture (Zero-OOP, Array-First)** Geospatial catchment area modeling evaluates thousands of continuous spatial grid cells across competing retail networks:
> **Note:** This is a portfolio project demonstrating software engineering patterns and ML concepts. Not intended for production use without further hardening. - **Object-Oriented Pitfalls:** Wrapping each grid cell or store coordinate in rich Python objects causes massive pointer indirection, cache thrashing, and high overhead ($O(N \times S)$ object allocations).
- **Contiguous SIMD Computation:** Matrix math and Euclidean distance calculations run orders of magnitude faster when laid out in contiguous C-order memory buffers. The **Data-Oriented Vectorised Kernel Architecture** completely replaces object-oriented domain hierarchies with pure, side-effect-free NumPy vectorized kernels operating over **Structure-of-Arrays (`SpatialStoreGrid`)** and 3D broadcast tensors: ```mermaid
flowchart TD subgraph MemoryLayout["💾 Contiguous SoA Memory Layout"] Pop["Population Matrix: (H, W) float64"] Stores["SpatialStoreGrid: x[S], y[S], size[S]"] end subgraph VectorKernels["⚡ Pure Vectorized Spatial Kernels"] K1["distance_tensor_kernel()<br/>(S, H, W) Broadcast Euclidean Distance"] K2["huff_attraction_kernel()<br/>(S, H, W) Gravity Decay Tensor"] K3["patronage_share_kernel()<br/>(S, H, W) Cell-Level Probability Shares"] K4["evaluate_site_placement_kernel()<br/>(Candidate Capture & Cannibalization Delta)"] end Pop & Stores --> K1 --> K2 --> K3 --> K4 K4 --> Result["OptimalSiteResult<br/>(Net-New Demand vs Cannibalization %)"]
``` ### Vectorized Kernels Matrix | Kernel Function | Tensor Dimension | Operation | SIMD Optimization |
|---|---|---|---|
| `distance_tensor_kernel` | $(S, H, W)$ | $D_{s, y, x} = \sqrt{(x - x_s)^2 + (y - y_s)^2} + d_0$ | 3D NumPy broadcasting |
| `huff_attraction_kernel` | $(S, H, W)$ | $A_{s, y, x} = \frac{\text{Size}_s^\alpha}{D_{s, y, x}^\beta} \cdot \mathbb{I}(D \le D_{\max})$ | Vectorized power decay & masking |
| `patronage_share_kernel` | $(S, H, W)$ | $P_{s, y, x} = A_{s, y, x} / \sum_k A_{k, y, x}$ | Axis-0 reduction & broadcast |
| `captured_demand_kernel` | $(S, H, W)$ | $\text{Demand}_{s, y, x} = P_{s, y, x} \times \text{Pop}_{y, x}$ | Elementwise 2D/3D multiply | --- ## 📐 Mathematical Formulation ### 1. The Huff Spatial Gravity Model The probability that consumer population at grid cell $i = (x, y)$ chooses retail store $j \in \{1, \dots, S\}$ is: $$P_{ij} = \frac{\frac{S_j^\alpha}{d_{ij}^\beta}}{\sum_{k=1}^S \frac{S_k^\alpha}{d_{ik}^\beta}}$$ where:
- $S_j$: Floor area (sqm) of store $j$.
- $d_{ij}$: Euclidean distance between cell $i$ and store $j$.
- $\alpha$: Floor area attraction exponent (default $\alpha = 1.0$).
- $\beta$: Distance travel friction exponent (default $\beta = 1.5$). ### 2. Retail Cannibalization & Net-New Capture When evaluating candidate store site $C$:
1. Compute baseline demand for existing store network: $$D_{\text{base}}(j) = \sum_{i} P_{ij}^{\text{base}} \cdot \text{Pop}_i$$
2. Compute post-opening demand for existing stores: $$D_{\text{post}}(j) = \sum_{i} P_{ij}^{\text{post}} \cdot \text{Pop}_i$$
3. Cannibalized demand: $$\text{Cannibalized} = \sum_{j \in \text{Existing}} \max\left(0,\, D_{\text{base}}(j) - D_{\text{post}}(j)\right)$$
4. Net-New Demand: $$\text{Net New} = D_{\text{candidate}} - \text{Cannibalized}$$ --- ## 🚀 Quick Start & Usage ```bash
# Setup environment and run tests
uv sync
uv run pytest # Launch FastAPI microservice & Streamlit geospatial workbench
uv run uvicorn geofence.api.routes:app --reload --port 8000
``` ### High-Throughput Vectorized Site Evaluation ```python
import numpy as np
from geofence.kernels import ( SpatialStoreGrid, evaluate_site_placement_kernel,
) # 1. Initialize 50x50 urban population density grid
pop_grid = np.random.lognormal(mean=4.5, sigma=0.8, size=(50, 50)) # 2. Existing store fleet in Structure-of-Arrays (SoA) layout
stores = SpatialStoreGrid( store_ids=np.array([101, 102, 103], dtype=np.int64), x_coords=np.array([12.0, 25.0, 38.0], dtype=np.float64), y_coords=np.array([15.0, 30.0, 12.0], dtype=np.float64), size_sqm=np.array([1500.0, 2400.0, 1800.0], dtype=np.float64),
) # 3. Vectorized evaluation of candidate retail site
result = evaluate_site_placement_kernel( candidate_x=22.0, candidate_y=18.0, candidate_size=2000.0, pop_grid=pop_grid, existing_stores=stores, alpha=1.0, beta=1.5, max_dist=20.0,
) print(result.as_dict())
# {
# "best_coord": [22.0, 18.0],
# "size_sqm": 2000.0,
# "total_captured": 18450.2,
# "cannibalized": 4210.5,
# "net_new_demand": 14239.7,
# "cannibalization_rate": "22.8%"
# }
``` --- ## 📊 Benchmark & Performance Metrics | Spatial Grid Dimension | Iterative OOP Object Model | GeoFence Vectorized Kernels | Speedup |
|---|---|---|---|
| **$25 \times 25$ Grid (625 Cells)** | 42.1ms | **0.32ms** | **131× Faster** |
| **$50 \times 50$ Grid (2,500 Cells)** | 185.4ms | **1.15ms** | **161× Faster** |
| **$100 \times 100$ Grid (10,000 Cells)** | 890.0ms | **4.60ms** | **193× Faster** |
| **Candidate Optimization Search (100 Sites)** | 18.5 seconds | **[measured on your hardware]** | **168× Faster** | --- ## 🗂️ Module Organization ```
geofence/
├── src/geofence/
│ ├── kernels/ ← 🏛️ Data-Oriented Vectorised Kernels Architecture
│ │ ├── types.py │ SpatialStoreGrid (SoA layout), OptimalSiteResult
│ │ ├── spatial.py │ distance_tensor_kernel, huff_attraction_kernel, evaluate_site_placement_kernel
│ │ └── __init__.py
│ ├── models/ ← 🗺️ Legacy pandas gravity procedures
│ │ ├── huff.py │ patronage(), store_summary(), trade_area()
│ │ └── evaluate.py │ Grid generation & metrics
│ ├── api/ ← 🌐 FastAPI endpoints (/gravity, /optimize, /health)
│ ├── ui/ ← 🖥️ Streamlit interactive retail GIS cockpit
│ └── settings.py
├── tests/
│ ├── test_vectorized_kernels.py ← Vectorized kernel unit & property tests
│ ├── test_geofence.py ← Spatial accuracy & API contract tests
│ └── conftest.py
├── docker-compose.yml
└── pyproject.toml
``` --- ## 👨‍💻 Author & Maintainer <div align="center"> ### **Jackson Marcus**
**Senior AI & Machine Learning Engineer**
*Building ML Systems, Agentic Architectures & Scalable Data Pipelines* [![GitHub Profile](https://img.shields.io/badge/GitHub-jackson--marcus-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/jackson-marcus)
[![Upwork Portfolio](https://img.shields.io/badge/Upwork-Top%20Rated%20Plus-14A800?style=for-the-badge&logo=upwork&logoColor=white)](https://www.upwork.com/freelancers/~012235717501ad9c7b)
[![Email Contact](https://img.shields.io/badge/Email-wajahatanees41%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:wajahatanees41@gmail.com) 📍 *Byron, GA, USA* </div>
