"""Streamlit demo: population heatmap with stores + site scorer."""

from __future__ import annotations

import os

import httpx
import numpy as np
import plotly.graph_objects as go
import streamlit as st

API_URL = os.environ.get("GEOFENCE_API_URL", "http://localhost:8290")

st.set_page_config(page_title="geofence", page_icon="📍", layout="wide")
st.title("📍 geofence")
st.caption("Huff gravity model: trade areas, site scoring, cannibalization")


def _ok() -> bool:
    try:
        return httpx.get(f"{API_URL}/health", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


if not _ok():
    st.error(f"API not reachable at {API_URL}. Start it with `make api`.")
    st.stop()

r = httpx.get(f"{API_URL}/heatmap", timeout=60)
if r.status_code != 200:
    st.warning(r.json().get("detail", r.text))
    st.stop()
body = r.json()
population = np.array(body["population"])

col1, col2 = st.columns([3, 2])
with col1:
    fig = go.Figure(go.Heatmap(z=population, colorscale="YlOrRd", showscale=False))
    fig.add_trace(
        go.Scatter(
            x=[s["x"] for s in body["stores"]],
            y=[s["y"] for s in body["stores"]],
            mode="markers+text",
            text=[str(s["store_id"]) for s in body["stores"]],
            textposition="top center",
            marker={"size": 12, "color": "blue", "symbol": "square"},
            name="stores",
        )
    )
    fig.update_layout(height=520, margin={"l": 10, "r": 10, "t": 10, "b": 10})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Score a candidate site")
    x = st.slider("x", 0.0, float(population.shape[1] - 1), 20.0)
    y = st.slider("y", 0.0, float(population.shape[0] - 1), 20.0)
    size = st.select_slider("Store size (sqm)", [600, 900, 1200, 1800], value=1200)
    if st.button("Score site", type="primary"):
        rs = httpx.post(f"{API_URL}/score-site", json={"x": x, "y": y, "size_sqm": size}, timeout=60)
        if rs.status_code == 200:
            s = rs.json()
            st.metric("Captured demand", f"{s['captured_demand']:,.0f}")
            st.metric("Net new (after cannibalization)", f"{s['net_new_demand']:,.0f}")
            st.metric("Cannibalization rate", f"{s['cannibalization_rate']:.0%}")

    st.subheader("Best open sites")
    rb = httpx.get(f"{API_URL}/best-sites", params={"top_k": 5}, timeout=300)
    if rb.status_code == 200:
        for site in rb.json():
            st.markdown(
                f"- ({site['x']:.0f},{site['y']:.0f}) net new **{site['net_new_demand']:,.0f}** "
                f"(cannibalization {site['cannibalization_rate']:.0%})"
            )
