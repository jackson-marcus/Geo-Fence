"""Streamlit demo: population heatmap, site scorer, and the live store network."""

from __future__ import annotations

import os

import httpx
import numpy as np
import plotly.graph_objects as go
import streamlit as st

API_URL = os.environ.get("GEOFENCE_API_URL", "http://localhost:8290")

st.set_page_config(page_title="geofence", page_icon="📍", layout="wide")
st.title("📍 geofence")
st.caption("Huff gravity model: trade areas, site scoring, cannibalization, sequential rollout")


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
net = httpx.get(f"{API_URL}/network", timeout=10).json()

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
    m1, m2, m3 = st.columns(3)
    m1.metric("Stores", net["n_stores"])
    m2.metric("Covered population", f"{net['coverage_fraction']:.1%}")
    m3.metric("Network events", f"{net['seq']} ({net['events_rejected']} rejected)")
    if net["seq"] and st.button("Reset network to the generated city"):
        httpx.post(f"{API_URL}/network/reset", timeout=10)
        st.rerun()

with col2:
    st.subheader("Score a candidate site")
    x = st.slider("x", 0.0, float(population.shape[1] - 1), 20.0)
    y = st.slider("y", 0.0, float(population.shape[0] - 1), 20.0)
    size = st.select_slider("Store size (sqm)", [600, 900, 1200, 1800], value=1200)
    rs = httpx.post(f"{API_URL}/score-site", json={"x": x, "y": y, "size_sqm": size}, timeout=60)
    if rs.status_code == 200:
        s = rs.json()
        st.metric("Captured demand", f"{s['captured_demand']:,.0f}")
        st.metric("Net new (after cannibalization)", f"{s['net_new_demand']:,.0f}")
        st.metric("Cannibalization rate", f"{s['cannibalization_rate']:.0%}")
        if st.button("Open a store here", type="primary"):
            ev = httpx.post(
                f"{API_URL}/network/events",
                json={"kind": "open", "x": x, "y": y, "size_sqm": size, "note": "opened from UI"},
                timeout=60,
            )
            if ev.status_code == 200:
                st.rerun()
            st.error(ev.json().get("reason", ev.text))

    st.subheader("Expansion plan")
    n = st.slider("Stores to open", 1, 6, 3)
    plan = httpx.post(f"{API_URL}/network/rollout", json={"n_stores": n}, timeout=300).json()
    for step in plan["plan"]:
        st.markdown(
            f"{step['step']}. ({step['x']:.0f},{step['y']:.0f}) net new "
            f"**{step['net_new_demand']:,.0f}** (cannibalization {step['cannibalization_rate']:.0%}), "
            f"cumulative {step['cumulative_gain']:,.0f}"
        )
    if len(plan["plan"]) < n:
        st.caption("Stopped early: no candidate cell still reaches uncovered population.")
    static = plan["static_top_k"]
    st.caption(
        f"Opening the top-{n} of a single sweep together would realise "
        f"{static['realized_gain']:,.0f}, not the {static['claimed_gain']:,.0f} their "
        f"individual scores add up to; sequential re-sweeps realise {plan['sequential_gain']:,.0f}."
    )
    if st.button("Commit this plan"):
        httpx.post(f"{API_URL}/network/rollout", json={"n_stores": n, "commit": True}, timeout=300)
        st.rerun()
