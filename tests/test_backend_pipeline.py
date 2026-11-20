"""
Tests for the graph side of RoadRishi:
  mask → graph extraction, healing, Kruskal repair planning, Monte Carlo edge-drop,
  risk-adjusted routing, mask ↔ OSM conflation, and the two FastAPI endpoints.

Run:  venv/bin/python -m pytest tests/test_backend_pipeline.py -v
Optional real-network test:  ROADRISHI_NETWORK_TESTS=1 venv/bin/python -m pytest tests -k network
"""
import io
import math
import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from PIL import Image
from skimage.draw import line as draw_line

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from healing import GraphHealingEngine, plan_repairs           # noqa: E402
from mask_to_graph import extract_road_graph, dangling_nodes   # noqa: E402
from resilience import CriticalityVarianceEngine               # noqa: E402
from routing import plan_routes, pick_far_pair, nearest_node   # noqa: E402
import osm_graph                                               # noqa: E402
from osm_graph import (OSMGraphLoader, build_synthetic_city_graph,   # noqa: E402
                       conflate_mask_with_graph, latlon_to_pixel)


# ── helpers ───────────────────────────────────────────────────────────────────
def road(m, r0, c0, r1, c1, half_width=4, value=1.0):
    rr, cc = draw_line(r0, c0, r1, c1)
    for r, c in zip(rr, cc):
        m[max(0, r - half_width):r + half_width + 1, max(0, c - half_width):c + half_width + 1] = value
    return m


def total_length(G):
    return sum(d["weight"] for _, _, d in G.edges(data=True))


# ── mask → graph ──────────────────────────────────────────────────────────────
class TestExtraction:
    def test_cross_gives_one_junction(self):
        m = road(road(np.zeros((400, 400), np.float32), 200, 0, 200, 399), 0, 200, 399, 200)
        G, pos = extract_road_graph(m)
        assert nx.is_connected(G)
        degs = sorted(d for _, d in G.degree())
        assert degs[-1] == 4 and degs[-2] <= 2          # exactly one 4-way junction
        assert total_length(G) == pytest.approx(800, rel=0.05)

    def test_gap_splits_graph_and_exposes_dangling_ends(self):
        m = road(road(np.zeros((400, 400), np.float32), 200, 0, 200, 170), 200, 230, 200, 399)
        G, pos = extract_road_graph(m)
        assert nx.number_connected_components(G) == 2
        ends = dangling_nodes(G, pos, m.shape)
        xs = sorted(pos[n][1] for n in ends)
        assert len(ends) == 2 and xs[0] == pytest.approx(170, abs=6) and xs[1] == pytest.approx(230, abs=6)

    def test_ring_is_a_cycle(self):
        yy, xx = np.ogrid[:400, :400]
        r = np.hypot(yy - 200, xx - 200)
        m = ((r >= 110) & (r <= 120)).astype(np.float32)
        G, _ = extract_road_graph(m)
        assert nx.is_connected(G) and len(nx.cycle_basis(G)) == 1
        assert total_length(G) == pytest.approx(2 * math.pi * 115, rel=0.08)

    def test_curve_is_followed(self):
        m = np.zeros((400, 400), np.float32)
        t = np.linspace(0, math.pi / 2, 400)
        for a in t:
            road(m, int(50 + 300 * math.sin(a)), int(50 + 300 * (1 - math.cos(a))),
                 int(50 + 300 * math.sin(a)), int(50 + 300 * (1 - math.cos(a))), 3)
        G, pos = extract_road_graph(m)
        assert nx.is_connected(G)
        assert total_length(G) == pytest.approx(300 * math.pi / 2, rel=0.08)
        assert G.number_of_nodes() >= 10                 # resampled along the arc

    def test_noise_and_empty(self):
        m = np.zeros((200, 200), np.float32)
        m[50:53, 50:53] = 1                              # 9-px speck
        G, _ = extract_road_graph(m)
        assert G.number_of_nodes() == 0
        G, _ = extract_road_graph(np.zeros((64, 64), np.float32))
        assert G.number_of_nodes() == 0

    def test_confidence_is_mean_probability(self):
        m = road(np.zeros((200, 200), np.float32), 100, 0, 100, 199, value=0.8)
        G, _ = extract_road_graph(m)
        confs = [d["confidence"] for *_, d in G.edges(data=True)]
        assert all(c == pytest.approx(0.8, abs=1e-3) for c in confs)


# ── healing ───────────────────────────────────────────────────────────────────
class TestHealing:
    def test_heals_occluded_gap_with_evidence_pe(self):
        m = road(np.zeros((400, 400), np.float32), 200, 171, 200, 229, value=0.3)  # "canopy"
        m = road(road(m, 200, 0, 200, 170), 200, 230, 200, 399)
        G, pos = extract_road_graph(m, threshold=0.5)
        healed, log = GraphHealingEngine().heal_graph(
            G, pos, m, [], candidate_nodes=dangling_nodes(G, pos, m.shape))
        assert nx.is_connected(healed) and len(log) == 1
        assert log[0]["length"] == pytest.approx(60, abs=10)
        assert 0.2 < log[0]["pe"] < 0.4                  # ≈ mean prob along the gap

    def test_does_not_bridge_beyond_max_distance(self):
        m = road(road(np.zeros((400, 600), np.float32), 200, 0, 200, 150), 200, 450, 200, 599)
        G, pos = extract_road_graph(m)
        healed, log = GraphHealingEngine(max_heal_dist=120).heal_graph(
            G, pos, m, [], candidate_nodes=dangling_nodes(G, pos, m.shape))
        assert log == [] and nx.number_connected_components(healed) == 2

    def test_repair_plan_is_minimum_spanning_forest(self):
        G, pos = build_synthetic_city_graph("Bengaluru", 8, 8)
        Gd = OSMGraphLoader.__new__(OSMGraphLoader).simulate_disaster(G, pct=0.45, mode="random", seed=3)
        healed, repairs = plan_repairs(G, Gd, pos, repair_pe=0.6)
        assert nx.number_connected_components(healed) == 1
        assert all(healed.edges[r["u"], r["v"]]["healed"] for r in repairs)
        assert all(healed.edges[r["u"], r["v"]]["pe"] == 0.6 for r in repairs)
        assert set(healed.nodes()) == set(G.nodes())       # isolated junctions restored too
        # Optimality: equals the MST of the component graph built from damaged roads
        Gd = Gd.copy()
        Gd.add_nodes_from(G.nodes())                       # isolates are 1-node components
        comp_of = {n: i for i, c in enumerate(nx.connected_components(Gd)) for n in c}
        CG = nx.Graph()
        for u, v, d in G.edges(data=True):
            if not Gd.has_edge(u, v) and comp_of[u] != comp_of[v]:
                a, b, w = comp_of[u], comp_of[v], d["weight"]
                if not CG.has_edge(a, b) or CG.edges[a, b]["weight"] > w:
                    CG.add_edge(a, b, weight=w)
        mst = sum(d["weight"] for *_, d in nx.minimum_spanning_tree(CG).edges(data=True))
        assert sum(r["length"] for r in repairs) == pytest.approx(mst)
        assert len(repairs) == nx.number_connected_components(Gd) - 1


# ── Monte Carlo ───────────────────────────────────────────────────────────────
class TestMonteCarlo:
    def test_edge_drop_probabilities(self):
        G = nx.path_graph(3)                       # 0 - 1 - 2
        nx.set_edge_attributes(G, 1.0, "weight")
        G.edges[0, 1]["pe"] = 0.0                  # always dropped
        eng = CriticalityVarianceEngine(num_samples=50)
        eng.run_monte_carlo(G, seed=0)
        assert eng.last_summary["expected_lcc_fraction"] == pytest.approx(2 / 3, abs=1e-3)
        assert eng.last_summary["p_fully_connected"] == 0.0

    def test_hazard_rate_matches_analytic(self):
        G = nx.path_graph(2)
        nx.set_edge_attributes(G, 1.0, "weight")
        eng = CriticalityVarianceEngine(num_samples=2000)
        eng.run_monte_carlo(G, default_pe=0.7, seed=1)
        assert eng.last_summary["p_fully_connected"] == pytest.approx(0.7, abs=0.03)

    def test_reliable_graph_has_zero_bc_variance(self):
        G, _ = build_synthetic_city_graph("Bengaluru", 5, 5)
        eng = CriticalityVarianceEngine(num_samples=5)
        stats, _ = eng.run_monte_carlo(G, seed=0)
        assert all(s["stdBC"] < 1e-12 for s in stats.values())
        assert eng.last_summary["uncertain_edges"] == 0

    def test_uncertain_bridge_creates_bc_variance(self):
        G = nx.barbell_graph(4, 0)                  # bridge 3-4 between two cliques
        nx.set_edge_attributes(G, 1.0, "weight")
        G.edges[3, 4]["pe"] = 0.5
        stats, _ = CriticalityVarianceEngine(num_samples=200).run_monte_carlo(G, seed=0)
        assert stats[3]["stdBC"] > 0.1 and stats[0]["stdBC"] < 1e-9

    def test_large_graph_uses_approximate_bc(self):
        G, _ = build_synthetic_city_graph("Bengaluru", 25, 25)   # 625 nodes
        eng = CriticalityVarianceEngine(num_samples=2)
        eng.run_monte_carlo(G, default_pe=0.95, seed=0)
        assert eng.last_summary["bc_approximate"] is True


# ── routing ───────────────────────────────────────────────────────────────────
class TestRouting:
    @pytest.fixture
    def diamond(self):
        G = nx.Graph()
        G.add_edge("s", "a", weight=1.0, pe=1.0)
        G.add_edge("a", "t", weight=1.0, pe=0.2)   # short but probably blocked
        G.add_edge("s", "b", weight=1.5, pe=1.0)
        G.add_edge("b", "t", weight=1.5, pe=1.0)   # 50 % longer, certain
        return G

    def test_risk_neutral_is_shortest(self, diamond):
        r = plan_routes(diamond, "s", "t", risk_aversion=0.0)
        assert r["shortest"]["nodes"] == ["s", "a", "t"] == r["risk_adjusted"]["nodes"]
        assert r["shortest"]["reliability"] == pytest.approx(0.2)

    def test_risk_averse_detours(self, diamond):
        r = plan_routes(diamond, "s", "t", risk_aversion=1.0)
        assert r["risk_adjusted"]["nodes"] == ["s", "b", "t"]
        assert r["risk_adjusted"]["reliability"] == 1.0
        assert r["detour_pct"] == pytest.approx(50.0)
        assert r["reliability_gain"] == pytest.approx(0.8)

    def test_unreachable(self, diamond):
        diamond.add_node("island")
        r = plan_routes(diamond, "s", "island")
        assert r["shortest"] is None and r["risk_adjusted"] is None

    def test_od_helpers(self):
        G, pos = build_synthetic_city_graph("Bengaluru", 6, 6)
        a, b = pick_far_pair(G, pos)
        assert math.dist(pos[a], pos[b]) == pytest.approx(math.dist(pos["n_0_0"], pos["n_5_5"]))
        assert nearest_node(pos, *pos["n_2_3"]) == "n_2_3"


# ── OSM conflation & coordinates ──────────────────────────────────────────────
BOUNDS = (12.970, 77.590, 12.980, 77.600)      # south, west, north, east


def geo_graph():
    """Three roads: A–B horizontal across the image, A–C vertical, D–E outside the image."""
    G = nx.Graph()
    for n, (lat, lon) in {"A": (12.975, 77.592), "B": (12.975, 77.598), "C": (12.971, 77.592),
                          "D": (12.990, 77.610), "E": (12.991, 77.611)}.items():
        G.add_node(n, lat=lat, lon=lon)
    G.add_edge("A", "B", weight=650.0)
    G.add_edge("A", "C", weight=440.0)
    G.add_edge("D", "E", weight=150.0)
    return G


class TestConflation:
    def test_support_follows_the_mask(self):
        G = geo_graph()
        shape = (500, 500)
        prob = np.zeros(shape, np.float32)
        r0, c0 = latlon_to_pixel(12.975, 77.592, BOUNDS, shape)
        r1, c1 = latlon_to_pixel(12.975, 77.598, BOUNDS, shape)
        road(prob, int(r0), int(c0), int(r1), int(c1), half_width=3, value=0.95)
        stats = conflate_mask_with_graph(G, prob, BOUNDS)
        ab, ac, de = G.edges["A", "B"], G.edges["A", "C"], G.edges["D", "E"]
        assert ab["support"] > 0.9 and ab["pe"] > 0.9            # seen by the model
        assert ac["support"] < 0.1 and ac["pe"] == pytest.approx(0.3, abs=0.05)  # unseen
        assert de["support"] is None and "pe" not in de          # outside the image
        assert stats["edges_scored"] == 2 and stats["edges_unseen"] == 1

    def test_serializer_uses_true_latlon(self):
        G = geo_graph()
        pos = {n: (1_434_000.0 + i, 780_000.0 + i) for i, n in enumerate(G)}   # UTM-like
        out = OSMGraphLoader.__new__(OSMGraphLoader).serialize_graph_for_api(G, pos)
        a = next(n for n in out["nodes"] if n["id"] == "A")
        assert (a["y"], a["x"]) == (12.975, 77.592)

    def test_to_networkx_offline_projection(self):
        """Exercises the real osmnx projection path on a hand-made OSM-style graph."""
        pytest.importorskip("osmnx")
        G_osm = nx.MultiDiGraph(crs="epsg:4326")
        for n, d in geo_graph().nodes(data=True):
            G_osm.add_node(n, x=d["lon"], y=d["lat"])
        G_osm.add_edge("A", "B", length=650.0)
        G_osm.add_edge("B", "A", length=650.0)
        G_osm.add_edge("A", "C", length=440.0)
        G, pos = OSMGraphLoader.__new__(OSMGraphLoader).to_networkx(G_osm)
        assert G.nodes["A"]["lat"] == 12.975 and G.number_of_edges() == 2
        # projected metres: A→B is ~650 m east
        assert math.dist(pos["A"], pos["B"]) == pytest.approx(650, rel=0.02)


# ── API ───────────────────────────────────────────────────────────────────────
@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient
    import main
    monkeypatch.setattr(main, "OSMNX_AVAILABLE", False)    # synthetic graph, no network
    return TestClient(main.app), main


def synthetic_tile():
    """RGB tile + its 'model output': a grid of roads with one occluded segment."""
    prob = np.zeros((512, 512), np.float32)
    for k in (128, 384):
        road(prob, k, 0, k, 511, value=0.9)
        road(prob, 0, k, 511, k, value=0.9)
    prob[120:137, 230:290] = 0.25                         # canopy over one road
    rgb = (np.stack([prob] * 3, -1) * 255).astype(np.uint8)
    return rgb, prob


class TestAPI:
    def test_osm_pipeline_repairs_and_routes(self, client):
        c, _ = client
        r = c.get("/api/osm-pipeline", params={"disruption_pct": 0.4, "mc_samples": 5,
                                               "risk_aversion": 2.0})
        assert r.status_code == 200
        d = r.json()
        assert d["meta"]["repaired_edges"] == len(d["healed_log"]) > 0
        healed_edges = [e for e in d["graphs"]["healed"]["edges"] if e["healed"]]
        assert len(healed_edges) == d["meta"]["repaired_edges"]
        assert d["metrics"]["RI_estimate"] >= d["metrics"]["RI_disrupted"]
        assert d["monte_carlo"]["uncertain_edges"] > 0
        rt = d["routes"]
        assert rt["shortest"] and rt["risk_adjusted"]
        assert rt["risk_adjusted"]["reliability"] >= rt["shortest"]["reliability"]
        assert all(12.9 < lat < 13.1 and 77.5 < lon < 77.7 for lat, lon in rt["shortest"]["coords"])

    def test_osm_pipeline_custom_od(self, client):
        c, _ = client
        r = c.get("/api/osm-pipeline", params={"mc_samples": 2, "origin_id": "n_0_0",
                                               "dest_id": "n_0_3", "disruption_pct": 0.0})
        rt = r.json()["routes"]
        assert rt["source"] == "n_0_0" and rt["target"] == "n_0_3"
        assert rt["shortest"]["length"] == pytest.approx(1500.0)

    def test_segment_live_requires_checkpoint(self, client, monkeypatch):
        c, main = client
        monkeypatch.setattr(main.model_core, "is_live", False)
        r = c.post("/api/segment-live", files={"image": ("t.png", b"x", "image/png")})
        assert r.json()["live"] is False

    def test_segment_live_end_to_end(self, client, monkeypatch):
        c, main = client
        rgb, prob = synthetic_tile()
        monkeypatch.setattr(main.model_core, "is_live", True)
        monkeypatch.setattr(main.model_core, "predict_from_image", lambda img: prob)
        buf = io.BytesIO(); Image.fromarray(rgb).save(buf, format="PNG")
        gt = io.BytesIO(); Image.fromarray(((prob > 0.5) * 255).astype(np.uint8)).save(gt, format="PNG")
        r = c.post("/api/segment-live", params={"mc_samples": 10},
                   files={"image": ("t.png", buf.getvalue(), "image/png"),
                          "mask": ("m.png", gt.getvalue(), "image/png")})
        assert r.status_code == 200, r.text
        d = r.json()
        m = d["metrics"]
        assert m["road_iou"] == 1.0 and m["metrics_source"] == "ground-truth mask"
        assert m["components"]["healed"] == 1
        # The occluded stretch (row 128, cols 230–290) is bridged by a healed edge
        gaps = [e for e in d["healed_log"]
                if abs(e["u_pos"][0] - 128) < 6 and abs(e["v_pos"][0] - 128) < 6
                and min(e["u_pos"][1], e["v_pos"][1]) <= 235 and max(e["u_pos"][1], e["v_pos"][1]) >= 285]
        assert len(gaps) == 1 and gaps[0]["pe"] < 0.5
        assert len(d["occlusion_zones"]) == len(d["healed_log"])
        before = sum(1 for n in d["graphs"]["pre_disruption"]["nodes"])
        assert before > 0
        assert d["routes"]["shortest"]["coords"]
        assert 0 < d["metrics"]["RI_estimate"] <= 1
        assert d["loss_breakdown"] is None                     # no fabricated losses

    def test_segment_live_without_gt_reports_no_fake_metrics(self, client, monkeypatch):
        c, main = client
        rgb, prob = synthetic_tile()
        monkeypatch.setattr(main.model_core, "is_live", True)
        monkeypatch.setattr(main.model_core, "predict_from_image", lambda img: prob)
        buf = io.BytesIO(); Image.fromarray(rgb).save(buf, format="PNG")
        d = c.post("/api/segment-live", params={"mc_samples": 3},
                   files={"image": ("t.png", buf.getvalue(), "image/png")}).json()
        assert d["metrics"]["mIoU"] is None and d["metrics"]["apls_score"] is None

    def test_segment_live_with_bounds_conflates_osm(self, client, monkeypatch):
        pytest.importorskip("osmnx")
        c, main = client
        rgb, prob = synthetic_tile()
        monkeypatch.setattr(main.model_core, "is_live", True)
        monkeypatch.setattr(main.model_core, "predict_from_image", lambda img: prob)
        G_osm = nx.MultiDiGraph(crs="epsg:4326")
        for n, d in geo_graph().nodes(data=True):
            G_osm.add_node(n, x=d["lon"], y=d["lat"])
        for u, v, l in (("A", "B", 650.0), ("A", "C", 440.0), ("D", "E", 150.0)):
            G_osm.add_edge(u, v, length=l)
        monkeypatch.setattr(main.OSMGraphLoader, "fetch_bbox_graph", lambda self, b: G_osm)
        buf = io.BytesIO(); Image.fromarray(rgb).save(buf, format="PNG")
        s, w, n, e = BOUNDS
        d = c.post("/api/segment-live", params={"mc_samples": 3},
                   data={"south": s, "west": w, "north": n, "east": e},
                   files={"image": ("t.png", buf.getvalue(), "image/png")}).json()
        oc = d["osm_conflation"]
        assert "error" not in oc, oc
        assert oc["stats"]["edges_scored"] == 2
        assert d["geo"]["bounds"] == list(BOUNDS)

    def test_bad_bounds_rejected(self, client, monkeypatch):
        c, main = client
        monkeypatch.setattr(main.model_core, "is_live", True)
        r = c.post("/api/segment-live", data={"south": 13.0, "north": 12.0, "west": 77.5, "east": 77.6},
                   files={"image": ("t.png", b"x", "image/png")})
        assert r.status_code == 400

    def test_tiled_inference_averages_overlaps(self, client, monkeypatch):
        _, main = client
        monkeypatch.setattr(main.model_core, "predict_tile_batch",
                            lambda tiles: [np.full(t.shape[:2], t[0, 0, 0] / 255.0) for t in tiles])
        rgb = np.full((900, 1100, 3), 128, np.uint8)
        p = main.predict_probability_map(rgb)
        assert p.shape == (900, 1100) and np.allclose(p, 128 / 255.0)


@pytest.mark.skipif(os.environ.get("ROADRISHI_NETWORK_TESTS") != "1", reason="needs network")
def test_network_real_bengaluru_osm(tmp_path):
    loader = OSMGraphLoader(cache_dir=str(tmp_path))
    G_osm = loader.fetch_city_graph("Bengaluru", use_cache=False, dist=600)
    G, pos = loader.to_networkx(G_osm)
    assert 50 < G.number_of_nodes() < 5000
    lats = [d["lat"] for _, d in G.nodes(data=True)]
    assert all(12.95 < v < 13.0 for v in lats)
