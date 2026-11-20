import io
import os
import tempfile
from fastapi import FastAPI, Query, File, Form, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
import uvicorn
import numpy as np
from PIL import Image

# Import custom pipeline modules
from model import RoadSegFormer
from healing import GraphHealingEngine
from resilience import CriticalityVarianceEngine
import networkx as nx
from healing import plan_repairs
from mask_to_graph import extract_road_graph, dangling_nodes
from osm_graph import (OSMGraphLoader, build_synthetic_city_graph, conflate_mask_with_graph,
                       OSMNX_AVAILABLE)
from routing import plan_routes, pick_far_pair

app = FastAPI(title="RoadRishi Geospatial Intelligence Pipeline", version="1.0.0")

# Enable CORS for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models and engines
model_core = RoadSegFormer()


def _serialize_graph(G, pos_dict, node_stats=None):
    """Serialize a NetworkX graph into the frontend-friendly schema used by the API."""
    nodes_list = []
    for n in G.nodes():
        y, x = pos_dict[n]
        node_stats_map = node_stats.get(n, {}) if node_stats else {}
        nodes_list.append({
            "id": n,
            "y": float(y),
            "x": float(x),
            "quadrant": node_stats_map.get("quadrant", "SAFE"),
            "meanBC": node_stats_map.get("meanBC", 0.0),
            "stdBC": node_stats_map.get("stdBC", 0.0)
        })

    edges_list = []
    for u, v, d in G.edges(data=True):
        edges_list.append({
            "u": u,
            "v": v,
            "u_pos": pos_dict[u],
            "v_pos": pos_dict[v],
            "healed": d.get("healed", False)
        })
    return {"nodes": nodes_list, "edges": edges_list}


@app.get("/api/pipeline")
def run_pipeline(
    temperature: float = Query(1.0, description="Temperature scaling parameter T"),
    w1: float = Query(1.0, description="Heuristic cost distance weight"),
    w2: float = Query(20.0, description="Heuristic cost angle weight"),
    w3: float = Query(5.0, description="Heuristic cost path integral weight"),
    w4: float = Query(10.0, description="Heuristic cost attention prior weight"),
    max_heal_dist: float = Query(120.0, description="Maximum healing search distance"),
    mc_samples: int = Query(30, description="Monte Carlo graph samples")
):
    """
    Executes the entire RoadRishi pipeline:
    1. Simulates multi-spectral scene image generation and SegFormer model logits.
    2. Runs model calibration with Temperature Scaling.
    3. Builds the spatial transport graph and fractures it under environmental occlusions.
    4. Heals the gaps using the customized Union-Find & MST cost optimization.
    5. Computes criticality variance stats and global resilience index (RI) metrics.
    """
    # 1. Simulating 1000x1000 pixel area
    width, height = 1000, 1000
    sim_data = model_core.simulate_scene_inference(width, height, temperature=temperature)
    
    # Predefined road layout matches model.py lines
    road_lines = [
        ((200, 0), (200, width)),      # Horizontal highway
        ((600, 0), (600, width)),      # Horizontal street
        ((0, 300), (height, 300)),     # Vertical avenue
        ((0, 700), (height, 700)),     # Vertical avenue
        ((0, 0), (height, width)),     # Diagonal arterial
        ((200, 300), (600, 700))       # Connector
    ]

    # 2. Extract and heal graph
    healing_engine = GraphHealingEngine(w1=w1, w2=w2, w3=w3, w4=w4, max_heal_dist=max_heal_dist)
    
    # Build complete pre-disruption graph
    pre_G, pos_dict = healing_engine.build_initial_graph(road_lines, (height, width))
    
    # Fracture graph based on the model's occlusion failure regions
    disrupted_G = healing_engine.fracture_graph(pre_G, pos_dict, sim_data["occlusion_mask"])
    
    # Heal the graph
    healed_G, healed_edges = healing_engine.heal_graph(
        disrupted_G, 
        pos_dict, 
        sim_data["probabilities"], 
        sim_data["attention_priors"]
    )

    # 3. Analyze Resilience and Criticality Variance
    resilience_engine = CriticalityVarianceEngine(num_samples=mc_samples)
    ri_metrics = resilience_engine.compute_resilience_index(pre_G, disrupted_G, healed_G)
    node_stats, top_priorities = resilience_engine.run_monte_carlo(healed_G, healed_edges)

    # Illustrative NDVI histogram for the demo scene (the real model is RGB-only)
    ndvi_hist_bins = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ndvi_hist_counts_road = [0, 0, 5, 12, 45, 120, 85, 30, 10, 2, 0]
    ndvi_hist_counts_canopy = [0, 1, 2, 8, 15, 35, 90, 180, 240, 140, 20]

    resilience_estimate = ri_metrics["RI_healed"]
    fiedler_values = ri_metrics["FiedlerLambda"]
    lcc_values = ri_metrics["LCC"]

    return {
        "scene": {
            "width": width,
            "height": height
        },
        "occlusion_zones": [
            {"center": [400, 400], "radius": 60, "label": "Canopy Blockage A"},
            {"center": [200, 500], "radius": 50, "label": "Canopy Blockage B"},
            {"center": [450, 700], "radius": 45, "label": "Building Shadow C"}
        ],
        "graphs": {
            "pre_disruption": _serialize_graph(pre_G, pos_dict, node_stats),
            "disrupted": _serialize_graph(disrupted_G, pos_dict, node_stats),
            "healed": _serialize_graph(healed_G, pos_dict, node_stats)
        },
        "healed_log": healed_edges,
        # SIMULATED scene: the graph metrics are computed on the synthetic graph; the
        # segmentation numbers are the real best-validation results of the DeepGlobe
        # fine-tune (they do not describe this synthetic scene). Unmeasured metrics are None.
        "simulated": True,
        "metrics": {
            "mIoU": 0.712,
            "dice_score": 0.629,
            "metrics_source": "best validation epoch of the DeepGlobe fine-tune (not this scene)",
            "occlusion_recall": None,
            "relaxed_iou": None,
            "RI_estimate": resilience_estimate,
            "connectivity_gain_pct": ri_metrics["connectivity_gain"],
            "apls_score": None,
            "apls_error_pct": None,
            "LCC": lcc_values,
            "Fiedler": fiedler_values
        },
        "re_tasking_priorities": top_priorities,
        "loss_breakdown": None,
        "ndvi_distribution": {
            "bins": ndvi_hist_bins,
            "road": ndvi_hist_counts_road,
            "canopy": ndvi_hist_counts_canopy
        }
    }


# ---------------------------------------------------------------------------
# /api/segment-live — Real checkpoint inference endpoint
# ---------------------------------------------------------------------------
@app.post("/api/segment-live")
async def segment_live(
    image: UploadFile = File(...),
    mask: UploadFile | None = File(None, description="Optional ground-truth road mask → real IoU"),
    south: float | None = Form(None), west: float | None = Form(None),
    north: float | None = Form(None), east: float | None = Form(None),
    temperature: float = Query(1.0, description="Temperature scaling parameter T"),
    threshold: float = Query(0.5, description="Road probability threshold for graph extraction"),
    w1: float = Query(1.0, description="Heuristic cost distance weight"),
    w2: float = Query(20.0, description="Heuristic cost angle weight"),
    w3: float = Query(5.0, description="Heuristic cost path integral weight"),
    w4: float = Query(10.0, description="Heuristic cost attention prior weight"),
    max_heal_dist: float = Query(120.0, description="Maximum healing search distance (px)"),
    mc_samples: int = Query(30, description="Monte Carlo graph samples"),
    risk_aversion: float = Query(1.0, ge=0.0, description="λ for risk-adjusted routing"),
):
    """
    Runs the real SegFormer checkpoint on an uploaded RGB image, extracts the road graph
    from the predicted mask (skeletonisation), heals gaps (Kruskal/Union-Find), runs Monte
    Carlo edge-drop BC analysis and plans shortest vs risk-adjusted routes.

    Optional:
      * `mask`  — ground-truth road mask for the same tile → real IoU / Dice / P / R.
      * `south/west/north/east` — image footprint (WGS84). The predicted mask is then also
        conflated with the OpenStreetMap road graph of that footprint (needs osmnx + network).
    """
    if not model_core.is_live:
        return {
            "live": False,
            "message": "Real SegFormer checkpoint is not loaded. Place training/checkpoints/roadrishi_finetuned.pth to enable live inference."
        }

    if not image.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="An image file is required.")

    bounds = None
    given = [v is not None for v in (south, west, north, east)]
    if any(given):
        if not all(given) or not (south < north and west < east):
            raise HTTPException(status_code=400, detail="Bounds need south < north and west < east.")
        bounds = (south, west, north, east)

    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(image.filename)[1] or ".png", delete=False) as tmp_file:
        tmp_file.write(await image.read())
        temp_path = tmp_file.name

    try:
        with Image.open(temp_path) as pil_image:
            rgb_image = np.array(pil_image.convert("RGB"), dtype=np.uint8)
        probability_map = predict_probability_map(rgb_image)

        gt = None
        if mask is not None and mask.filename:
            gt_img = Image.open(io.BytesIO(await mask.read())).convert("L")
            if gt_img.size != (rgb_image.shape[1], rgb_image.shape[0]):
                gt_img = gt_img.resize((rgb_image.shape[1], rgb_image.shape[0]), Image.NEAREST)
            gt = np.array(gt_img) > 127

        result = analyse_probability_map(
            probability_map, threshold=threshold, w=(w1, w2, w3, w4),
            max_heal_dist=max_heal_dist, mc_samples=mc_samples,
            risk_aversion=risk_aversion, gt_mask=gt)
        result["live"] = True
        result["geo"] = {"bounds": bounds} if bounds else None
        if bounds:
            try:
                result["osm_conflation"] = conflate_with_osm(
                    probability_map, bounds, mc_samples=mc_samples, risk_aversion=risk_aversion)
            except Exception as e:      # network / osmnx problems must not lose the analysis
                result["osm_conflation"] = {"error": str(e)}
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def predict_probability_map(rgb_image: np.ndarray, tile_size: int = 512, stride: int = 384) -> np.ndarray:
    """Tiled SegFormer inference; overlapping tile predictions are averaged."""
    height, width = rgb_image.shape[:2]
    if height <= tile_size and width <= tile_size:
        return np.clip(np.asarray(model_core.predict_from_image(rgb_image), dtype=np.float32), 0, 1)

    def starts(n):
        s = list(range(0, max(1, n - tile_size + 1), stride))
        if s[-1] + tile_size < n:
            s.append(n - tile_size)
        return s

    coords = [(y, x) for y in starts(height) for x in starts(width)]
    tiles = []
    for y0, x0 in coords:
        t = rgb_image[y0:y0 + tile_size, x0:x0 + tile_size]
        if t.shape[0] < tile_size or t.shape[1] < tile_size:
            t = np.pad(t, ((0, tile_size - t.shape[0]), (0, tile_size - t.shape[1]), (0, 0)), mode="edge")
        tiles.append(t)
    acc = np.zeros((height, width), dtype=np.float32)
    cnt = np.zeros((height, width), dtype=np.float32)
    for (y0, x0), p in zip(coords, model_core.predict_tile_batch(tiles)):
        h, w = min(tile_size, height - y0), min(tile_size, width - x0)
        acc[y0:y0 + h, x0:x0 + w] += np.asarray(p, dtype=np.float32)[:h, :w]
        cnt[y0:y0 + h, x0:x0 + w] += 1
    return np.clip(acc / np.maximum(cnt, 1), 0, 1)


def segmentation_metrics(prob_map: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> dict:
    """Pixel metrics vs a real ground-truth mask (same maths as the training script)."""
    p = prob_map >= threshold
    tp = int((p & gt).sum()); fp = int((p & ~gt).sum())
    fn = int((~p & gt).sum()); tn = int((~p & ~gt).sum())
    d = lambda a, b: a / b if b else 0.0
    iou_road, iou_bg = d(tp, tp + fp + fn), d(tn, tn + fp + fn)
    prec, rec = d(tp, tp + fp), d(tp, tp + fn)
    return {
        "mIoU": round((iou_road + iou_bg) / 2, 4),
        "road_iou": round(iou_road, 4),
        "dice_score": round(d(2 * tp, 2 * tp + fp + fn), 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
    }


def analyse_probability_map(prob_map, threshold=0.5, w=(1.0, 20.0, 5.0, 10.0),
                            max_heal_dist=120.0, mc_samples=30, risk_aversion=1.0,
                            gt_mask=None, seed=0):
    """
    probability map → extracted road graph → healed graph → Monte Carlo BC → routes.
    Pure function of its inputs (used by /api/segment-live and the tests).
    """
    prob_map = np.clip(np.asarray(prob_map, dtype=np.float32), 0.0, 1.0)
    height, width = prob_map.shape

    raw_G, pos = extract_road_graph(prob_map, threshold=threshold)
    # Observed road edges (the model already says "road" here): survival per 100 px rises
    # from 0.90 at the threshold to 1.0 at full confidence, scaled by segment length so a
    # route's reliability does not depend on how finely edges were resampled.
    # Heuristic, not calibrated — healed gaps (evidence pe 0.05–0.95) dominate route risk.
    for _, _, d in raw_G.edges(data=True):
        c = float(np.clip((d.get("confidence", 1.0) - threshold) / max(1e-6, 1.0 - threshold), 0, 1))
        d["pe"] = float((0.90 + 0.10 * c) ** (d["weight"] / 100.0))

    healing_engine = GraphHealingEngine(w1=w[0], w2=w[1], w3=w[2], w4=w[3], max_heal_dist=max_heal_dist)
    candidates = dangling_nodes(raw_G, pos, (height, width))
    healed_G, healed_edges = healing_engine.heal_graph(raw_G, pos, prob_map, [], candidate_nodes=candidates,
                                                       close_loops=True)

    resilience_engine = CriticalityVarianceEngine(num_samples=mc_samples)
    node_stats, top_priorities = resilience_engine.run_monte_carlo(healed_G, healed_edges, seed=seed)
    mc = resilience_engine.last_summary

    def lcc(G):
        return max((len(c) for c in nx.connected_components(G)), default=0)

    routes = None
    o, d_ = pick_far_pair(healed_G, pos)
    if o is not None:
        routes = plan_routes(healed_G, o, d_, risk_aversion=risk_aversion)
        for key in ("shortest", "risk_adjusted"):
            if routes.get(key):
                routes[key]["coords"] = [list(pos[n]) for n in routes[key]["nodes"]]

    metrics = {
        "mIoU": None, "road_iou": None, "dice_score": None, "precision": None, "recall": None,
        "relaxed_iou": None, "occlusion_recall": None, "apls_score": None,
        "RI_estimate": mc["expected_lcc_fraction"],
        "p_fully_connected": mc["p_fully_connected"],
        "components": {"extracted": nx.number_connected_components(raw_G) if len(raw_G) else 0,
                       "healed": nx.number_connected_components(healed_G) if len(healed_G) else 0},
        "LCC": {"original": lcc(raw_G), "disrupted": lcc(raw_G), "healed": lcc(healed_G),
                "gain_pct": round((lcc(healed_G) - lcc(raw_G)) / max(1, lcc(raw_G)) * 100, 1)},
        "connectivity_gain_pct": round((lcc(healed_G) - lcc(raw_G)) / max(1, len(healed_G)) * 100, 1),
        "Fiedler": {"healed": round(resilience_engine.get_fiedler_eigenvalue(healed_G), 4)},
        "road_pixel_fraction": round(float((prob_map >= threshold).mean()), 4),
        "metrics_source": "ground-truth mask" if gt_mask is not None else
                          "none (upload a ground-truth mask to score segmentation)",
    }
    if gt_mask is not None:
        metrics.update(segmentation_metrics(prob_map, gt_mask, threshold))

    return {
        "scene": {"width": width, "height": height},
        # Healed gaps are where the model lost the road (occlusion, shadow, damage)
        "occlusion_zones": [
            {"center": [(e["u_pos"][0] + e["v_pos"][0]) / 2, (e["u_pos"][1] + e["v_pos"][1]) / 2],
             "radius": e["length"] / 2 + 10, "label": f"Gap {i + 1} (pe={e['pe']})"}
            for i, e in enumerate(healed_edges)
        ],
        "graphs": {
            "pre_disruption": _serialize_graph(raw_G, pos, node_stats),
            "disrupted": _serialize_graph(raw_G, pos, node_stats),
            "healed": _serialize_graph(healed_G, pos, node_stats),
        },
        "healed_log": healed_edges,
        "metrics": metrics,
        "monte_carlo": mc,
        "routes": routes,
        "re_tasking_priorities": top_priorities,
        "loss_breakdown": None,
    }


def conflate_with_osm(prob_map, bounds, mc_samples=20, risk_aversion=1.0, loader=None, G_osm=None):
    """Score the OSM roads inside `bounds` by the predicted mask, then MC + routing on them."""
    loader = loader or OSMGraphLoader()
    if G_osm is None:
        G_osm = loader.fetch_bbox_graph(bounds)
    G, pos = loader.to_networkx(G_osm)
    stats = conflate_mask_with_graph(G, prob_map, bounds)
    engine = CriticalityVarianceEngine(num_samples=mc_samples)
    node_stats, top = engine.run_monte_carlo(G, [], seed=0)
    latlon = {n: (G.nodes[n]["lat"], G.nodes[n]["lon"]) for n in G.nodes()}
    o, d_ = pick_far_pair(G, pos)
    routes = plan_routes(G, o, d_, risk_aversion) if o is not None else None
    _attach_coords(routes, latlon)
    return {
        "stats": stats,
        "graph": loader.serialize_graph_for_api(G, pos, node_stats=node_stats),
        "monte_carlo": engine.last_summary,
        "routes": routes,
        "re_tasking_priorities": top,
    }


def _attach_coords(routes, coord_of):
    if not routes:
        return
    for key in ("shortest", "risk_adjusted"):
        if routes.get(key):
            routes[key]["coords"] = [list(coord_of[n]) for n in routes[key]["nodes"]]
            routes[key]["nodes"] = [str(n) for n in routes[key]["nodes"]]
    routes["source"], routes["target"] = str(routes["source"]), str(routes["target"])


# ---------------------------------------------------------------------------
# /api/osm-pipeline — Real OSM city graph endpoint
# ---------------------------------------------------------------------------
@app.get("/api/osm-pipeline")
def run_osm_pipeline(
    city: str = Query("Bengaluru", description="Indian city name"),
    disruption_pct: float = Query(0.3, ge=0.0, le=0.9, description="Fraction of edges to disrupt (0–0.9)"),
    disruption_mode: str = Query("random", description="Disruption mode: random | bridge | cascade | node"),
    mc_samples: int = Query(20, ge=1, le=200, description="Monte Carlo graph samples for BC variance"),
    use_cache: bool = Query(True, description="Use cached GraphML if available"),
    radius_m: float = Query(1500.0, ge=200, le=5000, description="Half-width of the area around the city centre"),
    hazard: float = Query(0.05, ge=0.0, le=0.9, description="Per-road failure probability in Monte Carlo"),
    repair_pe: float = Query(0.7, ge=0.0, le=1.0, description="Probability a planned repair is in place"),
    risk_aversion: float = Query(1.0, ge=0.0, le=50.0, description="λ for risk-adjusted routing"),
    origin_id: str | None = Query(None, description="Route origin node id (default: far pair)"),
    dest_id: str | None = Query(None, description="Route destination node id"),
    seed: int = Query(42),
):
    """
    Runs the RoadRishi pipeline on a REAL OpenStreetMap city graph.

    1. Loads / fetches the OSM road network around the city centre (synthetic grid fallback).
    2. Simulates a disaster using the chosen disruption mode.
    3. Plans the minimum-length set of road repairs that reconnects the network (Kruskal).
    4. Monte Carlo: every road fails with probability `hazard`, repairs with 1 − repair_pe;
       betweenness-centrality mean/std per node → risk quadrants.
    5. Shortest vs risk-adjusted route between two nodes on the repaired network.
    """
    loader = OSMGraphLoader()
    resilience_engine = CriticalityVarianceEngine(num_samples=mc_samples)

    using_real_osm = False
    if OSMNX_AVAILABLE:
        try:
            G_osm = loader.fetch_city_graph(city, use_cache=use_cache, dist=radius_m)
            G_full, pos_dict = loader.to_networkx(G_osm)
            using_real_osm = True
        except Exception as e:
            print(f"[OSM] Failed to load real graph: {e} — falling back to synthetic")
            G_full, pos_dict = build_synthetic_city_graph(city)
    else:
        G_full, pos_dict = build_synthetic_city_graph(city)

    G_disrupted = loader.simulate_disaster(G_full, pct=disruption_pct, mode=disruption_mode, seed=seed)
    G_healed, repairs = plan_repairs(G_full, G_disrupted, pos_dict, repair_pe=repair_pe)
    for _, _, d in G_healed.edges(data=True):
        if not d.get("healed"):
            d["pe"] = 1.0 - hazard

    ri_metrics = resilience_engine.compute_resilience_index(G_full, G_disrupted, G_healed)
    node_stats, top_priorities = resilience_engine.run_monte_carlo(G_healed, repairs, seed=seed)

    graph_full = loader.serialize_graph_for_api(G_full, pos_dict, node_stats=None, city_name=city)
    graph_disrupted = loader.serialize_graph_for_api(G_disrupted, pos_dict, node_stats=node_stats, city_name=city)
    graph_healed = loader.serialize_graph_for_api(G_healed, pos_dict, node_stats=node_stats, city_name=city)
    latlon = {n["id"]: (n["y"], n["x"]) for n in graph_full["nodes"]}

    # Routing (node ids arrive as strings from the frontend)
    by_str = {str(n): n for n in G_healed.nodes()}
    o, d_ = by_str.get(origin_id), by_str.get(dest_id)
    if o is None or d_ is None:
        o, d_ = pick_far_pair(G_healed, pos_dict)
    routes = plan_routes(G_healed, o, d_, risk_aversion=risk_aversion) if o is not None else None
    _attach_coords(routes, {n: latlon[str(n)] for n in G_healed.nodes()})

    repair_log = [{**r, "u": str(r["u"]), "v": str(r["v"]),
                   "u_pos": list(latlon[str(r["u"])]), "v_pos": list(latlon[str(r["v"])])}
                  for r in repairs]

    return {
        "meta": {
            "city": city,
            "using_real_osm": using_real_osm,
            "radius_m": radius_m,
            "disruption_pct": disruption_pct,
            "disruption_mode": disruption_mode,
            "original_nodes": G_full.number_of_nodes(),
            "original_edges": G_full.number_of_edges(),
            "disrupted_edges": G_disrupted.number_of_edges(),
            "repaired_edges": len(repairs),
            "repair_length_m": round(sum(r["length"] for r in repairs), 1),
        },
        "graphs": {
            "pre_disruption": graph_full,
            "disrupted": graph_disrupted,
            "healed": graph_healed,
        },
        "healed_log": repair_log,
        "metrics": {
            "RI_estimate": ri_metrics["RI_healed"],
            "RI_disrupted": ri_metrics["RI_disrupted"],
            "connectivity_gain_pct": ri_metrics["connectivity_gain"],
            "LCC": ri_metrics["LCC"],
            "Fiedler": ri_metrics["FiedlerLambda"],
            "expected_lcc_fraction": resilience_engine.last_summary["expected_lcc_fraction"],
            "p_fully_connected": resilience_engine.last_summary["p_fully_connected"],
            # No imagery in this mode → no segmentation metrics
            "mIoU": None,
            "dice_score": None,
            "occlusion_recall": None,
            "relaxed_iou": None,
            "apls_score": None,
        },
        "monte_carlo": resilience_engine.last_summary,
        "routes": routes,
        "re_tasking_priorities": [{**p, "node_id": str(p["node_id"])} for p in top_priorities],
    }


# ---------------------------------------------------------------------------
# /api/checkpoint-status — Reports whether real model weights are loaded
# ---------------------------------------------------------------------------
@app.get("/api/checkpoint-status")
def checkpoint_status():
    """
    Returns whether the SegFormer real checkpoint is loaded
    or the system is running in simulation mode.
    """
    status_info = model_core.get_status()  # renamed to avoid shadowing fastapi.status
    return {
        "mode": status_info["mode"],
        "checkpoint_exists": model_core.is_live,
        "osmnx_available": OSMNX_AVAILABLE,
        "device": status_info["device"],
        "message": (
            "Real SegFormer checkpoint loaded — live inference active."
            if model_core.is_live
            else "Running in simulation mode. Place roadrishi_finetuned.pth in training/checkpoints/ to enable live inference."
        ),
    }



# Serving the static frontend code
frontend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/frontend", StaticFiles(directory=frontend_path), name="frontend")

@app.get("/")
def read_root():
    """Redirect root to the beautiful dashboard index.html."""
    return RedirectResponse(url="/frontend/index.html")

if __name__ == "__main__":
    # Boot the uvicorn web server
    print("\n" + "=" * 60)
    print("      Starting RoadRishi Geospatial Intelligence Server")
    print("      Open http://localhost:8000 in your browser to access UI")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
