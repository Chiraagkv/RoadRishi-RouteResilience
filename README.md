# RoadRishi

**RoadRishi** is a geospatial pipeline for post-disaster road-network analysis. It segments
roads from satellite imagery with a fine-tuned SegFormer, turns the predicted mask into a
road graph, heals gaps left by occlusions (tree canopy, shadows), and measures how resilient
the network is — including which intersections are most critical and which route between two
points is most likely to stay passable.

```
satellite tile ──► SegFormer MiT-B3 ──► road probability map
                                             │
                     skeletonise + graph extraction (mask_to_graph.py)
                                             │
             gap healing: Kruskal / Union-Find + loop closing (healing.py)
                                             │
     optional: conflate with the OpenStreetMap roads of the tile footprint (osm_graph.py)
                                             │
  Monte Carlo edge-drop ──► betweenness-centrality mean / σ ──► risk quadrants (resilience.py)
                                             │
             shortest vs risk-adjusted route (routing.py) ──► React + Leaflet dashboard
```

---

## Results

SegFormer MiT-B3 (ImageNet/ADE20K-pretrained) fine-tuned on the labelled DeepGlobe Road
Extraction set (6,226 tiles, 0.5 m/px), 90/10 seeded train/validation split, 384 px input,
2× T4 GPUs:

| Metric (held-out validation, best epoch = 3) | Value |
|---|---|
| mIoU (mean of road IoU and background IoU) | **0.712** |
| Dice / F1 (road class, averaged over validation batches) | 0.629 |

Road-class IoU alone is roughly 0.43–0.46 (0.43 was logged directly in a later run at mIoU 0.696;
the 0.712 run predates separate road-IoU logging, and Dice 0.629 implies ≈ 0.46). Background IoU
is ~0.95+ on DeepGlobe and inflates mIoU, so the current script reports road IoU separately and
computes every metric from dataset-level pixel counts. Tiles are downsampled from 2448 px to
384 px, so the effective ground resolution during evaluation is ~3.2 m/px, not 0.5 m.

The training script writes `metrics.json` (held-out evaluation of the best checkpoint),
`history.json`, `curves.png` and `predictions_grid.png` — those files are the source of truth
for any number quoted.

---

## Repository layout

```
backend/        FastAPI server and the graph pipeline
  main.py           API: /api/segment-live, /api/osm-pipeline, /api/pipeline (simulation)
  model.py          SegFormer wrapper (live checkpoint or simulation fallback)
  mask_to_graph.py  probability map → skeleton → road graph
  healing.py        gap healing (Kruskal/Union-Find) and OSM repair planning (MST)
  resilience.py     Monte Carlo edge-drop, betweenness centrality, Fiedler value, RI
  routing.py        shortest vs risk-adjusted (log-reliability) routing
  osm_graph.py      OSM download (osmnx), disruption simulation, mask ↔ OSM conflation
training/
  kaggle_segformer_train.py   fine-tuning script (Kaggle or local), crash-safe resume
preprocessing/  GeoTIFF tiling utilities (rasterio)
frontend/       React 19 + Vite + Tailwind + Leaflet + Recharts dashboard
tests/          pytest suites (training smoke tests, graph pipeline, API)
```

---

## Setup

### Backend

```bash
python -m venv venv
venv/bin/pip install -r backend/requirements.txt
venv/bin/pip install torch transformers osmnx      # live inference + real OSM graphs
cd backend && ../venv/bin/python main.py           # http://localhost:8000  (docs: /docs)
```

Without a checkpoint the backend runs in **simulation mode**. To enable live inference, place
the trained `roadrishi_finetuned.pth` in `training/checkpoints/`.

### Frontend

```bash
cd frontend
npm install
npm run dev                                        # http://localhost:5173
```

### Training

On Kaggle (GPU T4 ×2, DeepGlobe attached) use **Save Version → Save & Run All**: a committed
run keeps going when the browser tab sleeps. Locally (e.g. a 6 GB GPU):

```bash
venv/bin/pip install -r training/requirements_training.txt
ROADRISHI_DATA_ROOT=data/deepglobe/train ROADRISHI_BATCH=4 ROADRISHI_ACCUM=4 \
    venv/bin/python training/kaggle_segformer_train.py
```

Any config value can be overridden with `ROADRISHI_<NAME>` environment variables (see the top
of the script). The script checkpoints mid-epoch (`latest.pth`), resumes exactly where it
stopped after a crash, Ctrl-C or Kaggle time limit, and can import a previous Kaggle run's
output attached as an input.

### Tests

```bash
venv/bin/pip install pytest httpx
venv/bin/python -m pytest tests -v                 # ~35 s, CPU only, no network
ROADRISHI_NETWORK_TESTS=1 venv/bin/python -m pytest tests -k network   # real OSM download
```

The training tests run the real script on a tiny synthetic dataset and check that a hard crash
or graceful stop mid-epoch resumes to weights bit-identical with an uninterrupted run.

---

## Dashboard modes

| Mode | What it shows |
|---|---|
| **Simulation** | Illustrative demo scene (synthetic graph, nodes, healing log). The headline mIoU / Dice cards show the real validation numbers above. |
| **Real OSM** | OSM drive network within ~1.5 km of the city centre (synthetic grid fallback if OSM is unreachable). Simulated disruption → minimum-length repair plan → Monte Carlo resilience → shortest vs risk-adjusted route. Click nodes to choose origin / destination. |
| **Live Model** | Upload a tile: real SegFormer inference → extracted + healed road graph → risk quadrants → routes. Optional ground-truth mask gives real IoU / Dice; optional tile bounds add OSM conflation (OSM roads coloured by how strongly the model sees them). |

Segmentation metrics are only shown when they are actually measured; otherwise the cards show "—".

---

## Methods in brief

- **Loss:** cross-entropy with class weights (background 1, road 4) + Dice on the road class.
- **Augmentation:** random resized crop, flips, 90° rotation, colour jitter, Gaussian noise/blur,
  elastic transform, grid distortion.
- **Graph extraction:** threshold → remove specks / fill pin-holes → skeletonise → junction and
  endpoint nodes → contract degree-2 chains, prune short spurs → resample every 40 px.
- **Healing:** candidate links between dangling ends scored by
  `C = w1·distance + w2·misalignment + w3·∫(1 − P_road) + w4·(1 − pe)`; Kruskal accepts links
  that join different components, then a loop-closing pass repairs gaps inside a component when
  the detour around them is ≥ 3× the gap.
- **Repair planning (OSM):** Kruskal over the damaged roads → the minimum total repair length that
  reconnects the network (a minimum spanning forest over components).
- **Monte Carlo:** each edge survives with its existence probability `pe`; per-node betweenness
  centrality mean and σ over the samples define CRITICAL / RELIABLE / UNCERTAIN / SAFE quadrants.
- **Resilience Index:** `RI = 0.6·(LCC_post / LCC_pre) + 0.4·(λ₂ ratio)` with λ₂ the Fiedler value.
- **Routing:** Dijkstra on `length + λ·L_ref·(−ln pe)`, i.e. trading distance against the
  probability that every edge on the route is passable.

The edge-existence probabilities `pe` (from model confidence, gap evidence, repair likelihood or
mask support) are documented heuristics, not calibrated probabilities.

---

## Limitations / future work

- The model is trained on RGB DeepGlobe imagery only. Multispectral sensors (Resourcesat LISS-IV,
  Sentinel-2, Cartosat-3) are not yet supported — see `DATASET_GUIDE.md`.
- No APLS or occlusion-specific evaluation yet (needs ground-truth road graphs / occlusion labels).
- `pe` values are heuristic; calibrating them against labelled outcomes is open work.
