# Dataset Guide — RoadRishi RouteResilience

This document covers the datasets RoadRishi uses today and the ones planned for future work.

> **Current state:** the model is trained and evaluated on **DeepGlobe (RGB) only**. LISS-IV, Sentinel-2
> and Cartosat-3 are *not* supported yet — there are no loaders for them; the sections below describe
> how to obtain them for future work.

---

## Quick Reference

| Dataset | Role | Resolution | Size | Status | Loader |
|---|---|---|---|---|---|
| DeepGlobe Road | Training + validation (binary seg) | 0.5 m | ~3.8 GB | ✅ Used | `DeepGlobeDataset` (training script) |
| OSM Road Vectors | Road graph, conflation with masks | Vector | per area | ✅ Used | `osm_graph.py` |
| Resourcesat LISS-IV | Indian-terrain fine-tune | 5.8 m | Varies | ⏳ Future work | — |
| Sentinel-2 L2A | Multi-resolution fine-tune | 10 m | ~800 MB/scene | ⏳ Future work | — |
| Cartosat-3 | Evaluation | ~0.3 m | — | ⏳ Future work | — |

---

## Directory Structure

```
data/
├── deepglobe/
│   ├── train/
│   │   ├── <id>_sat.jpg      ← satellite image
│   │   └── <id>_mask.png     ← binary road mask
│   └── valid/
└── osm/
    ├── bengaluru_drive_1500m.graphml   ← auto-generated cache (osm_graph.py)
    └── ...
```

---

## Dataset 1 — DeepGlobe Road Extraction (PRIMARY)

**Where to get it:**
- Kaggle: https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset
- The dataset is already attached when you "Copy & Edit" the training notebook on Kaggle

**Format:** JPG satellite images + PNG binary masks (white = road)

**Download to local (optional):**
```bash
pip install kaggle
kaggle datasets download -d balraj98/deepglobe-road-extraction-dataset -p data/deepglobe --unzip
```

---

## Dataset 2 — OSM Road Vectors (AUTO-FETCHED)

**No manual download needed.** The `osm_graph.py` module fetches this automatically via `osmnx`:

```python
from backend.osm_graph import OSMGraphLoader
loader = OSMGraphLoader()
G = loader.fetch_city_graph("Bengaluru", dist=1500)  # ~1.5 km around the centre, cached
```

Cached as `data/osm/bengaluru_drive_1500m.graphml` after the first run. `dist=None` fetches the
whole city (~150k nodes — too large for repeated betweenness centrality).

---

## Dataset 3 — Resourcesat LISS-IV (FUTURE WORK — not supported yet)

**Where to get it:**
- Register (free): https://bhuvan.nrsc.gov.in/
- Navigate to: Bhuvan Thematic Services → LISS-IV → select Karnataka AOI
- Download as GeoTIFF, place in `data/lissiv/`

**Bands:** Band 2 (Green), Band 3 (Red), Band 4 (NIR) at 5.8 m resolution

---

## Dataset 4 — Sentinel-2 L2A (FUTURE WORK — not supported yet)

**Where to get it:**
- Register (free): https://dataspace.copernicus.eu/
- Search for Karnataka tiles (UTM zone 43N, tile T43PGR covers Bengaluru)
- Download `.SAFE` folder, place in `data/sentinel2/`

**Bands used:** B02 (Blue), B03 (Green), B04 (Red), B08 (NIR) at 10 m

---

## Dataset 5 — Cartosat-3 (FUTURE WORK — not supported yet)

- ~0.3 m resolution panchromatic + multispectral
- Would need a loader and band handling; the current model expects 3-band RGB

---

## Notes

- The training script (`training/kaggle_segformer_train.py`) uses **DeepGlobe only**. Supporting the
  multispectral sensors above would need new loaders and a model input change (RGB → N bands).
- OSM vectors are fetched automatically — no account needed.
- After training on Kaggle, download `roadrishi_finetuned.pth` and place it at `training/checkpoints/roadrishi_finetuned.pth` to enable live inference mode.
