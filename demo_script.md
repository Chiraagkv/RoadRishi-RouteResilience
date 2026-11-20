# RoadRishi — Demo Script (~10 minutes)

> **Audience:** reviewers with an ML / geospatial background
> **Goal:** show the pipeline end-to-end: segmentation → road graph → healing → resilience → routing
> **Rule:** every number you quote must come from the screen or from `training/checkpoints/metrics.json`.

---

## Before you start

```
Terminal 1:  cd backend && ../venv/bin/python main.py     → http://localhost:8000
Terminal 2:  cd frontend && npm run dev                   → http://localhost:5173
```

Top-bar badge: `Live Model` (green) = `training/checkpoints/roadrishi_finetuned.pth` is loaded;
`Sim Mode` (grey) = no checkpoint, live upload is disabled.

Have ready: one DeepGlobe validation tile (`*_sat.jpg`) and its mask (`*_mask.png`).

---

## 1. Introduction (1 min)

> "RoadRishi takes a satellite tile, segments the roads, turns the mask into a road graph,
> repairs gaps caused by occlusions, and then asks: which intersections matter most, and which
> route between two points is most likely to stay passable after a disaster?"

## 2. Segmentation results (1 min) — Simulation mode

Point to the metric cards:
> "SegFormer MiT-B3 fine-tuned on DeepGlobe reaches 0.71 mIoU on a held-out 10 % split. That's
> the mean of road and background IoU — road IoU alone is about 0.45, which is the number I'd
> push on next."

Be explicit that the map in Simulation mode is an illustrative demo scene.

## 3. Live model (3 min)

1. Data Source → **Live Model**, upload the tile and its ground-truth mask, **Run Live Segmentation**.
2. Point to the Live Result panel:
   > "The mask is skeletonised into a graph. Components went from N to M after healing —
   > healing links dangling road ends using distance, direction, and how much road
   > probability lies along the gap. Kruskal accepts links that join components; a second
   > pass closes gaps inside a component when the detour around them is much longer."
3. Metric cards now say **Measured** — real IoU / Dice against the uploaded mask.
4. Routes: white dotted = shortest, orange = risk-adjusted.
   > "Each edge has an existence probability. The risk-adjusted route minimises length plus
   > λ times the negative log of that probability, i.e. it trades distance for the chance
   > that every edge on the route is passable."
5. Optional: enter the tile's bounds to conflate with OpenStreetMap — OSM roads are coloured by
   how strongly the model sees them (green seen, red unseen: occluded, flooded or missing).

## 4. Real OSM network (3 min)

1. Data Source → **Real OSM**, Bengaluru, Disruption **Bridge Attack**, 30 %, **Load OSM Graph**.
   (If OSM is unreachable the toast says *Synthetic grid* — say so.)
2. > "Bridge attack removes the highest edge-betweenness roads — the targeted worst case.
   > The green dashed edges are the repair plan: Kruskal over the damaged roads, i.e. the
   > minimum total repair length that reconnects the network."
3. Point to RI disrupted → repaired and MC E[LCC] / P(connected):
   > "RI combines the largest-connected-component ratio with the Fiedler value. The Monte Carlo
   > numbers come from sampling road failures at the hazard rate and repairs at their
   > completion probability."
4. Click two nodes → *Set origin* / *Set destination*; drag the **λ** slider and watch the
   risk-adjusted route change.

## 5. Risk quadrants (1 min) — on the map

Node colours on the map (Live / Real OSM modes) are the real quadrants; click a red node:
> "Each node's betweenness is computed across the Monte Carlo samples. High mean and high σ
> is CRITICAL: important and fragile — that's where you'd re-task a satellite first."

(The Graph Analysis and Resilience tabs are still illustrative demo data — don't present them
as results.)

## 6. Close (1 min)

> "What I'd do next: report and improve road IoU, calibrate the edge probabilities against
> real outcomes, and add APLS on ground-truth road graphs."

---

## Likely questions

| Question | Honest answer |
|---|---|
| What data? | DeepGlobe Road Extraction, 6,226 labelled RGB tiles at 0.5 m; seeded 90/10 split. |
| Why mIoU and not road IoU? | mIoU is the common headline; road IoU (~0.45) is the stricter number and is logged separately. |
| Multispectral / Indian sensors? | Not yet — the model is RGB-only. LISS-IV / Sentinel-2 are future work. |
| Where do edge probabilities come from? | Heuristics: model confidence on detected roads, mean road probability across a healed gap, an assumed repair-completion probability, or mask support for OSM roads. Not calibrated. |
| Why Monte Carlo? | Edges are uncertain; sampling gives a distribution of betweenness per node, separating critical-and-fragile from critical-and-stable. |
| What's the Fiedler value? | Second-smallest eigenvalue of the graph Laplacian — algebraic connectivity; larger = harder to cut apart. |
| Is it tested? | pytest: graph extraction on known shapes, MST optimality of repairs, Monte Carlo vs analytic probabilities, routing on hand-built graphs, API endpoints, and crash-resume of training. |
