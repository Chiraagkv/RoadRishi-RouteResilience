"""
mask_to_graph.py — turn a road probability map into a spatial road graph.

Pipeline:
  1. threshold → binary road mask, drop specks and fill pin-holes
  2. morphological skeleton (1-px centrelines)
  3. pixel graph: endpoints (1 neighbour) and junctions (≥3 neighbours) become nodes,
     the 1-px chains between them become edges with their polyline length
  4. clean-up: contract degree-2 nodes (staircase artefacts), prune short spurs
  5. resample every edge with a node every `node_spacing_px` along its polyline so the
     graph follows curved roads (and matches the granularity healing.py expects)

Each edge carries:
  weight     — polyline length in pixels
  confidence — mean road probability along the edge (used as edge-existence pe)

Node positions are (row, col) = (y, x) floats, the convention used by healing.py.
"""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

_EIGHT = np.ones((3, 3), dtype=bool)
_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def clean_mask(mask: np.ndarray, min_area_px: int = 64, max_hole_px: int = 32) -> np.ndarray:
    """Remove connected blobs smaller than min_area_px and fill holes smaller than max_hole_px."""
    mask = mask.astype(bool)
    lab, n = ndimage.label(mask, structure=_EIGHT)
    if n:
        sizes = np.bincount(lab.ravel())
        keep = sizes >= min_area_px
        keep[0] = False
        mask = keep[lab]
    # Holes = background components not touching the border
    lab, n = ndimage.label(~mask)
    if n:
        sizes = np.bincount(lab.ravel())
        border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        small = sizes < max_hole_px
        small[border] = False
        small[0] = False
        mask = mask | small[lab]
    return mask


def _skeleton_pixel_graph(skel: np.ndarray):
    """
    Returns (nodes, edges):
      nodes: {node_id: [(r, c), ...]}  pixels belonging to the node (junction clusters >1 px)
      edges: [(node_u, node_v, [(r, c), ...])]  pixel path including both end pixels
    """
    H, W = skel.shape
    nbr = ndimage.convolve(skel.astype(np.uint8), _EIGHT.astype(np.uint8), mode="constant") - 1
    nbr = np.where(skel, nbr, 0)

    key = skel & (nbr != 2)
    # Each pure cycle (all pixels have 2 neighbours) needs one artificial key pixel
    lab, n = ndimage.label(skel, structure=_EIGHT)
    if n:
        has_key = np.zeros(n + 1, dtype=bool)
        has_key[np.unique(lab[key])] = True
        for comp in np.nonzero(~has_key[1:])[0] + 1:
            r, c = np.argwhere(lab == comp)[0]
            key[r, c] = True

    # Cluster adjacent junction/key pixels into one node
    klab, _ = ndimage.label(key, structure=_EIGHT)
    pixel_node = {}
    nodes: dict[int, list[tuple[int, int]]] = {}
    for r, c in zip(*np.nonzero(key)):
        nid = int(klab[r, c])
        pixel_node[(r, c)] = nid
        nodes.setdefault(nid, []).append((int(r), int(c)))

    def neighbours(r, c):
        for dr, dc in _OFFSETS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and skel[rr, cc]:
                yield rr, cc

    edges = []
    visited = set()          # non-key pixels already assigned to an edge
    direct = set()           # key-key adjacencies already emitted
    for start, nid in pixel_node.items():
        for q in neighbours(*start):
            qn = pixel_node.get(q)
            if qn is not None:
                if qn != nid:
                    pair = (min(nid, qn), max(nid, qn), min(start, q), max(start, q))
                    if pair not in direct:
                        direct.add(pair)
                        edges.append((nid, qn, [start, q]))
                continue
            if q in visited:
                continue
            path, prev, cur = [start, q], start, q
            visited.add(q)
            while True:
                nxt = None
                for cand in neighbours(*cur):
                    if cand == prev or cand in visited:
                        continue
                    if cand in pixel_node:
                        # Prefer stepping onto a key pixel (ends the chain)
                        nxt = cand
                        break
                    if nxt is None:
                        nxt = cand
                if nxt is None:
                    # Dead end inside a chain (rare skeleton artefact): close on a key
                    # pixel adjacent to the current one, otherwise drop the chain.
                    back = [k for k in neighbours(*cur) if k in pixel_node and k != prev]
                    if back:
                        path.append(back[0])
                        edges.append((nid, pixel_node[back[0]], path))
                    break
                path.append(nxt)
                if nxt in pixel_node:
                    edges.append((nid, pixel_node[nxt], path))
                    break
                visited.add(nxt)
                prev, cur = cur, nxt
    return nodes, edges


def _polyline_length(path) -> float:
    p = np.asarray(path, dtype=float)
    return float(np.sqrt((np.diff(p, axis=0) ** 2).sum(axis=1)).sum()) if len(p) > 1 else 0.0


def _contract_degree_two(M: nx.MultiGraph):
    """Merge chains through degree-2 nodes into single edges (keeps pure cycles as self-loops)."""
    changed = True
    while changed:
        changed = False
        for n in list(M.nodes()):
            if n not in M or M.degree(n) != 2 or M.number_of_edges(n, n):
                continue
            (_, a, ka, da), (_, b, kb, db) = list(M.edges(n, keys=True, data=True))
            pa = da["path"] if da["path"][-1] == M.nodes[n]["anchor"] else da["path"][::-1]
            pb = db["path"] if db["path"][0] == M.nodes[n]["anchor"] else db["path"][::-1]
            # Paths meet at n's anchor pixel (or adjacent pixels of a junction cluster)
            merged = pa + pb[1:] if pa[-1] == pb[0] else pa + pb
            M.remove_node(n)
            M.add_edge(a, b, path=merged)
            changed = True


def extract_road_graph(
    prob_map: np.ndarray,
    threshold: float = 0.5,
    min_area_px: int = 64,
    min_spur_px: float = 15.0,
    node_spacing_px: float = 40.0,
):
    """
    Converts a road probability map (H, W) in [0, 1] into (G, pos):
      G   — nx.Graph, nodes "n_<k>", edge attrs weight (px) and confidence (mean prob)
      pos — {node: (y, x)}
    """
    prob_map = np.asarray(prob_map, dtype=np.float32)
    mask = clean_mask(prob_map >= threshold, min_area_px=min_area_px)
    skel = skeletonize(mask)
    pnodes, pedges = _skeleton_pixel_graph(skel)

    M = nx.MultiGraph()
    for nid, pix in pnodes.items():
        arr = np.asarray(pix, dtype=float)
        centre = arr.mean(axis=0)
        anchor = min(pix, key=lambda p: (p[0] - centre[0]) ** 2 + (p[1] - centre[1]) ** 2)
        M.add_node(nid, pos=(float(centre[0]), float(centre[1])), anchor=anchor)
    for u, v, path in pedges:
        # Normalise each path to start/end on the nodes' anchor pixels
        path = list(path)
        if path[0] != M.nodes[u]["anchor"]:
            path = [M.nodes[u]["anchor"]] + path
        if path[-1] != M.nodes[v]["anchor"]:
            path = path + [M.nodes[v]["anchor"]]
        M.add_edge(u, v, path=path)

    # Contract staircase artefacts first, then prune spurs (short dead-end branches
    # hanging off a junction), then contract again.
    _contract_degree_two(M)
    for _ in range(2):
        spurs = [(u, v, k) for u, v, k, d in M.edges(keys=True, data=True)
                 if u != v and (M.degree(u) == 1) != (M.degree(v) == 1)
                 and max(M.degree(u), M.degree(v)) >= 3
                 and _polyline_length(d["path"]) < min_spur_px]
        if not spurs:
            break
        for u, v, k in spurs:
            if M.has_edge(u, v, k):
                M.remove_edge(u, v, k)
        M.remove_nodes_from([n for n in list(M.nodes()) if M.degree(n) == 0])
        _contract_degree_two(M)
    M.remove_nodes_from([n for n in list(M.nodes()) if M.degree(n) == 0])

    # Resample into the final simple graph
    G = nx.Graph()
    pos: dict[str, tuple[float, float]] = {}
    ids: dict = {}
    counter = [0]

    def new_node(p):
        nid = f"n_{counter[0]}"
        counter[0] += 1
        pos[nid] = (float(p[0]), float(p[1]))
        G.add_node(nid, pos=pos[nid])
        return nid

    for n, d in M.nodes(data=True):
        ids[n] = new_node(d["pos"])

    H, W = prob_map.shape
    for u, v, d in M.edges(data=True):
        path = np.asarray(d["path"], dtype=float)
        seg = np.sqrt((np.diff(path, axis=0) ** 2).sum(axis=1))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = cum[-1]
        if total < 1.0:
            continue
        steps = max(1, int(round(total / node_spacing_px)))
        if u == v:
            steps = max(steps, 3)            # a cycle needs ≥3 nodes to stay a simple graph
        chain = [ids[u]]
        for s in range(1, steps):
            idx = int(np.searchsorted(cum, total * s / steps))
            chain.append(new_node(path[min(idx, len(path) - 1)]))
        chain.append(ids[v])
        # Split cumulative length between consecutive resampled nodes
        marks = [0] + [int(np.searchsorted(cum, total * s / steps)) for s in range(1, steps)] + [len(path) - 1]
        for a, b, i0, i1 in zip(chain[:-1], chain[1:], marks[:-1], marks[1:]):
            if a == b:
                continue
            pts = path[i0:i1 + 1].astype(int)
            pts[:, 0] = np.clip(pts[:, 0], 0, H - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, W - 1)
            conf = float(prob_map[pts[:, 0], pts[:, 1]].mean()) if len(pts) else threshold
            length = float(cum[i1] - cum[i0]) or math.dist(pos[a], pos[b])
            if G.has_edge(a, b):
                continue
            G.add_edge(a, b, weight=length, confidence=round(conf, 4))
    G.remove_nodes_from([n for n in list(G.nodes()) if G.degree(n) == 0])
    pos = {n: pos[n] for n in G.nodes()}
    return G, pos


def dangling_nodes(G: nx.Graph, pos: dict, shape, border_margin_px: float = 12.0):
    """Degree-1 nodes that are not just roads leaving the image (those touch the border)."""
    H, W = shape
    out = []
    for n in G.nodes():
        if G.degree(n) != 1:
            continue
        y, x = pos[n]
        if border_margin_px <= y < H - border_margin_px and border_margin_px <= x < W - border_margin_px:
            out.append(n)
    return out
