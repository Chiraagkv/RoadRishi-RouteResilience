"""
routing.py — risk-adjusted routing on a road graph with uncertain edges.

Every edge has a length (`weight`) and an existence probability `pe` (1.0 = certainly
passable). For a path P the probability that every edge is passable (independent edges) is

    R(P) = Π pe(e)          ⇔          −ln R(P) = Σ −ln pe(e)

so the risk-adjusted cost

    cost(e) = length(e) + λ · L_ref · (−ln pe(e))

trades distance against log-reliability: λ = 0 gives the plain shortest path, and larger λ
accepts longer detours to avoid edges that may be blocked. L_ref (the median edge length)
puts the penalty in the same units as length, so λ is dimensionless.
"""
from __future__ import annotations

import math

import networkx as nx
import numpy as np

MIN_PE = 1e-3


def _pe(d) -> float:
    pe = d.get("pe", 1.0)
    return float(min(1.0, max(MIN_PE, 1.0 if pe is None else pe)))


def reference_length(G: nx.Graph, weight: str = "weight") -> float:
    lengths = [d.get(weight, 1.0) for _, _, d in G.edges(data=True)]
    return float(np.median(lengths)) if lengths else 1.0


def path_stats(G: nx.Graph, path, weight: str = "weight") -> dict:
    length, log_rel, uncertain = 0.0, 0.0, 0
    for u, v in zip(path[:-1], path[1:]):
        d = G.edges[u, v]
        length += d.get(weight, 1.0)
        pe = _pe(d)
        log_rel += math.log(pe)
        uncertain += pe < 1.0
    return {
        "nodes": list(path),
        "length": round(length, 1),
        "reliability": round(math.exp(log_rel), 4),
        "uncertain_edges": int(uncertain),
        "hops": len(path) - 1,
    }


def risk_adjusted_path(G: nx.Graph, source, target, risk_aversion: float = 1.0,
                       weight: str = "weight"):
    """Minimum risk-adjusted-cost path, or None if target is unreachable."""
    l_ref = reference_length(G, weight)

    def cost(u, v, d):
        return d.get(weight, 1.0) + risk_aversion * l_ref * (-math.log(_pe(d)))

    try:
        return nx.dijkstra_path(G, source, target, weight=cost)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def plan_routes(G: nx.Graph, source, target, risk_aversion: float = 1.0,
                weight: str = "weight") -> dict:
    """Shortest vs risk-adjusted route between source and target, with their statistics."""
    out = {"source": source, "target": target, "risk_aversion": risk_aversion,
           "shortest": None, "risk_adjusted": None}
    try:
        sp = nx.dijkstra_path(G, source, target, weight=weight)
        out["shortest"] = path_stats(G, sp, weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return out
    rp = risk_adjusted_path(G, source, target, risk_aversion, weight)
    if rp is not None:
        out["risk_adjusted"] = path_stats(G, rp, weight)
        s, r = out["shortest"], out["risk_adjusted"]
        out["detour_pct"] = round((r["length"] / s["length"] - 1) * 100, 1) if s["length"] else 0.0
        out["reliability_gain"] = round(r["reliability"] - s["reliability"], 4)
    return out


def nearest_node(pos: dict, y: float, x: float, candidates=None):
    """Closest node to (y, x) in the same coordinate system as `pos`."""
    keys = list(candidates if candidates is not None else pos.keys())
    if not keys:
        return None
    arr = np.array([pos[k] for k in keys], dtype=float)
    i = int(np.argmin(((arr - np.array([y, x])) ** 2).sum(axis=1)))
    return keys[i]


def pick_far_pair(G: nx.Graph, pos: dict):
    """Two far-apart nodes in the largest connected component (a sensible default O/D)."""
    if G.number_of_nodes() < 2:
        return None, None
    lcc = max(nx.connected_components(G), key=len)
    keys = [n for n in lcc if n in pos]
    if len(keys) < 2:
        return None, None
    arr = np.array([pos[k] for k in keys], dtype=float)
    a = int(np.argmax(((arr - arr.mean(axis=0)) ** 2).sum(axis=1)))
    b = int(np.argmax(((arr - arr[a]) ** 2).sum(axis=1)))
    return keys[a], keys[b]
