import numpy as np
import networkx as nx

try:
    from scipy.linalg import eigh as _scipy_eigh
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    _scipy_eigh = None

class CriticalityVarianceEngine:
    """
    Criticality Variance & Monte Carlo Resilience Engine.
    Simulates network flow vulnerabilities by drawing M graph samples using calibrated edge probabilities pe.
    Computes Betweenness Centrality statistics (meanBC & stdBC) to map node risks,
    and calculates the global Urban Resilience Index (RI) post-disruption.
    """
    def __init__(self, num_samples=30, wc=0.6, wd=0.4):
        self.num_samples = num_samples
        self.wc = wc  # Weight for LCC ratio
        self.wd = wd  # Weight for Algebraic Connectivity ratio

    def get_fiedler_eigenvalue(self, G):
        """
        Computes the Fiedler eigenvalue (algebraic connectivity) of a graph.
        If graph is disconnected, computes it on the largest connected component.
        """
        if len(G) == 0:
            return 0.0
            
        # Get largest connected component
        lcc_nodes = max(nx.connected_components(G), key=len)
        sub_G = G.subgraph(lcc_nodes)
        
        if len(sub_G) <= 2:
            return 0.0
            
        try:
            # nx.algebraic_connectivity is built-in and uses SciPy/NumPy internally
            val = nx.algebraic_connectivity(sub_G, weight='weight')
            return float(val)
        except Exception:
            # Fallback manual calculation using SciPy eigh on Laplacians
            try:
                if HAS_SCIPY and _scipy_eigh is not None:
                    L = nx.laplacian_matrix(sub_G).toarray()
                    eigenvals = _scipy_eigh(L, eigvals_only=True)
                    # Second smallest eigenvalue
                    if len(eigenvals) > 1:
                        return float(eigenvals[1])
            except Exception:
                pass
        return 0.0

    def compute_resilience_index(self, original_G, disrupted_G, healed_G):
        """
        Computes the Urban Resilience Index (RI) based on:
        RI = wc * (|LCC_post| / |LCC_pre|) + wd * (lambda_post / lambda_pre)
        """
        def get_lcc_size(G):
            if len(G) == 0:
                return 0
            return len(max(nx.connected_components(G), key=len))

        lcc_pre = get_lcc_size(original_G)
        lcc_disrupted = get_lcc_size(disrupted_G)
        lcc_post = get_lcc_size(healed_G)

        lambda_pre = self.get_fiedler_eigenvalue(original_G)
        lambda_disrupted = self.get_fiedler_eigenvalue(disrupted_G)
        lambda_post = self.get_fiedler_eigenvalue(healed_G)

        # LCC and Fiedler connectivity ratios
        lcc_ratio_disrupted = lcc_disrupted / lcc_pre if lcc_pre > 0 else 0.0
        lcc_ratio_healed = lcc_post / lcc_pre if lcc_pre > 0 else 0.0
        
        # Clip the algebraic-connectivity ratio to [0, 1] so a smaller remnant cannot appear more resilient than the original graph.
        lambda_ratio_disrupted = min(1.0, max(0.0, (lambda_disrupted / lambda_pre) * lcc_ratio_disrupted)) if lambda_pre > 0 else 0.0
        lambda_ratio_healed = min(1.0, max(0.0, (lambda_post / lambda_pre) * lcc_ratio_healed)) if lambda_pre > 0 else 0.0

        # Disruption RI vs Healed RI
        ri_disrupted = self.wc * lcc_ratio_disrupted + self.wd * lambda_ratio_disrupted
        ri_healed = self.wc * lcc_ratio_healed + self.wd * lambda_ratio_healed

        return {
            "LCC": {
                "original": lcc_pre,
                "disrupted": lcc_disrupted,
                "healed": lcc_post,
                "gain_pct": float(round((lcc_post - lcc_disrupted) / max(1, lcc_disrupted) * 100, 1))
            },
            "FiedlerLambda": {
                "original": float(round(lambda_pre, 4)),
                "disrupted": float(round(lambda_disrupted, 4)),
                "healed": float(round(lambda_post, 4))
            },
            "RI_disrupted": float(round(ri_disrupted, 3)),
            "RI_healed": float(round(ri_healed, 3)),
            "connectivity_gain": float(round((ri_healed - ri_disrupted) * 100, 1))
        }

    def edge_pe(self, data, healed_lookup=None, u=None, v=None, default_pe=1.0):
        """Existence probability of an edge: explicit 'pe' attr > healed lookup > default."""
        if "pe" in data and data["pe"] is not None:
            return float(data["pe"])
        if data.get("healed", False):
            return (healed_lookup or {}).get((u, v), 0.5)
        return default_pe

    def run_monte_carlo(self, G, healed_edges=None, default_pe=1.0, bc_exact_max_nodes=400,
                        bc_k=100, seed=None):
        """
        Monte Carlo edge-drop simulation.
        In every trial each edge survives with its existence probability pe (Bernoulli):
          * edges with a 'pe' attribute use it (healed gaps, repairs, mask-conflated OSM edges)
          * healed edges without one use the pe in `healed_edges` (0.5 if missing)
          * all other edges use `default_pe` (1.0 = perfectly reliable; set <1 for a hazard rate)
        Computes Betweenness Centrality mean / std per node over the trials. For graphs larger
        than `bc_exact_max_nodes`, BC is approximated from `bc_k` sampled source nodes.

        Also stores self.last_summary with the expected fraction of nodes in the largest
        connected component and the probability that the whole graph stays connected.
        """
        rng = np.random.default_rng(seed)
        healed_lookup = {}
        for edge in healed_edges or []:
            u, v = edge["u"], edge["v"]
            healed_lookup[(u, v)] = edge["pe"]
            healed_lookup[(v, u)] = edge["pe"]

        nodes = list(G.nodes())
        bc_history = {node: [] for node in G.nodes()}
        edge_list = list(G.edges(data=True))
        pes = np.array([self.edge_pe(d, healed_lookup, u, v, default_pe) for u, v, d in edge_list])
        uncertain = np.nonzero(pes < 1.0)[0]
        approx = len(nodes) > bc_exact_max_nodes
        lcc_fracs, connected = [], []

        for t in range(self.num_samples):
            sample_G = G.copy()
            draws = rng.random(len(uncertain))
            drop = uncertain[draws > pes[uncertain]]
            sample_G.remove_edges_from((edge_list[i][0], edge_list[i][1]) for i in drop)

            try:
                if approx:
                    bc = nx.betweenness_centrality(sample_G, k=min(bc_k, len(nodes)),
                                                   weight='weight', seed=int(rng.integers(2**31)))
                else:
                    bc = nx.betweenness_centrality(sample_G, weight='weight')
            except Exception:
                bc = {node: 0.0 for node in nodes}

            for node in nodes:
                bc_history[node].append(bc.get(node, 0.0))
            if len(sample_G):
                comps = list(nx.connected_components(sample_G))
                lcc_fracs.append(max(len(c) for c in comps) / len(sample_G))
                connected.append(len(comps) == 1)

        self.last_summary = {
            "samples": self.num_samples,
            "uncertain_edges": int(len(uncertain)),
            "expected_lcc_fraction": float(round(np.mean(lcc_fracs), 4)) if lcc_fracs else 0.0,
            "p_fully_connected": float(round(np.mean(connected), 4)) if connected else 0.0,
            "bc_approximate": bool(approx),
        }

        # Compute summary statistics
        node_stats = {}
        for node in nodes:
            history = bc_history[node]
            mean_bc = float(np.mean(history))
            std_bc = float(np.std(history))
            node_stats[node] = {
                "meanBC": mean_bc,
                "stdBC": std_bc
            }

        # Segregate nodes into Risk Quadrants based on relative thresholds (e.g. median)
        mean_bcs = [stat["meanBC"] for stat in node_stats.values()]
        std_bcs = [stat["stdBC"] for stat in node_stats.values()]
        
        # Calculate thresholds
        mean_thresh = np.percentile(mean_bcs, 60) if mean_bcs else 0.05
        std_thresh = np.percentile(std_bcs, 60) if std_bcs else 0.02
        
        # Avoid dividing or thresholding at 0
        mean_thresh = max(0.01, mean_thresh)
        std_thresh = max(0.005, std_thresh)

        for node, stats in node_stats.items():
            mean = stats["meanBC"]
            std = stats["stdBC"]
            
            if mean >= mean_thresh:
                if std >= std_thresh:
                    quadrant = "CRITICAL"  # High centrality, high uncertainty
                else:
                    quadrant = "RELIABLE"  # High centrality, low uncertainty
            else:
                if std >= std_thresh:
                    quadrant = "UNCERTAIN"  # Low centrality, high uncertainty
                else:
                    quadrant = "SAFE"      # Low centrality, low uncertainty
                    
            node_stats[node]["quadrant"] = quadrant

        # Highlight top 3 satellite re-tasking priorities (high meanBC and high stdBC)
        re_tasking_list = []
        for node, stats in node_stats.items():
            if stats["quadrant"] == "CRITICAL":
                # Rank by meanBC * stdBC
                score = stats["meanBC"] * stats["stdBC"]
                re_tasking_list.append((node, score, stats["meanBC"], stats["stdBC"]))
                
        re_tasking_list.sort(key=lambda x: x[1], reverse=True)
        top_priorities = []
        for i, (node, score, mean, std) in enumerate(re_tasking_list[:3]):
            top_priorities.append({
                "node_id": node,
                "priority": f"PRIORITY {i + 1}",
                "meanBC": float(round(mean, 3)),
                "stdBC": float(round(std, 3))
            })

        return node_stats, top_priorities
