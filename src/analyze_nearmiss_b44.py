#!/usr/bin/env python3
"""Analyze near-miss graphs for Bound 44.

Tasks:
1. Find the best tree-like graph for Bound 44 by searching paths and caterpillars
2. Analyze structural properties (degree sequence, diameter, clustering, etc.)
3. Compute per-edge Bound 44 values to find tightest edge
4. Evaluate all 38 bounds on the near-miss graph
"""

import json
import os
import sys
import time

import networkx as nx
import numpy as np

sys.path.insert(0, 'D:/Public Analysis/src')
from exhaustive_bound_search import (
    laplacian_spectral_radius,
    compute_dv_mv,
    compute_vertex_bounds,
    compute_edge_bounds,
    evaluate_all_bounds,
    VERTEX_BOUND_IDS,
    EDGE_BOUND_IDS,
    ALL_BOUND_IDS,
)
from amcs_bound_search import (
    graph_to_adj,
    score_single_bound,
    save_graph_to_json,
    random_tree,
    amcs,
)


def bound44_per_edge(G):
    """Compute Bound 44 value for each edge.

    Bound 44: max_(i~j) 2 + sqrt(2*((di-1)^2 + (dj-1)^2 + mi*mj - di*dj))

    Returns list of (u, v, bound44_value, gap_to_mu).
    """
    A = graph_to_adj(G)
    mu = laplacian_spectral_radius(A)
    dv, mv = compute_dv_mv(A)

    results = []
    for u, v in G.edges():
        di, dj = dv[u], dv[v]
        mi, mj = mv[u], mv[v]
        inner = 2.0 * ((di - 1)**2 + (dj - 1)**2 + mi * mj - di * dj)
        if inner < 0:
            inner = 0.0
        b44 = 2.0 + np.sqrt(inner)
        gap = b44 - mu
        results.append((u, v, float(b44), float(gap)))

    return mu, results


def analyze_graph_structure(G, name="Graph"):
    """Print detailed structural analysis of a graph."""
    n = G.number_of_nodes()
    e = G.number_of_edges()

    print(f"\n{'='*60}")
    print(f"Structural Analysis: {name}")
    print(f"{'='*60}")
    print(f"  Nodes: {n}")
    print(f"  Edges: {e}")
    print(f"  Density: {nx.density(G):.6f}")
    print(f"  Is tree: {e == n - 1 and nx.is_connected(G)}")

    # Degree sequence
    degrees = sorted([d for _, d in G.degree()], reverse=True)
    print(f"  Degree sequence: {degrees}")
    print(f"  Max degree: {max(degrees)}")
    print(f"  Min degree: {min(degrees)}")
    print(f"  Avg degree: {2*e/n:.4f}")

    # Degree distribution
    from collections import Counter
    deg_dist = Counter(degrees)
    print(f"  Degree distribution: {dict(sorted(deg_dist.items()))}")

    # Diameter and radius
    if nx.is_connected(G):
        diameter = nx.diameter(G)
        radius = nx.radius(G)
        center = nx.center(G)
        print(f"  Diameter: {diameter}")
        print(f"  Radius: {radius}")
        print(f"  Center vertices: {center}")

    # Clustering
    avg_clustering = nx.average_clustering(G)
    print(f"  Average clustering: {avg_clustering:.6f}")

    # Check if it's a tree
    if e == n - 1 and nx.is_connected(G):
        # Count leaves
        leaves = [v for v in G.nodes() if G.degree(v) == 1]
        print(f"  Leaves: {len(leaves)}")

        # Check if it's a path
        internal = [v for v in G.nodes() if G.degree(v) > 1]
        is_path = all(G.degree(v) <= 2 for v in G.nodes())
        print(f"  Is path: {is_path}")

        # Check if it's a caterpillar
        # A caterpillar = removing all leaves gives a path
        internal_subgraph = G.subgraph(internal).copy()
        if internal_subgraph.number_of_nodes() > 0:
            is_caterpillar_path = all(
                internal_subgraph.degree(v) <= 2
                for v in internal_subgraph.nodes()
            )
            print(f"  Internal nodes form path: {is_caterpillar_path}")
            if internal_subgraph.number_of_nodes() > 0:
                print(f"  Internal nodes: {internal_subgraph.number_of_nodes()}")

    return degrees


def evaluate_all_38_bounds(G, name="Graph"):
    """Evaluate all 38 bounds on graph and print sorted by gap."""
    A = graph_to_adj(G)
    mu, bound_vals, gaps = evaluate_all_bounds(A)

    print(f"\n{'='*60}")
    print(f"All 38 Bounds on {name}")
    print(f"{'='*60}")
    print(f"  mu(G) = {mu:.8f}")
    print(f"  n={G.number_of_nodes()} e={G.number_of_edges()}")
    print()
    print(f"  {'Bound':>6} {'Gap':>12} {'mu':>12} {'BoundVal':>12} {'Type':>8}")
    print(f"  {'-'*52}")

    sorted_bounds = sorted(ALL_BOUND_IDS, key=lambda b: gaps[b])

    for bid in sorted_bounds:
        gap = gaps[bid]
        bval = bound_vals[bid]
        btype = "vertex" if bid in VERTEX_BOUND_IDS else "edge"
        marker = " <-- TIGHT" if abs(gap) < 0.01 else ""
        marker = " *** VIOLATED" if gap < -1e-6 else marker
        print(f"  {bid:>6} {gap:>+12.6f} {mu:>12.6f} {bval:>12.6f} {btype:>8}{marker}")

    # Summary
    tightest_5 = sorted_bounds[:5]
    print(f"\n  Top 5 tightest bounds:")
    for bid in tightest_5:
        print(f"    Bound {bid}: gap={gaps[bid]:+.8f}")

    return mu, bound_vals, gaps


def find_best_path_for_b44():
    """Search paths P_n for various n to find closest to Bound 44 violation.

    From the log, the near-miss was a tree with n=51, e=50 grown from a path.
    """
    print("\n" + "="*60)
    print("Searching paths P_n for Bound 44 near-misses")
    print("="*60)

    best_gap = float('inf')
    best_n = None

    for n in range(3, 101):
        G = nx.path_graph(n)
        A = graph_to_adj(G)
        mu = laplacian_spectral_radius(A)
        dv, mv = compute_dv_mv(A)
        edge_bounds = compute_edge_bounds(A, dv, mv)
        b44 = edge_bounds.get(44, 0.0)
        gap = b44 - mu

        if gap < best_gap:
            best_gap = gap
            best_n = n

        if n <= 10 or n % 10 == 0 or abs(gap) < 0.01:
            print(f"  P_{n:3d}: mu={mu:.8f}  B44={b44:.8f}  gap={gap:+.8f}")

    print(f"\n  Best path: P_{best_n} with gap={best_gap:+.8f}")
    return best_n


def find_best_caterpillar_for_b44():
    """Search caterpillar trees for Bound 44 near-misses.

    A caterpillar is a tree where all vertices are within distance 1 of a central path.
    """
    print("\n" + "="*60)
    print("Searching caterpillars for Bound 44 near-misses")
    print("="*60)

    best_gap = float('inf')
    best_graph = None
    best_desc = ""

    # Try various caterpillar shapes
    for spine_len in range(5, 40):
        for n_leaves_per in [0, 1, 2, 3]:
            # Build caterpillar: path of spine_len + leaves
            G = nx.path_graph(spine_len)
            next_node = spine_len

            for v in range(1, spine_len - 1):  # internal nodes
                for _ in range(n_leaves_per):
                    G.add_edge(v, next_node)
                    next_node += 1

            n = G.number_of_nodes()
            if n < 3 or n > 80:
                continue

            score = score_single_bound(G, 44)
            gap = -score  # gap = bound - mu (positive = holds)

            if gap < best_gap:
                best_gap = gap
                best_graph = G.copy()
                best_desc = f"Caterpillar(spine={spine_len}, leaves_per_internal={n_leaves_per})"

    if best_graph:
        print(f"  Best caterpillar: {best_desc}")
        print(f"  n={best_graph.number_of_nodes()} e={best_graph.number_of_edges()} gap={best_gap:+.8f}")

    return best_graph, best_gap


def find_best_tree_via_amcs(n_range=(40, 60), time_per=30, n_trials=10):
    """Use short AMCS runs starting from random trees to find near-miss trees.

    This reproduces the Restart 36 pattern from the original log.
    """
    print("\n" + "="*60)
    print(f"AMCS tree search: n in {n_range}, {time_per}s per trial, {n_trials} trials")
    print("="*60)

    score_fn = lambda G: score_single_bound(G, 44)

    global_best_graph = None
    global_best_score = -999.0

    for trial in range(n_trials):
        n_init = np.random.randint(n_range[0], n_range[1] + 1)
        G_init = random_tree(n_init)

        G_best, best_score, hist = amcs(
            score_fn, time_budget=time_per, max_depth=5, max_level=2,
            initial_graph=G_init, max_nodes=100, verbose=False
        )

        n = G_best.number_of_nodes()
        e = G_best.number_of_edges()
        is_tree = (e == n - 1) and nx.is_connected(G_best)

        print(f"  Trial {trial+1:2d}: n={n:3d} e={e:3d} tree={is_tree} "
              f"score={best_score:+.8f} iters={len(hist)}")

        if best_score > global_best_score:
            global_best_score = best_score
            global_best_graph = G_best.copy()

    if global_best_graph:
        n = global_best_graph.number_of_nodes()
        e = global_best_graph.number_of_edges()
        print(f"\n  Global best: n={n} e={e} score={global_best_score:+.8f}")

    return global_best_graph, global_best_score


def main():
    print("="*60)
    print("Near-Miss Bound 44 Analysis")
    print("="*60)

    # ─────────────────────────────────────────────────────────────
    # Step 1: Search paths for best n
    # ─────────────────────────────────────────────────────────────
    best_path_n = find_best_path_for_b44()

    # Analyze the best path
    G_path = nx.path_graph(best_path_n)
    analyze_graph_structure(G_path, f"Path P_{best_path_n}")

    # Per-edge analysis on the best path
    mu, edge_results = bound44_per_edge(G_path)
    print(f"\n  Per-edge Bound 44 analysis (mu={mu:.8f}):")
    # Sort by gap (ascending = tightest)
    edge_results.sort(key=lambda x: x[3])
    for u, v, b44, gap in edge_results[:10]:
        du, dv_val = G_path.degree(u), G_path.degree(v)
        print(f"    Edge ({u},{v}): deg=({du},{dv_val}) B44={b44:.8f} gap={gap:+.8f}")

    # Evaluate all 38 bounds on best path
    evaluate_all_38_bounds(G_path, f"Path P_{best_path_n}")

    # ─────────────────────────────────────────────────────────────
    # Step 2: Search caterpillars
    # ─────────────────────────────────────────────────────────────
    G_cat, cat_gap = find_best_caterpillar_for_b44()
    if G_cat:
        analyze_graph_structure(G_cat, "Best Caterpillar")
        mu_cat, edge_results_cat = bound44_per_edge(G_cat)
        print(f"\n  Per-edge Bound 44 analysis (mu={mu_cat:.8f}):")
        edge_results_cat.sort(key=lambda x: x[3])
        for u, v, b44, gap in edge_results_cat[:10]:
            du, dv_val = G_cat.degree(u), G_cat.degree(v)
            print(f"    Edge ({u},{v}): deg=({du},{dv_val}) B44={b44:.8f} gap={gap:+.8f}")

        evaluate_all_38_bounds(G_cat, "Best Caterpillar")

    # ─────────────────────────────────────────────────────────────
    # Step 3: AMCS tree search (reproducing Restart 36 pattern)
    # ─────────────────────────────────────────────────────────────
    G_amcs, amcs_score = find_best_tree_via_amcs(
        n_range=(40, 60), time_per=30, n_trials=10
    )

    if G_amcs:
        analyze_graph_structure(G_amcs, "Best AMCS Graph")
        mu_amcs, edge_results_amcs = bound44_per_edge(G_amcs)
        print(f"\n  Per-edge Bound 44 analysis (mu={mu_amcs:.8f}):")
        edge_results_amcs.sort(key=lambda x: x[3])
        for u, v, b44, gap in edge_results_amcs[:10]:
            du, dv_val = G_amcs.degree(u), G_amcs.degree(v)
            print(f"    Edge ({u},{v}): deg=({du},{dv_val}) B44={b44:.8f} gap={gap:+.8f}")

        evaluate_all_38_bounds(G_amcs, "Best AMCS Graph")

        # Save best AMCS graph
        save_graph_to_json(G_amcs, 'resources/best_b44_amcs_tree.json',
                          bound_id=44, score=amcs_score)

    # ─────────────────────────────────────────────────────────────
    # Step 4: Compare all candidates
    # ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Summary: Best Candidates")
    print("="*60)

    candidates = []

    # Path
    score_path = score_single_bound(G_path, 44)
    candidates.append(("Path P_" + str(best_path_n), G_path, score_path))

    # Caterpillar
    if G_cat:
        score_cat = score_single_bound(G_cat, 44)
        candidates.append(("Best Caterpillar", G_cat, score_cat))

    # AMCS
    if G_amcs:
        candidates.append(("AMCS Tree", G_amcs, amcs_score))

    candidates.sort(key=lambda x: -x[2])

    for name, G, score in candidates:
        n = G.number_of_nodes()
        e = G.number_of_edges()
        is_tree = (e == n - 1) and nx.is_connected(G)
        print(f"  {name:30s}: n={n:3d} e={e:3d} tree={is_tree} "
              f"score={score:+.10f}")

    # Save the overall best
    best_name, best_G, best_score = candidates[0]
    save_graph_to_json(best_G, 'resources/best_b44_overall.json',
                      bound_id=44, score=best_score,
                      extra={'source': best_name})

    print(f"\n  Overall best: {best_name} (saved to resources/best_b44_overall.json)")


if __name__ == '__main__':
    main()
