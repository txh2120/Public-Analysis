#!/usr/bin/env python3
"""AMCS (Adaptive Monte Carlo Search) for BHS Laplacian Spectral Radius Bounds.

Implements the AMCS algorithm (Vito 2023) to search for counterexamples to
38 open BHS (Brankov-Hansen-Stevanovic 2006) upper bounds on mu(G).

The algorithm combines:
  - NMCS (Nested Monte Carlo Search) with recursive depth/level
  - Adaptive shrink/grow phases to escape local optima
  - Graph mutation operations: add_leaf, subdivide_edge, add_non_edge

Usage:
  python src/amcs_bound_search.py --test                  # Smoke test
  python src/amcs_bound_search.py --bound 44 --time 60    # Single bound, 60s
  python src/amcs_bound_search.py --all --time 300        # All 38 bounds, 300s each
"""

import argparse
import copy
import random
import sys
import time

import networkx as nx
import numpy as np

# ─────────────────────────────────────────────────────────────────────
# Bridge to existing bound evaluation code
# ─────────────────────────────────────────────────────────────────────

sys.path.insert(0, 'D:/Public Analysis/src')
from exhaustive_bound_search import (
    laplacian_spectral_radius,
    compute_dv_mv,
    compute_vertex_bounds,
    compute_edge_bounds,
    VERTEX_BOUND_IDS,
    EDGE_BOUND_IDS,
    ALL_BOUND_IDS,
    evaluate_all_bounds,
)


# ─────────────────────────────────────────────────────────────────────
# 1. Graph <-> Adjacency Matrix Conversion
# ─────────────────────────────────────────────────────────────────────

def graph_to_adj(G: nx.Graph) -> np.ndarray:
    """Convert NetworkX graph to numpy adjacency matrix.

    Handles non-contiguous node IDs by sorting nodes.
    """
    nodes = sorted(G.nodes())
    return nx.to_numpy_array(G, nodelist=nodes, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────
# 2. Scoring Functions
# ─────────────────────────────────────────────────────────────────────

def score_single_bound(G: nx.Graph, bound_id: int) -> float:
    """Score a graph against a single bound.

    Returns mu - bound_value. Positive means counterexample found.
    """
    if G.number_of_nodes() < 2 or G.number_of_edges() < 1:
        return -999.0

    A = graph_to_adj(G)
    mu = laplacian_spectral_radius(A)
    dv, mv = compute_dv_mv(A)

    if bound_id in VERTEX_BOUND_IDS:
        bounds = compute_vertex_bounds(dv, mv)
    elif bound_id in EDGE_BOUND_IDS:
        bounds = compute_edge_bounds(A, dv, mv)
    else:
        return -999.0

    bound_val = bounds.get(bound_id, 0.0)
    if bound_val == 0.0:
        return -999.0

    return mu - bound_val


def score_all_bounds(G: nx.Graph) -> float:
    """Score a graph against all 38 bounds. Returns max(mu - bound) over all.

    Positive means at least one bound is violated (counterexample).
    """
    if G.number_of_nodes() < 2 or G.number_of_edges() < 1:
        return -999.0

    A = graph_to_adj(G)
    mu, bound_vals, gaps = evaluate_all_bounds(A)

    # gaps = bound - mu (positive = bound holds), we want mu - bound
    max_violation = max(-gap for gap in gaps.values())
    return max_violation


def score_all_bounds_detailed(G: nx.Graph):
    """Score against all bounds, return (max_score, worst_bound_id, details)."""
    if G.number_of_nodes() < 2 or G.number_of_edges() < 1:
        return -999.0, None, {}

    A = graph_to_adj(G)
    mu, bound_vals, gaps = evaluate_all_bounds(A)

    # Find the bound with smallest gap (most violated / tightest)
    worst_bid = min(gaps, key=gaps.get)
    max_violation = -gaps[worst_bid]  # mu - bound

    return max_violation, worst_bid, {'mu': mu, 'bounds': bound_vals, 'gaps': gaps}


# ─────────────────────────────────────────────────────────────────────
# 3. Graph Mutation Operations (Moves)
# ─────────────────────────────────────────────────────────────────────

def random_tree(n: int) -> nx.Graph:
    """Generate a random tree on n vertices using Prufer sequence."""
    if n <= 1:
        G = nx.Graph()
        G.add_node(0)
        return G
    if n == 2:
        G = nx.Graph()
        G.add_edge(0, 1)
        return G
    # Random Prufer sequence
    seq = [random.randint(0, n - 1) for _ in range(n - 2)]
    G = nx.from_prufer_sequence(seq)
    return G


def enumerate_moves(G: nx.Graph):
    """Enumerate all possible graph moves.

    Yields (move_type, args) tuples:
      - ('add_leaf', v): add new vertex connected to v
      - ('subdivide', u, v): subdivide edge (u, v)
      - ('add_edge', u, v): add non-edge between u and v
      - ('add_edge_degree_diverse', u, v): add non-edge between vertices of
        different degree (heuristic from RA feedback)
    """
    nodes = list(G.nodes())
    edges = list(G.edges())

    # 1. Add leaf to each existing vertex
    for v in nodes:
        yield ('add_leaf', v)

    # 2. Subdivide each existing edge
    for u, v in edges:
        yield ('subdivide', u, v)

    # 3. Add non-edges (from complement graph)
    #    For efficiency, limit enumeration for larger graphs
    node_set = set(nodes)
    non_edges = list(nx.non_edges(G))

    # Prioritize degree-diverse non-edges (RA feedback heuristic)
    if len(non_edges) > 0:
        degrees = dict(G.degree())
        # Sort: edges between vertices of most different degree first
        non_edges.sort(key=lambda e: -abs(degrees[e[0]] - degrees[e[1]]))

    for u, v in non_edges:
        yield ('add_edge', u, v)


def apply_move(G: nx.Graph, move) -> nx.Graph:
    """Apply a move to a graph, returning a new graph.

    Does NOT check connectivity — caller must verify.
    """
    G_new = G.copy()
    move_type = move[0]

    if move_type == 'add_leaf':
        v = move[1]
        new_node = max(G_new.nodes()) + 1
        G_new.add_edge(v, new_node)

    elif move_type == 'subdivide':
        u, v = move[1], move[2]
        if G_new.has_edge(u, v):
            new_node = max(G_new.nodes()) + 1
            G_new.remove_edge(u, v)
            G_new.add_edge(u, new_node)
            G_new.add_edge(new_node, v)

    elif move_type == 'add_edge':
        u, v = move[1], move[2]
        if not G_new.has_edge(u, v):
            G_new.add_edge(u, v)

    return G_new


# ─────────────────────────────────────────────────────────────────────
# 4. Shrink Operations
# ─────────────────────────────────────────────────────────────────────

def get_leaves(G: nx.Graph):
    """Get all leaf nodes (degree 1)."""
    return [v for v in G.nodes() if G.degree(v) == 1]


def get_subdivision_vertices(G: nx.Graph):
    """Get all degree-2 vertices that can be contracted (subdivision vertices)."""
    result = []
    for v in G.nodes():
        if G.degree(v) == 2:
            neighbors = list(G.neighbors(v))
            # Only a subdivision vertex if its two neighbors are not already connected
            if len(neighbors) == 2 and not G.has_edge(neighbors[0], neighbors[1]):
                result.append(v)
    return result


def remove_leaf(G: nx.Graph) -> nx.Graph:
    """Remove a random leaf vertex. Returns copy."""
    leaves = get_leaves(G)
    if not leaves or G.number_of_nodes() <= 3:
        return G.copy()

    leaf = random.choice(leaves)
    G_new = G.copy()
    G_new.remove_node(leaf)
    return G_new


def remove_subdivision(G: nx.Graph) -> nx.Graph:
    """Remove a random subdivision vertex (degree-2, non-adjacent neighbors).

    Contracts v out: removes v, adds edge between its two neighbors.
    Returns copy.
    """
    subdiv = get_subdivision_vertices(G)
    if not subdiv or G.number_of_nodes() <= 3:
        return G.copy()

    v = random.choice(subdiv)
    neighbors = list(G.neighbors(v))
    G_new = G.copy()
    G_new.remove_node(v)
    G_new.add_edge(neighbors[0], neighbors[1])
    return G_new


def maybe_shrink(G: nx.Graph, depth: int) -> nx.Graph:
    """Probabilistically shrink graph to escape local optima.

    With probability depth/(depth+1): shrink (remove leaf or subdivision).
    With probability 1/(depth+1): keep as-is.
    """
    p_shrink = depth / (depth + 1) if depth > 0 else 0.0

    if random.random() >= p_shrink:
        return G.copy()

    # Choose shrink operation
    leaves = get_leaves(G)
    subdiv = get_subdivision_vertices(G)

    candidates = []
    if leaves and G.number_of_nodes() > 3:
        candidates.append('leaf')
    if subdiv and G.number_of_nodes() > 3:
        candidates.append('subdiv')

    if not candidates:
        return G.copy()

    op = random.choice(candidates)
    if op == 'leaf':
        return remove_leaf(G)
    else:
        return remove_subdivision(G)


# ─────────────────────────────────────────────────────────────────────
# 5. Random Rollout
# ─────────────────────────────────────────────────────────────────────

def random_rollout(G: nx.Graph, depth: int) -> nx.Graph:
    """Apply 'depth' random moves to graph. Returns new graph.

    Each move is randomly chosen from available moves.
    Only applies moves that keep the graph connected.
    """
    G_current = G.copy()

    for _ in range(max(1, depth)):
        moves = list(enumerate_moves(G_current))
        if not moves:
            break

        random.shuffle(moves)
        applied = False

        # Try up to 10 random moves to find one that keeps connectivity
        for move in moves[:10]:
            G_after = apply_move(G_current, move)
            if nx.is_connected(G_after):
                G_current = G_after
                applied = True
                break

        if not applied:
            # Fallback: just add a leaf (always keeps connectivity)
            nodes = list(G_current.nodes())
            v = random.choice(nodes)
            new_node = max(G_current.nodes()) + 1
            G_current.add_edge(v, new_node)

    return G_current


# ─────────────────────────────────────────────────────────────────────
# 6. NMCS (Nested Monte Carlo Search)
# ─────────────────────────────────────────────────────────────────────

def nmcs(G: nx.Graph, depth: int, level: int, score_fn, deadline: float,
         max_nodes: int = 50) -> tuple:
    """Nested Monte Carlo Search.

    Args:
        G: current graph
        depth: number of random moves in rollout
        level: recursion level (0 = random rollout)
        score_fn: callable(G) -> float
        deadline: time.time() deadline
        max_nodes: maximum graph size to prevent blowup

    Returns:
        (best_graph, best_score)
    """
    # Time check
    if time.time() >= deadline:
        return G.copy(), score_fn(G)

    if level == 0:
        # Random rollout
        G_new = random_rollout(G, depth)
        return G_new, score_fn(G_new)

    # Level > 0: systematic enumeration
    current_score = score_fn(G)
    best = (G.copy(), current_score)

    moves = list(enumerate_moves(G))
    # Shuffle for diversity, but keep degree-diverse non-edges early
    # (they're already sorted by degree difference in enumerate_moves)
    random.shuffle(moves)

    for move in moves:
        if time.time() >= deadline:
            break

        G_after = apply_move(G, move)

        # Skip if disconnected
        if not nx.is_connected(G_after):
            continue

        # Skip if graph too large
        if G_after.number_of_nodes() > max_nodes:
            continue

        # Recurse at level-1
        G_result, score = nmcs(G_after, depth, level - 1, score_fn, deadline,
                               max_nodes)

        if score > best[1]:
            best = (G_result.copy(), score)

            # Early exit for large graphs to avoid combinatorial explosion
            if G.number_of_nodes() > 20:
                break

    return best


# ─────────────────────────────────────────────────────────────────────
# 7. AMCS Outer Loop
# ─────────────────────────────────────────────────────────────────────

def amcs(score_fn, time_budget: float, max_depth: int = 5, max_level: int = 2,
         initial_graph: nx.Graph = None, max_nodes: int = 50,
         verbose: bool = True) -> tuple:
    """Adaptive Monte Carlo Search for graph counterexamples.

    Args:
        score_fn: callable(G) -> float. Positive = counterexample found.
        time_budget: seconds to search.
        max_depth: maximum rollout depth.
        max_level: maximum NMCS recursion level.
        initial_graph: warm start graph (None = random tree).
        max_nodes: maximum allowed graph size.
        verbose: print progress.

    Returns:
        (best_graph, best_score, history)
    """
    start_time = time.time()
    deadline = start_time + time_budget

    # Initialize
    if initial_graph is not None:
        G = initial_graph.copy()
    else:
        G = random_tree(5)

    best_score = score_fn(G)
    best_graph = G.copy()
    depth = 0
    level = 1
    iteration = 0
    history = []

    if verbose:
        print(f"AMCS start | n={G.number_of_nodes()} e={G.number_of_edges()} "
              f"| score={best_score:.6f}")

    while time.time() < deadline and level <= max_level:
        iteration += 1

        # Shrink phase (escape local optima)
        G_shrunk = maybe_shrink(G, depth)

        # NMCS search from shrunk graph
        remaining = deadline - time.time()
        # Allocate time per iteration: more time for higher levels
        iter_budget = min(remaining, max(5.0, remaining * 0.1))
        iter_deadline = time.time() + iter_budget

        G_new, new_score = nmcs(G_shrunk, depth, level, score_fn,
                                iter_deadline, max_nodes)

        elapsed = time.time() - start_time
        history.append({
            'iteration': iteration,
            'time': elapsed,
            'score': new_score,
            'best_score': best_score,
            'depth': depth,
            'level': level,
            'n': G_new.number_of_nodes(),
            'e': G_new.number_of_edges(),
        })

        if new_score > best_score:
            improvement = new_score - best_score
            best_score = new_score
            best_graph = G_new.copy()
            G = G_new
            depth = 0
            level = 1  # Reset on improvement

            if verbose:
                print(f"  iter {iteration:4d} | t={elapsed:6.1f}s | "
                      f"IMPROVED +{improvement:.6f} -> {best_score:.6f} | "
                      f"n={G.number_of_nodes()} e={G.number_of_edges()} | "
                      f"depth={depth} level={level}")

            # Early exit if counterexample found
            if best_score > 1e-6:
                if verbose:
                    print(f"  *** COUNTEREXAMPLE FOUND at iter {iteration} ***")
                break
        else:
            if depth < max_depth:
                depth += 1
            else:
                depth = 0
                level += 1  # Escalate

            if verbose and iteration % 10 == 0:
                print(f"  iter {iteration:4d} | t={elapsed:6.1f}s | "
                      f"score={new_score:.6f} best={best_score:.6f} | "
                      f"depth={depth} level={level}")

    elapsed = time.time() - start_time
    if verbose:
        print(f"AMCS done  | {elapsed:.1f}s | {iteration} iterations | "
              f"best={best_score:.6f} | n={best_graph.number_of_nodes()} "
              f"e={best_graph.number_of_edges()}")

    return best_graph, best_score, history


# ─────────────────────────────────────────────────────────────────────
# 8. Multi-restart AMCS
# ─────────────────────────────────────────────────────────────────────

def amcs_multi_restart(score_fn, time_budget: float, restart_budget: float = 30.0,
                       max_depth: int = 5, max_level: int = 2,
                       initial_graph: nx.Graph = None, max_nodes: int = 50,
                       verbose: bool = True) -> tuple:
    """Run AMCS with multiple restarts within a total time budget.

    Args:
        score_fn: scoring function
        time_budget: total seconds
        restart_budget: seconds per restart attempt
        max_depth, max_level: AMCS parameters
        initial_graph: warm start for first restart
        max_nodes: max graph size
        verbose: print progress

    Returns:
        (best_graph, best_score, all_histories)
    """
    start_time = time.time()
    deadline = start_time + time_budget

    global_best_graph = None
    global_best_score = -999.0
    all_histories = []
    restart_num = 0

    while time.time() < deadline:
        restart_num += 1
        remaining = deadline - time.time()
        this_budget = min(restart_budget, remaining)

        if this_budget < 2.0:
            break

        if verbose:
            print(f"\n--- Restart {restart_num} | {remaining:.0f}s remaining ---")

        # Use warm start for first restart if provided
        init = initial_graph if (restart_num == 1 and initial_graph is not None) else None

        G, score, hist = amcs(score_fn, this_budget, max_depth, max_level,
                              init, max_nodes, verbose)

        all_histories.append(hist)

        if score > global_best_score:
            global_best_score = score
            global_best_graph = G.copy()

            if score > 0:
                if verbose:
                    print(f"*** GLOBAL COUNTEREXAMPLE at restart {restart_num} ***")
                break

    if verbose:
        total_elapsed = time.time() - start_time
        print(f"\n=== AMCS Multi-Restart Summary ===")
        print(f"Total time: {total_elapsed:.1f}s | Restarts: {restart_num}")
        print(f"Best score: {global_best_score:.6f}")
        if global_best_graph is not None:
            print(f"Best graph: n={global_best_graph.number_of_nodes()} "
                  f"e={global_best_graph.number_of_edges()}")

    return global_best_graph, global_best_score, all_histories


# ─────────────────────────────────────────────────────────────────────
# 9. CLI Entry Points
# ─────────────────────────────────────────────────────────────────────

def run_test():
    """Smoke test: run AMCS for a short time on Bound 44."""
    print("=" * 60)
    print("AMCS Smoke Test")
    print("=" * 60)

    # Test graph operations
    print("\n[1] Testing graph operations...")
    G = random_tree(5)
    print(f"  Random tree: n={G.number_of_nodes()} e={G.number_of_edges()}")
    assert nx.is_connected(G), "Random tree should be connected"
    assert G.number_of_nodes() == 5, "Should have 5 nodes"

    # Test add_leaf
    G2 = apply_move(G, ('add_leaf', 0))
    assert G2.number_of_nodes() == 6, "add_leaf should add 1 node"
    assert nx.is_connected(G2), "add_leaf should keep connectivity"

    # Test subdivide
    edges = list(G.edges())
    if edges:
        u, v = edges[0]
        G3 = apply_move(G, ('subdivide', u, v))
        assert G3.number_of_nodes() == 6, "subdivide should add 1 node"
        assert nx.is_connected(G3), "subdivide should keep connectivity"

    # Test shrink
    G4 = remove_leaf(G)
    assert G4.number_of_nodes() <= G.number_of_nodes()

    print("  Graph operations: OK")

    # Test scoring
    print("\n[2] Testing scoring...")
    G_star = nx.star_graph(4)  # Star with 5 nodes
    s44 = score_single_bound(G_star, 44)
    print(f"  Star(5) vs Bound 44: score={s44:.6f}")

    s_all = score_all_bounds(G_star)
    print(f"  Star(5) vs all bounds: max_score={s_all:.6f}")

    score_detail, worst_bid, details = score_all_bounds_detailed(G_star)
    print(f"  Tightest bound: #{worst_bid} (score={score_detail:.6f})")
    print("  Scoring: OK")

    # Test NMCS level 0
    print("\n[3] Testing NMCS level 0 (random rollout)...")
    score_fn = lambda G: score_single_bound(G, 44)
    deadline = time.time() + 5.0
    G_roll, s_roll = nmcs(G_star, depth=3, level=0, score_fn=score_fn,
                          deadline=deadline)
    print(f"  Rollout: n={G_roll.number_of_nodes()} e={G_roll.number_of_edges()} "
          f"score={s_roll:.6f}")
    print("  NMCS level 0: OK")

    # Test AMCS short run
    print("\n[4] Testing AMCS (10s on Bound 44)...")
    G_best, best_score, hist = amcs(score_fn, time_budget=10.0, verbose=True)
    print(f"  Result: score={best_score:.6f} | iterations={len(hist)}")
    print(f"  Best graph: n={G_best.number_of_nodes()} e={G_best.number_of_edges()}")

    if best_score > 1e-6:
        print("  *** COUNTEREXAMPLE FOUND ***")
    else:
        print(f"  Gap to counterexample: {-best_score:.6f}")

    print("\n" + "=" * 60)
    print("All smoke tests passed!")
    print("=" * 60)


def run_single_bound(bound_id: int, time_budget: float, restart_budget: float = 30.0):
    """Search for counterexample to a specific bound."""
    print(f"Searching for counterexample to Bound {bound_id}")
    print(f"Time budget: {time_budget:.0f}s | Restart budget: {restart_budget:.0f}s")
    print("=" * 60)

    score_fn = lambda G: score_single_bound(G, bound_id)

    G_best, best_score, histories = amcs_multi_restart(
        score_fn, time_budget, restart_budget, verbose=True
    )

    # Report results
    print("\n" + "=" * 60)
    print(f"FINAL RESULT for Bound {bound_id}")
    print("=" * 60)

    if G_best is not None:
        A = graph_to_adj(G_best)
        mu = laplacian_spectral_radius(A)
        dv, mv = compute_dv_mv(A)

        if bound_id in VERTEX_BOUND_IDS:
            bvals = compute_vertex_bounds(dv, mv)
        else:
            bvals = compute_edge_bounds(A, dv, mv)

        bound_val = bvals.get(bound_id, 0.0)
        print(f"  mu(G) = {mu:.8f}")
        print(f"  Bound {bound_id} = {bound_val:.8f}")
        print(f"  Gap (mu - bound) = {mu - bound_val:.8f}")
        print(f"  Graph: n={G_best.number_of_nodes()} e={G_best.number_of_edges()}")
        print(f"  Edges: {sorted(G_best.edges())}")

        if best_score > 1e-6:
            print(f"\n  *** COUNTEREXAMPLE FOUND ***")
        else:
            print(f"\n  No counterexample found (closest gap: {-best_score:.8f})")

    return G_best, best_score


def run_all_bounds(time_budget: float, restart_budget: float = 30.0):
    """Search all 38 bounds, allocating time equally."""
    print(f"Searching all {len(ALL_BOUND_IDS)} bounds")
    print(f"Time per bound: {time_budget:.0f}s")
    print("=" * 60)

    results = {}
    for i, bid in enumerate(ALL_BOUND_IDS):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(ALL_BOUND_IDS)}] Bound {bid}")
        print(f"{'='*60}")

        score_fn = lambda G, b=bid: score_single_bound(G, b)

        G_best, best_score, histories = amcs_multi_restart(
            score_fn, time_budget, restart_budget, verbose=True
        )

        results[bid] = {
            'graph': G_best,
            'score': best_score,
            'is_counterexample': best_score > 0,
        }

        if best_score > 1e-6:
            print(f"  *** COUNTEREXAMPLE for Bound {bid} ***")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: All Bounds")
    print("=" * 60)
    print(f"{'Bound':>6} {'Score':>12} {'Status':>15} {'n':>4} {'e':>4}")
    print("-" * 45)

    counterexamples = []
    for bid in ALL_BOUND_IDS:
        r = results[bid]
        status = "COUNTER!" if r['is_counterexample'] else "no CE"
        n = r['graph'].number_of_nodes() if r['graph'] else 0
        e = r['graph'].number_of_edges() if r['graph'] else 0
        print(f"{bid:>6} {r['score']:>12.6f} {status:>15} {n:>4} {e:>4}")
        if r['is_counterexample']:
            counterexamples.append(bid)

    if counterexamples:
        print(f"\nCounterexamples found for bounds: {counterexamples}")
    else:
        print(f"\nNo counterexamples found.")
        # Show top-5 closest
        sorted_bounds = sorted(ALL_BOUND_IDS, key=lambda b: -results[b]['score'])
        print("Top-5 closest to counterexample:")
        for bid in sorted_bounds[:5]:
            print(f"  Bound {bid}: score={results[bid]['score']:.8f}")

    return results


# ─────────────────────────────────────────────────────────────────────
# 10. Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='AMCS search for BHS bound counterexamples'
    )
    parser.add_argument('--test', action='store_true',
                        help='Run smoke test')
    parser.add_argument('--bound', type=int, default=None,
                        help='Target bound ID (e.g., 44)')
    parser.add_argument('--all', action='store_true',
                        help='Search all 38 bounds')
    parser.add_argument('--time', type=float, default=60.0,
                        help='Time budget in seconds (default: 60)')
    parser.add_argument('--restart-time', type=float, default=30.0,
                        help='Time per restart in seconds (default: 30)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--max-nodes', type=int, default=50,
                        help='Maximum graph size (default: 50)')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.test:
        run_test()
    elif args.bound is not None:
        if args.bound not in ALL_BOUND_IDS:
            print(f"Error: Bound {args.bound} not in known bounds.")
            print(f"Available: {ALL_BOUND_IDS}")
            sys.exit(1)
        run_single_bound(args.bound, args.time, args.restart_time)
    elif args.all:
        run_all_bounds(args.time, args.restart_time)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
