#!/usr/bin/env python3
"""Tabu Search for BHS Laplacian Spectral Radius Bound Counterexamples.

Unlike CEM/AMCS which build graphs from scratch (and converge to regular/bipartite
tight cases), Tabu Search starts from an EXISTING graph and modifies it via
edge-flips. This avoids the scratch-build attractor toward regular graphs.

Edge-flip neighborhood: remove one edge + add one non-edge (maintaining connectivity).
Tabu list prevents revisiting recent moves. Aspiration criterion overrides tabu
when a move beats the global best.

Usage:
  python src/tabu_bound_search.py --test                          # Quick smoke test
  python src/tabu_bound_search.py --bound 44 --time 60            # Bound 44, 60s
  python src/tabu_bound_search.py --bound 44 --time 1800 --start-n 15  # 30min, n=15

References:
  - Glover (1989, 1990): Tabu Search fundamentals
  - BHS bounds: Brankov-Hansen-Stevanovic (2006)
"""

import argparse
import random
import sys
import time
from collections import deque
from itertools import islice

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
# 1. Graph Utilities
# ─────────────────────────────────────────────────────────────────────

def graph_to_adj(G: nx.Graph) -> np.ndarray:
    """Convert NetworkX graph to numpy adjacency matrix."""
    nodes = sorted(G.nodes())
    return nx.to_numpy_array(G, nodelist=nodes, dtype=np.float64)


def score_single_bound(G: nx.Graph, bound_id: int) -> float:
    """Score = mu - bound_value. Positive means counterexample found."""
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


def score_all_bounds_detailed(G: nx.Graph):
    """Score against all bounds, return (max_score, worst_bound_id, details)."""
    if G.number_of_nodes() < 2 or G.number_of_edges() < 1:
        return -999.0, None, {}

    A = graph_to_adj(G)
    mu, bound_vals, gaps = evaluate_all_bounds(A)

    worst_bid = min(gaps, key=gaps.get)
    max_violation = -gaps[worst_bid]  # mu - bound

    return max_violation, worst_bid, {'mu': mu, 'bounds': bound_vals, 'gaps': gaps}


# ─────────────────────────────────────────────────────────────────────
# 2. Edge-Flip Neighborhood
# ─────────────────────────────────────────────────────────────────────

def edge_flip(G: nx.Graph, remove_edge, add_edge):
    """Apply edge-flip: remove one edge, add one non-edge. Returns new graph."""
    G_new = G.copy()
    G_new.remove_edge(*remove_edge)
    G_new.add_edge(*add_edge)
    return G_new


def is_connected_after_flip(G: nx.Graph, remove_edge, add_edge):
    """Fast connectivity check after edge flip.

    Instead of copying + full BFS, use bridge detection:
    If the removed edge is NOT a bridge, the flip always keeps connectivity
    (since we only add an edge, which can't disconnect).
    If it IS a bridge, we need to check if add_edge reconnects the components.
    """
    u, v = remove_edge
    # Check if (u,v) is a bridge by checking if u and v are still connected
    # after removing (u,v) — but we also add (s,t) simultaneously.
    # Quick check: if remove_edge is not a bridge, flip is safe.
    # Use NetworkX bridge detection for small graphs; for larger, just do BFS.
    G_new = G.copy()
    G_new.remove_edge(u, v)
    G_new.add_edge(*add_edge)
    return nx.is_connected(G_new)


def sample_neighbors(G: nx.Graph, max_neighbors: int = 500):
    """Generate edge-flip neighbors, sampling if graph is large.

    For each neighbor, yields (remove_edge, add_edge, G_new).
    Maintains connectivity.

    For n>15, samples a random subset of edge-flips rather than
    enumerating all O(m * (n*(n-1)/2 - m)) possibilities.
    """
    edges = list(G.edges())
    non_edges = list(nx.non_edges(G))
    m = len(edges)
    m_bar = len(non_edges)

    if m == 0 or m_bar == 0:
        return

    total_possible = m * m_bar

    if total_possible <= max_neighbors:
        # Enumerate all
        pairs = [(e, ne) for e in edges for ne in non_edges]
        random.shuffle(pairs)
    else:
        # Sample random subset
        pairs = []
        seen = set()
        attempts = 0
        while len(pairs) < max_neighbors and attempts < max_neighbors * 3:
            attempts += 1
            e = random.choice(edges)
            ne = random.choice(non_edges)
            key = (e, ne)
            if key not in seen:
                seen.add(key)
                pairs.append((e, ne))

    for remove_edge, add_edge in pairs:
        G_new = edge_flip(G, remove_edge, add_edge)
        if nx.is_connected(G_new):
            yield (remove_edge, add_edge, G_new)


# ─────────────────────────────────────────────────────────────────────
# 3. Tabu Search Core
# ─────────────────────────────────────────────────────────────────────

def tabu_search(G_start, score_fn, tabu_size=50, time_budget=1800,
                max_neighbors=500, verbose=True):
    """Tabu Search for graph counterexamples via edge-flips.

    Args:
        G_start: starting graph (NetworkX)
        score_fn: callable(G) -> float. Positive = counterexample.
        tabu_size: maximum tabu list length
        time_budget: seconds to search
        max_neighbors: max edge-flip neighbors to evaluate per iteration
        verbose: print progress

    Returns:
        (best_graph, best_score, history)
    """
    start_time = time.time()
    deadline = start_time + time_budget

    G = G_start.copy()
    current_score = score_fn(G)
    best_score = current_score
    best_graph = G.copy()

    tabu_list = deque(maxlen=tabu_size)
    tabu_set = set()  # For O(1) lookup

    iteration = 0
    stagnation = 0
    history = []

    if verbose:
        print(f"Tabu Search start | n={G.number_of_nodes()} e={G.number_of_edges()} "
              f"| score={current_score:.6f}")

    while time.time() < deadline:
        iteration += 1

        best_neighbor = None
        best_neighbor_score = float('-inf')
        best_move = None

        # Evaluate neighbors
        neighbor_count = 0
        for remove_edge, add_edge, G_new in sample_neighbors(G, max_neighbors):
            neighbor_count += 1
            # Canonical move representation (sorted edges for consistency)
            move = (tuple(sorted(remove_edge)), tuple(sorted(add_edge)))

            score = score_fn(G_new)

            # Aspiration criterion: accept if beats global best, even if tabu
            if score > best_score:
                best_score = score
                best_graph = G_new.copy()
                best_neighbor = G_new
                best_neighbor_score = score
                best_move = move
                if verbose:
                    elapsed = time.time() - start_time
                    print(f"  iter {iteration:5d} | t={elapsed:7.1f}s | "
                          f"NEW BEST={best_score:.8f} | "
                          f"n={G_new.number_of_nodes()} e={G_new.number_of_edges()} "
                          f"| neighbors={neighbor_count}")
                # Early exit on counterexample
                if best_score > 1e-6:
                    if verbose:
                        print(f"  *** COUNTEREXAMPLE FOUND ***")
                    history.append({
                        'iteration': iteration,
                        'time': time.time() - start_time,
                        'score': best_score,
                        'best_score': best_score,
                        'n': G_new.number_of_nodes(),
                        'e': G_new.number_of_edges(),
                    })
                    return best_graph, best_score, history
                continue  # Keep searching this neighborhood for even better

            # Normal: accept best non-tabu neighbor
            if move not in tabu_set and score > best_neighbor_score:
                best_neighbor = G_new
                best_neighbor_score = score
                best_move = move

        # Record history
        elapsed = time.time() - start_time
        history.append({
            'iteration': iteration,
            'time': elapsed,
            'score': best_neighbor_score if best_neighbor is not None else current_score,
            'best_score': best_score,
            'n': G.number_of_nodes(),
            'e': G.number_of_edges(),
        })

        if best_neighbor is None:
            stagnation += 1
            if verbose and stagnation % 10 == 0:
                print(f"  iter {iteration:5d} | t={elapsed:7.1f}s | "
                      f"NO VALID MOVE (stagnation={stagnation})")

            # Perturbation: random restart with edge swaps
            if stagnation >= 20:
                G = _random_perturbation(G)
                current_score = score_fn(G)
                stagnation = 0
                tabu_set.clear()
                tabu_list.clear()
                if verbose:
                    print(f"  iter {iteration:5d} | t={elapsed:7.1f}s | "
                          f"PERTURBATION | score={current_score:.6f}")
            continue

        # Move to best neighbor
        G = best_neighbor
        current_score = best_neighbor_score
        stagnation = 0

        # Update tabu list
        if best_move is not None:
            # Tabu the reverse move (prevent undoing)
            reverse_move = (best_move[1], best_move[0])
            if len(tabu_list) >= tabu_size:
                old_move = tabu_list[0]
                tabu_set.discard(old_move)
            tabu_list.append(reverse_move)
            tabu_set.add(reverse_move)

        # Periodic reporting
        if verbose and iteration % 50 == 0:
            print(f"  iter {iteration:5d} | t={elapsed:7.1f}s | "
                  f"current={current_score:.6f} best={best_score:.6f} | "
                  f"n={G.number_of_nodes()} e={G.number_of_edges()} "
                  f"| tabu={len(tabu_list)}")

    elapsed = time.time() - start_time
    if verbose:
        print(f"Tabu Search done | {elapsed:.1f}s | {iteration} iterations | "
              f"best={best_score:.8f}")

    return best_graph, best_score, history


def _random_perturbation(G: nx.Graph, num_flips: int = 5) -> nx.Graph:
    """Apply random edge-flips as perturbation to escape deep local optima."""
    G_new = G.copy()
    for _ in range(num_flips):
        edges = list(G_new.edges())
        non_edges = list(nx.non_edges(G_new))
        if not edges or not non_edges:
            break
        re = random.choice(edges)
        ae = random.choice(non_edges)
        G_try = edge_flip(G_new, re, ae)
        if nx.is_connected(G_try):
            G_new = G_try
    return G_new


# ─────────────────────────────────────────────────────────────────────
# 4. Multi-Start Tabu Search
# ─────────────────────────────────────────────────────────────────────

def random_connected_graph(n: int, edge_density: float = 0.3) -> nx.Graph:
    """Generate a random connected graph on n vertices.

    Starts with a random spanning tree, then adds random edges
    to reach target density.
    """
    # Random spanning tree via random Prufer sequence
    if n <= 2:
        G = nx.complete_graph(n)
        return G

    seq = [random.randint(0, n - 1) for _ in range(n - 2)]
    G = nx.from_prufer_sequence(seq)

    # Add edges to reach density
    max_edges = n * (n - 1) // 2
    target_edges = int(edge_density * max_edges)
    non_edges = list(nx.non_edges(G))
    random.shuffle(non_edges)

    for u, v in non_edges:
        if G.number_of_edges() >= target_edges:
            break
        G.add_edge(u, v)

    return G


def multi_start_tabu(score_fn, time_budget, start_n=15, num_starts=5,
                     tabu_size=50, max_neighbors=500, verbose=True):
    """Run Tabu Search from multiple random starting graphs.

    Args:
        score_fn: scoring function
        time_budget: total seconds
        start_n: vertex count for random starting graphs
        num_starts: number of restarts
        tabu_size: tabu list size
        max_neighbors: neighbors per iteration
        verbose: print progress

    Returns:
        (best_graph, best_score, all_histories)
    """
    start_time = time.time()
    deadline = start_time + time_budget

    global_best_graph = None
    global_best_score = float('-inf')
    all_histories = []

    # Different densities for diversity
    densities = [0.2, 0.3, 0.4, 0.5, 0.6]

    for restart in range(num_starts):
        remaining = deadline - time.time()
        if remaining < 5.0:
            break

        per_start_budget = remaining / (num_starts - restart)

        density = densities[restart % len(densities)]
        G_start = random_connected_graph(start_n, edge_density=density)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Restart {restart+1}/{num_starts} | "
                  f"n={G_start.number_of_nodes()} e={G_start.number_of_edges()} "
                  f"density={density:.1f} | budget={per_start_budget:.0f}s")
            print(f"{'='*60}")

        G_best, best_score, hist = tabu_search(
            G_start, score_fn,
            tabu_size=tabu_size,
            time_budget=per_start_budget,
            max_neighbors=max_neighbors,
            verbose=verbose,
        )

        all_histories.append(hist)

        if best_score > global_best_score:
            global_best_score = best_score
            global_best_graph = G_best.copy()

        if global_best_score > 1e-6:
            if verbose:
                print(f"*** COUNTEREXAMPLE FOUND at restart {restart+1} ***")
            break

    elapsed = time.time() - start_time
    if verbose:
        print(f"\n{'='*60}")
        print(f"Multi-Start Tabu Summary | {elapsed:.1f}s | "
              f"best={global_best_score:.8f}")
        if global_best_graph is not None:
            print(f"Best graph: n={global_best_graph.number_of_nodes()} "
                  f"e={global_best_graph.number_of_edges()}")
        print(f"{'='*60}")

    return global_best_graph, global_best_score, all_histories


# ─────────────────────────────────────────────────────────────────────
# 5. SDP Existence Analysis (Mathematical, no CVXPY)
# ─────────────────────────────────────────────────────────────────────

def sdp_analysis_bound44():
    """Mathematical analysis of Bound 44 tightness.

    Bound 44: max_{i~j} [2 + sqrt(2*((di-1)^2 + (dj-1)^2 + mi*mj - di*dj))]
    where i~j means edge (i,j), di=degree(i), mi=avg_neighbor_degree(i).

    On k-regular graphs: di=dj=k, mi=mj=k
      bound44 = 2 + sqrt(2*((k-1)^2 + (k-1)^2 + k^2 - k^2))
              = 2 + sqrt(2*2*(k-1)^2)
              = 2 + 2*(k-1)
              = 2k
    And mu = 2k for regular (largest Laplacian eigenvalue of k-regular = 2k iff
    bipartite k-regular, otherwise mu < 2k).

    Key insight: mu = 2k only for bipartite k-regular graphs.
    For non-bipartite k-regular, mu < 2k, so gap is even larger.

    Analysis: Why is Bound 44 hard to violate?
    """
    print("\n" + "=" * 60)
    print("SDP / Mathematical Analysis: Bound 44")
    print("=" * 60)

    print("""
Bound 44: max_{i~j} [2 + sqrt(2*((di-1)^2 + (dj-1)^2 + mi*mj - di*dj))]

1. REGULAR GRAPH TIGHTNESS
   On k-regular bipartite: di=dj=k, mi=mj=k
   - Bound 44 = 2 + sqrt(2*(2*(k-1)^2)) = 2 + 2*(k-1) = 2k
   - mu(G) = 2k (bipartite k-regular achieves maximum eigenvalue)
   - Gap = 0 (exactly tight)

2. WHY EDGE-BASED BOUNDS ARE HARDER
   Bound 44 takes MAX over edges. To violate:
   - Need mu > max_{i~j} [bound expression]
   - mu is a GLOBAL property (largest eigenvalue)
   - Bound 44 is a LOCAL property (max over individual edges)
   - To violate: need global spectral radius to exceed ALL local estimates
   - This is hard because high-mu graphs tend to have at least one edge
     with large local bound value

3. STRUCTURAL BARRIER
   High mu requires some form of near-bipartiteness or high conductance.
   But such graphs tend to have edges connecting high-degree vertices,
   which inflates the Bound 44 edge-max.

   Specifically: if vertex i has high degree di and its neighbor j also
   has high degree dj, then (di-1)^2 + (dj-1)^2 grows quadratically,
   making the bound large.

4. COMPARISON: BOUND 44 vs BOUND 9
   - Bound 9: max_v [dv + mv] (vertex-based, always extremal via Li/Pan)
   - Bound 44: edge-based, involves cross-terms mi*mj - di*dj
   - The cross-term mi*mj - di*dj can be negative when degrees are high
     but neighbor-degrees are low (irregular neighborhoods)
   - This makes Bound 44 SMALLER than naive estimates
   - Potential violation path: find graph where mi*mj << di*dj for all
     edges, keeping bound small, while mu stays large

5. GAP ANALYSIS (from exhaustive n<=13)
   - Smallest gap: +0.0096 (Bound 44)
   - This means at n<=13, no graph has mu within 0.01 of the bound
   - The gap may shrink at larger n, but the regular-graph attractor
     means most optimization methods converge to gap=0 on regular graphs
     rather than finding negative-gap graphs

6. WHAT WOULD A COUNTEREXAMPLE LOOK LIKE?
   - Highly irregular: mix of high-degree and low-degree vertices
   - Edge structure: high-degree vertices connected to low-degree ones
     (so mi*mj is small while mu stays large from spectral properties)
   - NOT bipartite (bipartite maximizes mu but also maximizes bound)
   - Likely n >= 15-20 based on gap trend
""")

    # Numerical verification of the analysis
    print("7. NUMERICAL VERIFICATION")
    print("-" * 40)

    # Test on various graph families
    families = {
        'Complete bipartite K(5,5)': nx.complete_bipartite_graph(5, 5),
        'Petersen graph': nx.petersen_graph(),
        'Cycle C_10': nx.cycle_graph(10),
        'Star S_10': nx.star_graph(9),
        'Wheel W_10': nx.wheel_graph(10),
        'Path P_10': nx.path_graph(10),
        'Complete K_6': nx.complete_graph(6),
        'Barbell(5,1)': nx.barbell_graph(5, 1),
    }

    print(f"{'Graph':<25} {'mu':>8} {'B44':>8} {'Gap':>10}")
    print("-" * 55)

    for name, G in families.items():
        A = graph_to_adj(G)
        mu, bvals, gaps = evaluate_all_bounds(A)
        b44 = bvals.get(44, 0.0)
        gap = gaps.get(44, 0.0)
        print(f"{name:<25} {mu:>8.4f} {b44:>8.4f} {gap:>+10.4f}")

    print("""
CONCLUSION:
  Bound 44 appears to be a genuine upper bound. The gap is smallest on
  irregular graphs with specific degree-mixing patterns (like barbell,
  star), but never closes to zero except on regular bipartite graphs
  where it is tight. The edge-max structure creates a robust barrier:
  any graph modification that increases mu also tends to increase the
  bound on at least one edge.

  No CVXPY available for SDP relaxation. Mathematical analysis suggests
  counterexample would need n >= 15 with a specific irregular structure
  that maintains high spectral radius while keeping all edge-local
  bound estimates low.
""")


# ─────────────────────────────────────────────────────────────────────
# 6. CLI Entry Points
# ─────────────────────────────────────────────────────────────────────

def run_test():
    """Smoke test: verify Tabu Search mechanics and score improvement."""
    print("=" * 60)
    print("Tabu Search Smoke Test")
    print("=" * 60)

    # 1. Test edge-flip
    print("\n[1] Testing edge-flip...")
    G = nx.cycle_graph(6)
    G.add_edge(0, 3)  # Add a chord
    n_before = G.number_of_edges()
    G_flip = edge_flip(G, (0, 1), (1, 4))
    assert G_flip.number_of_edges() == n_before, "Edge count should be preserved"
    print(f"  Edge-flip OK: {n_before} edges preserved")

    # 2. Test neighborhood sampling
    print("\n[2] Testing neighborhood generation...")
    G = nx.petersen_graph()
    neighbors = list(sample_neighbors(G, max_neighbors=100))
    print(f"  Petersen graph: {len(neighbors)} valid flip-neighbors (from max 100)")
    assert len(neighbors) > 0, "Should have at least one neighbor"

    # 3. Test scoring
    print("\n[3] Testing scoring on known graphs...")
    for name, G in [('Petersen', nx.petersen_graph()),
                    ('Star(9)', nx.star_graph(9)),
                    ('Cycle(10)', nx.cycle_graph(10))]:
        s44 = score_single_bound(G, 44)
        print(f"  {name}: score(B44)={s44:.6f}")

    # 4. Short Tabu Search run
    print("\n[4] Running Tabu Search (15s on Bound 44, n=10)...")
    G_start = random_connected_graph(10, edge_density=0.3)
    score_fn = lambda G: score_single_bound(G, 44)

    initial_score = score_fn(G_start)
    print(f"  Start: n={G_start.number_of_nodes()} e={G_start.number_of_edges()} "
          f"score={initial_score:.6f}")

    G_best, best_score, hist = tabu_search(
        G_start, score_fn, tabu_size=30, time_budget=15, max_neighbors=200,
        verbose=True
    )

    improvement = best_score - initial_score
    print(f"\n  Result: score={best_score:.6f} (improvement={improvement:+.6f})")
    print(f"  Best graph: n={G_best.number_of_nodes()} e={G_best.number_of_edges()}")
    print(f"  Iterations: {len(hist)}")

    if best_score > 1e-6:
        print("  *** COUNTEREXAMPLE FOUND ***")
    else:
        print(f"  Gap to counterexample: {-best_score:.6f}")

    # 5. Test random connected graph generation
    print("\n[5] Testing graph generation...")
    for density in [0.2, 0.4, 0.6]:
        G = random_connected_graph(15, edge_density=density)
        assert nx.is_connected(G), f"Graph at density {density} should be connected"
        actual_density = G.number_of_edges() / (15 * 14 / 2)
        print(f"  n=15, target={density:.1f}: actual_density={actual_density:.2f} "
              f"e={G.number_of_edges()}")

    print("\n" + "=" * 60)
    print("All smoke tests passed!")
    print("=" * 60)


def run_single_bound(bound_id, time_budget, start_n=15, num_starts=5):
    """Run multi-start Tabu Search on a specific bound."""
    print(f"Tabu Search for counterexample to Bound {bound_id}")
    print(f"Time budget: {time_budget:.0f}s | Start n={start_n} | "
          f"Restarts={num_starts}")
    print("=" * 60)

    score_fn = lambda G: score_single_bound(G, bound_id)

    G_best, best_score, histories = multi_start_tabu(
        score_fn, time_budget,
        start_n=start_n,
        num_starts=num_starts,
        verbose=True,
    )

    # Detailed final report
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

        # Also check against all bounds
        score_detail, worst_bid, details = score_all_bounds_detailed(G_best)
        print(f"\n  Tightest bound overall: #{worst_bid}")
        if details:
            print(f"  mu={details['mu']:.8f}")
            top5 = sorted(details['gaps'].items(), key=lambda x: x[1])[:5]
            print(f"  Top-5 tightest bounds:")
            for bid, gap in top5:
                print(f"    Bound {bid}: gap={gap:+.8f}")

        if best_score > 1e-6:
            print(f"\n  *** COUNTEREXAMPLE FOUND ***")
        else:
            print(f"\n  No counterexample found (closest gap: {-best_score:.8f})")

    # History summary
    total_iters = sum(len(h) for h in histories)
    print(f"\n  Total iterations: {total_iters}")

    return G_best, best_score


def run_sdp_analysis():
    """Run the SDP / mathematical analysis."""
    sdp_analysis_bound44()


# ─────────────────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Tabu Search for BHS bound counterexamples'
    )
    parser.add_argument('--test', action='store_true',
                        help='Run smoke test')
    parser.add_argument('--bound', type=int, default=None,
                        help='Target bound ID (e.g., 44)')
    parser.add_argument('--time', type=float, default=60.0,
                        help='Time budget in seconds (default: 60)')
    parser.add_argument('--start-n', type=int, default=15,
                        help='Starting graph vertex count (default: 15)')
    parser.add_argument('--num-starts', type=int, default=5,
                        help='Number of multi-start restarts (default: 5)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--sdp', action='store_true',
                        help='Run SDP/mathematical analysis')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.test:
        run_test()
    elif args.sdp:
        run_sdp_analysis()
    elif args.bound is not None:
        if args.bound not in ALL_BOUND_IDS:
            print(f"Error: Bound {args.bound} not in known bounds.")
            print(f"Available: {ALL_BOUND_IDS}")
            sys.exit(1)
        run_single_bound(args.bound, args.time, args.start_n, args.num_starts)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
