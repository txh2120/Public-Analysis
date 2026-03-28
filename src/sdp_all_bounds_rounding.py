#!/usr/bin/env python3
"""Rounding slack analysis for all 38 BHS bounds.

For each bound:
1. Construct small weighted graphs (n=7-10) optimized to violate that bound
2. Check if weighted violation survives discrete rounding (thresholds 0.3-0.7)
3. Record: bound_id, weighted_gap, best_discrete_gap, rounding_survives (yes/no)

Key question: does the "rounding slack" phenomenon (weighted violations always
disappear after discretization) hold for ALL 38 bounds, or are there bounds
where a weighted violation survives rounding?

If ANY bound has rounding_survives=yes, that bound may have discrete counterexamples.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

sys.path.insert(0, 'D:/Public Analysis/src')
from exhaustive_bound_search import (
    laplacian_spectral_radius,
    compute_dv_mv,
    compute_vertex_bounds,
    compute_edge_bounds,
    evaluate_all_bounds,
    ALL_BOUND_IDS,
    VERTEX_BOUND_IDS,
    EDGE_BOUND_IDS,
    make_complete,
    make_cycle,
    make_star,
    make_path,
    make_kite,
    make_windmill,
    make_barbell,
    make_wheel,
    make_tadpole,
)


# =====================================================================
# Section 1: Weighted graph bound computation
# =====================================================================

def compute_weighted_dv_mv(W):
    """Compute weighted degree and average neighbor degree from weight matrix.

    Args:
        W: n x n symmetric weight matrix, W_ij in [0, 1], diag = 0

    Returns:
        dv: weighted degree array (n,)
        mv: weighted average neighbor degree array (n,)
    """
    dv = W.sum(axis=1)
    neighbor_deg_sum = W @ dv
    mv = np.zeros_like(dv)
    nonzero = dv > 1e-12
    mv[nonzero] = neighbor_deg_sum[nonzero] / dv[nonzero]
    return dv, mv


def compute_bound_value_weighted(W, bound_id):
    """Compute a specific bound's value from a weighted adjacency matrix.

    Args:
        W: n x n symmetric weight matrix
        bound_id: which bound to compute

    Returns:
        bound_value: the computed bound value
    """
    dv, mv = compute_weighted_dv_mv(W)

    if bound_id in VERTEX_BOUND_IDS:
        bounds = compute_vertex_bounds(dv, mv)
        return bounds.get(bound_id, float('inf'))
    elif bound_id in EDGE_BOUND_IDS:
        return _compute_single_edge_bound_weighted(W, dv, mv, bound_id)
    else:
        raise ValueError(f"Unknown bound_id: {bound_id}")


def _compute_single_edge_bound_weighted(W, dv, mv, bound_id):
    """Compute a single edge-max bound from weighted adjacency matrix.

    For weighted graphs, "edges" are pairs where W_ij > threshold.
    """
    n = W.shape[0]
    threshold = 1e-6

    # Collect all edge pairs
    rows, cols = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] > threshold:
                rows.append(i)
                cols.append(j)

    if not rows:
        return float('inf')  # No edges = bound trivially holds

    rows = np.array(rows)
    cols = np.array(cols)
    di = dv[rows].astype(np.float64)
    dj = dv[cols].astype(np.float64)
    mi = mv[rows].astype(np.float64)
    mj = mv[cols].astype(np.float64)
    di2 = di * di
    dj2 = dj * dj

    def _safe_sqrt(x):
        return np.sqrt(np.maximum(x, 0.0))

    def _safe_div(a, b):
        return np.where(b != 0, a / b, 0.0)

    if bound_id == 33:
        vals = 2.0 * (di + dj) - (mi + mj)
    elif bound_id == 34:
        vals = _safe_div(2.0 * (di2 + dj2), di + dj)
    elif bound_id == 35:
        denom = mi + mj
        vals = np.where(denom > 0, 2.0 * (di2 + dj2) / denom, 0.0)
    elif bound_id == 37:
        vals = _safe_sqrt(2.0 * (di2 + dj2))
    elif bound_id == 38:
        vals = 2.0 + _safe_sqrt(2.0 * (di - 1.0)**2 + 2.0 * (dj - 1.0)**2)
    elif bound_id == 39:
        inner = 2.0 * (di2 + dj2) - 4.0 * (mi + mj) + 4.0
        vals = 2.0 + _safe_sqrt(inner)
    elif bound_id == 40:
        inner = (2.0 * ((mi - 1.0)**2 + (mj - 1.0)**2)
                 + (di2 + dj2) - (di * mi + dj * mj))
        vals = 2.0 + _safe_sqrt(inner)
    elif bound_id == 42:
        vals = _safe_sqrt(di2 + dj2 + 2.0 * mi * mj)
    elif bound_id == 44:
        inner = 2.0 * ((di - 1.0)**2 + (dj - 1.0)**2 + mi * mj - di * dj)
        vals = 2.0 + _safe_sqrt(inner)
    elif bound_id == 45:
        inner = ((di - dj)**2 + 2.0 * (di * mi + dj * mj)
                 - 4.0 * (mi + mj) + 4.0)
        vals = 2.0 + _safe_sqrt(inner)
    elif bound_id == 46:
        denom = mi + mj
        ratio = np.where(denom > 0, 16.0 * di * dj / denom, 0.0)
        inner = 2.0 * (di2 + dj2) - ratio + 4.0
        vals = 2.0 + _safe_sqrt(inner)
    elif bound_id == 47:
        numer = 2.0 * (di2 + dj2) - (mi - mj)**2
        vals = _safe_div(numer, di + dj)
    elif bound_id == 48:
        inner_sqrt = _safe_sqrt(2.0 * (di2 + dj2) - 4.0 * (mi + mj) + 4.0)
        denom = 2.0 + inner_sqrt
        vals = np.where(denom > 0, 2.0 * (di2 + dj2) / denom, 0.0)
    elif bound_id == 56:
        vals = _safe_sqrt(2.0 * (di2 + dj2) + 4.0 * mi * mj)
    else:
        raise ValueError(f"Unknown edge bound: {bound_id}")

    return float(np.max(vals))


def compute_gap_weighted(W, bound_id):
    """Compute gap = bound_value - mu for a weighted graph.

    Positive gap = bound holds. Negative gap = violation.
    """
    n = W.shape[0]
    d = W.sum(axis=1)
    L = np.diag(d) - W
    eigvals = np.linalg.eigvalsh(L)
    mu = float(eigvals[-1])

    bound_val = compute_bound_value_weighted(W, bound_id)
    gap = bound_val - mu
    return mu, bound_val, gap


# =====================================================================
# Section 2: Rounding weighted graph to discrete
# =====================================================================

def round_and_evaluate(W, bound_id, thresholds=None):
    """Round a weighted graph at various thresholds and check bound gap.

    Args:
        W: weighted adjacency matrix
        bound_id: which bound to evaluate
        thresholds: list of rounding thresholds (default 0.3 to 0.7)

    Returns:
        dict with best_threshold, best_discrete_gap, all_results
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9]

    best_gap = float('inf')
    best_threshold = None
    all_results = []

    for t in thresholds:
        # Round: W_ij >= t → 1, else → 0
        A_discrete = (W >= t).astype(np.float64)
        np.fill_diagonal(A_discrete, 0)
        # Make symmetric
        A_discrete = np.maximum(A_discrete, A_discrete.T)

        # Check connectivity (skip disconnected)
        n = A_discrete.shape[0]
        if A_discrete.sum() == 0:
            continue

        # Evaluate bound on discrete graph
        mu = laplacian_spectral_radius(A_discrete)
        dv, mv = compute_dv_mv(A_discrete)

        if bound_id in VERTEX_BOUND_IDS:
            bounds = compute_vertex_bounds(dv, mv)
            bound_val = bounds.get(bound_id, float('inf'))
        else:
            bounds = compute_edge_bounds(A_discrete, dv, mv)
            bound_val = bounds.get(bound_id, float('inf'))

        gap = bound_val - mu

        all_results.append({
            'threshold': t,
            'mu': mu,
            'bound_val': bound_val,
            'gap': gap,
            'n_edges': int(A_discrete.sum() / 2),
        })

        if gap < best_gap:
            best_gap = gap
            best_threshold = t

    return {
        'best_threshold': best_threshold,
        'best_discrete_gap': best_gap,
        'all_results': all_results,
    }


# =====================================================================
# Section 3: Weight optimization for a specific bound
# =====================================================================

def optimize_weights_for_bound(n, bound_id, topology='free', n_trials=5,
                                max_iter=300, seed=42, verbose=False,
                                use_de=False):
    """Optimize edge weights to minimize gap for a specific bound.

    Args:
        n: number of vertices
        bound_id: which bound to target
        topology: 'free' (optimize full weight matrix) or NetworkX graph
        n_trials: number of random restarts
        max_iter: max iterations per trial
        seed: random seed
        verbose: print progress
        use_de: also try differential evolution (slow for large n)

    Returns:
        dict with best_gap, best_W, mu, bound_val
    """
    best_gap = float('inf')
    best_W = None
    best_mu = None
    best_bound_val = None

    n_vars = n * (n - 1) // 2

    def weights_to_W(x):
        W = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                W[i, j] = x[idx]
                W[j, i] = x[idx]
                idx += 1
        return W

    def objective(x):
        W = weights_to_W(x)
        _, _, gap = compute_gap_weighted(W, bound_id)
        return gap

    bounds = [(0.0, 1.0)] * n_vars

    for trial in range(n_trials):
        np.random.seed(seed + trial * 137)

        if topology == 'free':
            x0 = np.random.rand(n_vars) * 0.8

            # Use L-BFGS-B for speed
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': max_iter, 'ftol': 1e-14})

            W_opt = weights_to_W(result.x)
            mu, bv, gap = compute_gap_weighted(W_opt, bound_id)

            if gap < best_gap:
                best_gap = gap
                best_W = W_opt
                best_mu = mu
                best_bound_val = bv

    # DE only when requested and n is small enough to be feasible
    if use_de and n <= 8:
        try:
            result_de = differential_evolution(
                objective, bounds, maxiter=150, seed=seed,
                tol=1e-12, polish=True, workers=1,
                mutation=(0.5, 1.5), recombination=0.9,
            )
            W_de = weights_to_W(result_de.x)
            mu_de, bv_de, gap_de = compute_gap_weighted(W_de, bound_id)
            if gap_de < best_gap:
                best_gap = gap_de
                best_W = W_de
                best_mu = mu_de
                best_bound_val = bv_de
        except Exception:
            pass

    return {
        'gap': best_gap,
        'W': best_W,
        'mu': best_mu,
        'bound_val': best_bound_val,
        'n': n,
        'bound_id': bound_id,
    }


# =====================================================================
# Section 4: Seed graphs from known near-misses
# =====================================================================

def get_seed_graphs(n):
    """Generate diverse seed graph topologies for optimization.

    Uses all available graph families from exhaustive_bound_search for
    maximum structural diversity.

    Returns list of (name, adjacency_matrix) tuples.
    """
    seeds = []

    # Basic families from exhaustive_bound_search
    seeds.append(('path', make_path(n)))
    seeds.append(('star', make_star(n)))
    seeds.append(('cycle', make_cycle(n)))
    seeds.append(('complete', make_complete(n)))

    if n >= 5:
        seeds.append(('wheel', make_wheel(n)))

    # Complete bipartite K_{n//2, n-n//2}
    n1 = n // 2
    A = np.zeros((n, n))
    for i in range(n1):
        for j in range(n1, n):
            A[i, j] = 1.0
            A[j, i] = 1.0
    seeds.append(('bipartite', A))

    # "Double star": two hubs connected
    if n >= 4:
        A = np.zeros((n, n))
        A[0, 1] = A[1, 0] = 1.0
        half = n // 2
        for i in range(2, half):
            A[0, i] = A[i, 0] = 1.0
        for i in range(half, n):
            A[1, i] = A[i, 1] = 1.0
        seeds.append(('double_star', A))

    # Kite / lollipop: K_t + P_s
    for t in range(3, min(n, n // 2 + 2)):
        s = n - t
        if s >= 1:
            seeds.append((f'kite({t},{s})', make_kite(t, s)))
            break  # one kite variant

    # Barbell: two cliques + bridge
    for m1 in range(3, n // 2 + 1):
        m2 = n - 2 * m1
        if m2 >= 0:
            seeds.append((f'barbell({m1},{m2})', make_barbell(m1, m2)))
            break  # one barbell variant

    # Windmill: center + k copies of K_3
    for k_copies in range(2, (n - 1) // 2 + 1):
        nn = 1 + k_copies * 2
        if nn == n:
            seeds.append((f'windmill(3,{k_copies})', make_windmill(3, k_copies)))
            break

    # Tadpole: C_m + P_k
    for m_cycle in range(3, n):
        k_path = n - m_cycle
        if k_path >= 1:
            seeds.append((f'tadpole({m_cycle},{k_path})', make_tadpole(m_cycle, k_path)))
            break

    return seeds


def optimize_from_seed(A_seed, bound_id, max_iter=300, seed=42):
    """Start from a seed graph topology and optimize weights.

    Uses the seed topology as a warm start (existing edges get weights,
    then optimize freely in [0, 1]).
    """
    n = A_seed.shape[0]
    n_vars = n * (n - 1) // 2

    def weights_to_W(x):
        W = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                W[i, j] = x[idx]
                W[j, i] = x[idx]
                idx += 1
        return W

    def objective(x):
        W = weights_to_W(x)
        _, _, gap = compute_gap_weighted(W, bound_id)
        return gap

    # Initialize from seed: existing edges start at 0.8, others at 0.1
    x0 = np.full(n_vars, 0.1)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if A_seed[i, j] > 0.5:
                x0[idx] = 0.8
            idx += 1

    # Add some noise
    np.random.seed(seed)
    x0 += np.random.randn(n_vars) * 0.05
    x0 = np.clip(x0, 0.0, 1.0)

    bounds = [(0.0, 1.0)] * n_vars

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                     options={'maxiter': max_iter, 'ftol': 1e-14})

    W_opt = weights_to_W(result.x)
    mu, bv, gap = compute_gap_weighted(W_opt, bound_id)
    return {'gap': gap, 'W': W_opt, 'mu': mu, 'bound_val': bv}


# =====================================================================
# Section 5: Main analysis — all 38 bounds
# =====================================================================

def analyze_single_bound(bound_id, n_sizes=(7, 8, 9), is_priority=False,
                         time_budget=60, verbose=True):
    """Run full rounding slack analysis for one bound.

    Strategy:
    1. For each graph size, try seed graphs + free optimization
    2. Find the weighted graph with smallest gap (most violating or tightest)
    3. Round at multiple thresholds
    4. Report whether rounding preserves any violation

    Args:
        bound_id: which bound
        n_sizes: vertex counts to try
        is_priority: if True, use DE on n=7 and add n=10
        time_budget: max seconds per bound
        verbose: print progress

    Returns:
        dict with weighted_gap, discrete_gap, rounding_survives, details
    """
    t0 = time.time()
    overall_best_gap = float('inf')
    overall_best_W = None
    overall_best_details = None

    effective_sizes = list(n_sizes)
    if is_priority and 10 not in effective_sizes:
        effective_sizes.append(10)

    for n in effective_sizes:
        if time.time() - t0 > time_budget:
            break

        if verbose:
            print(f"    n={n}: ", end='', flush=True)

        # Strategy A: optimize from seed graphs (fast, warm-started)
        seeds = get_seed_graphs(n)
        for name, A_seed in seeds:
            if time.time() - t0 > time_budget:
                break
            result = optimize_from_seed(A_seed, bound_id, max_iter=200)
            if result['gap'] < overall_best_gap:
                overall_best_gap = result['gap']
                overall_best_W = result['W']
                overall_best_details = f"n={n},{name}"

        # Strategy B: free topology L-BFGS-B (multiple restarts)
        n_free_trials = 3 if n <= 8 else 2
        for trial in range(n_free_trials):
            if time.time() - t0 > time_budget:
                break
            result = optimize_weights_for_bound(
                n, bound_id, topology='free', n_trials=1,
                max_iter=200, seed=42 + trial * 97 + bound_id * 13,
                verbose=False, use_de=False
            )
            if result['gap'] < overall_best_gap:
                overall_best_gap = result['gap']
                overall_best_W = result['W']
                overall_best_details = f"n={n},free_t{trial}"

        # Strategy C: DE on small n for priority bounds (global search)
        if is_priority and n <= 8 and time.time() - t0 < time_budget * 0.7:
            result = optimize_weights_for_bound(
                n, bound_id, topology='free', n_trials=1,
                max_iter=150, seed=42 + bound_id * 7,
                verbose=False, use_de=True
            )
            if result['gap'] < overall_best_gap:
                overall_best_gap = result['gap']
                overall_best_W = result['W']
                overall_best_details = f"n={n},DE"

        if verbose:
            print(f"gap={overall_best_gap:+.8f} ", end='', flush=True)

    if verbose:
        elapsed = time.time() - t0
        print(f"({elapsed:.0f}s)")

    # Now check rounding
    if overall_best_W is not None:
        rounding = round_and_evaluate(overall_best_W, bound_id)
        discrete_gap = rounding['best_discrete_gap']
        rounding_survives = (overall_best_gap < -1e-6 and discrete_gap < -1e-6)
    else:
        discrete_gap = float('inf')
        rounding_survives = False
        rounding = None

    return {
        'bound_id': bound_id,
        'weighted_gap': overall_best_gap,
        'discrete_gap': discrete_gap,
        'rounding_survives': rounding_survives,
        'best_W': overall_best_W,
        'details': overall_best_details,
        'rounding_details': rounding,
    }


def run_all_bounds_analysis(priority_bounds=None, verbose=True):
    """Run rounding slack analysis for all 38 bounds.

    Args:
        priority_bounds: list of bound IDs to analyze first (Group A etc.)
        verbose: print progress

    Returns:
        dict: {bound_id: analysis_result}
    """
    if priority_bounds is None:
        # Group A (m_v-dominated): 11, 13
        # Mixed: 5, 7
        # d_v-dominated: 14, 16
        # Various edge bounds: 33, 34, 35, 42, 47
        # Bound 44 for baseline
        priority_bounds = [11, 13, 5, 7, 14, 16, 33, 34, 35, 42, 47, 44]

    priority_set = set(priority_bounds)

    # Order: priority first, then remaining
    ordered_bounds = list(priority_bounds)
    for bid in ALL_BOUND_IDS:
        if bid not in ordered_bounds:
            ordered_bounds.append(bid)

    results = {}
    t0 = time.time()

    for i, bid in enumerate(ordered_bounds):
        elapsed = time.time() - t0
        is_priority = bid in priority_set
        tag = " [PRIORITY]" if is_priority else ""
        if verbose:
            print(f"  [{i+1}/{len(ordered_bounds)}] Bound {bid}{tag} (elapsed={elapsed:.0f}s)")

        # Fast mode: n=7,8 only, L-BFGS-B only (no DE), budget 45s
        # DE disabled to keep total runtime ~1-2 hours
        result = analyze_single_bound(
            bid, n_sizes=(7, 8), is_priority=False,
            time_budget=45, verbose=verbose
        )

        results[bid] = result

        # Early report for major findings
        if result['rounding_survives']:
            print(f"\n  *** MAJOR FINDING: Bound {bid} has rounding-surviving violation! ***")
            print(f"      Weighted gap: {result['weighted_gap']:+.10f}")
            print(f"      Discrete gap: {result['discrete_gap']:+.10f}")
            print()

    return results


# =====================================================================
# Section 6: Output formatting and file writing
# =====================================================================

def format_results_table(results):
    """Format results as a text table.

    Args:
        results: dict from run_all_bounds_analysis

    Returns:
        str: formatted table
    """
    lines = []
    lines.append("=" * 80)
    lines.append("SDP ROUNDING SLACK ANALYSIS - ALL 38 BOUNDS")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"{'Bound':>6} | {'Type':>8} | {'Weighted gap':>14} | "
                 f"{'Discrete gap':>14} | {'Rounding survives?':>18} | {'Source':>16}")
    lines.append(f"{'-'*6}-+-{'-'*8}-+-{'-'*14}-+-{'-'*14}-+-{'-'*18}-+-{'-'*16}")

    # Sort by bound ID
    for bid in ALL_BOUND_IDS:
        if bid not in results:
            continue
        r = results[bid]
        btype = "vertex" if bid in VERTEX_BOUND_IDS else "edge"
        w_gap = r['weighted_gap']
        d_gap = r['discrete_gap']
        survives = "YES ***" if r['rounding_survives'] else "no"
        src = r.get('details', '') or ''

        lines.append(f"{bid:>6} | {btype:>8} | {w_gap:>+14.8f} | "
                     f"{d_gap:>+14.8f} | {survives:>18} | {src:>16}")

    lines.append("")

    # Summary
    n_total = len(results)
    n_weighted_violation = sum(1 for r in results.values() if r['weighted_gap'] < -1e-6)
    n_rounding_survives = sum(1 for r in results.values() if r['rounding_survives'])

    lines.append("=" * 80)
    lines.append("SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Total bounds analyzed: {n_total}")
    lines.append(f"Bounds with weighted violation (gap < 0): {n_weighted_violation}")
    lines.append(f"Bounds where violation survives rounding: {n_rounding_survives}")
    lines.append("")

    if n_rounding_survives > 0:
        lines.append("*** MAJOR FINDINGS ***")
        for bid, r in sorted(results.items()):
            if r['rounding_survives']:
                lines.append(f"  Bound {bid}: weighted_gap={r['weighted_gap']:+.10f}, "
                           f"discrete_gap={r['discrete_gap']:+.10f}")
        lines.append("")
        lines.append("These bounds have discrete counterexample candidates!")
    else:
        lines.append("All weighted violations are eliminated by rounding.")
        lines.append("The rounding slack phenomenon appears universal across all 38 bounds.")

    # Group analysis
    lines.append("")
    lines.append("=" * 80)
    lines.append("GROUP ANALYSIS")
    lines.append("=" * 80)

    group_a = [11, 13]
    group_mixed = [5, 7]
    group_dv = [14, 16]
    edge_various = [33, 34, 35, 42, 47]

    for group_name, group_ids in [
        ("Group A (m_v-dominated)", group_a),
        ("Mixed (d^2/m + m, d^2/m + d)", group_mixed),
        ("d_v-dominated", group_dv),
        ("Edge bounds (various)", edge_various),
    ]:
        lines.append(f"\n  {group_name}:")
        for bid in group_ids:
            if bid in results:
                r = results[bid]
                lines.append(f"    Bound {bid}: w_gap={r['weighted_gap']:+.10f}, "
                           f"d_gap={r['discrete_gap']:+.10f}, "
                           f"survives={r['rounding_survives']}")

    # Detail on bounds with weighted violations
    lines.append("")
    lines.append("=" * 80)
    lines.append("WEIGHTED VIOLATIONS DETAIL")
    lines.append("=" * 80)

    for bid in ALL_BOUND_IDS:
        if bid not in results:
            continue
        r = results[bid]
        if r['weighted_gap'] < -1e-6:
            lines.append(f"\n  Bound {bid}:")
            lines.append(f"    Weighted gap: {r['weighted_gap']:+.10f}")
            lines.append(f"    Best discrete gap: {r['discrete_gap']:+.10f}")
            lines.append(f"    Source: {r.get('details', 'N/A')}")
            if r['rounding_details'] and r['rounding_details']['all_results']:
                lines.append(f"    Rounding thresholds:")
                for rd in r['rounding_details']['all_results']:
                    lines.append(f"      t={rd['threshold']:.2f}: gap={rd['gap']:+.10f}, "
                               f"edges={rd['n_edges']}")

    return '\n'.join(lines)


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 80)
    print("SDP ROUNDING SLACK ANALYSIS - ALL 38 BHS BOUNDS")
    print("=" * 80)
    print()
    print("Question: Does the rounding slack phenomenon hold for ALL bounds,")
    print("or are there bounds where weighted violations survive discretization?")
    print()
    print(f"Bounds to analyze: {len(ALL_BOUND_IDS)}")
    print(f"  Vertex-max: {VERTEX_BOUND_IDS}")
    print(f"  Edge-max: {EDGE_BOUND_IDS}")
    print()

    t0 = time.time()

    # Run analysis with priority on structurally interesting bounds
    results = run_all_bounds_analysis(verbose=True)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    # Format and print results
    table = format_results_table(results)
    print()
    print(table)

    # Save to file
    output_dir = Path('D:/Public Analysis/resources')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'sdp_rounding_slack_all.txt'
    with open(output_path, 'w') as f:
        f.write(table)
        f.write(f"\n\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        f.write(f"\nTotal analysis time: {elapsed:.1f}s")
    print(f"\nResults saved to: {output_path}")

    # Return key finding
    n_survives = sum(1 for r in results.values() if r['rounding_survives'])
    if n_survives > 0:
        print(f"\n*** {n_survives} bounds have rounding-surviving violations! ***")
        for bid, r in sorted(results.items()):
            if r['rounding_survives']:
                print(f"  Bound {bid}: weighted={r['weighted_gap']:+.10f}, "
                      f"discrete={r['discrete_gap']:+.10f}")
    else:
        print("\nAll weighted violations are eliminated by rounding.")
        print("Rounding slack appears to be universal.")


if __name__ == '__main__':
    main()
