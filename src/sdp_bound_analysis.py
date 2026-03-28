#!/usr/bin/env python3
"""SDP-based existence analysis for Bound 44 counterexamples.

Cross-domain approach: instead of searching discrete graphs (0/1 adjacency),
relax to weighted graphs (0 <= W_ij <= 1) and use semidefinite programming
to determine if the bound can be violated in the continuous relaxation.

Bound 44: mu(G) <= max_(i~j) 2 + sqrt(2*((d_i-1)^2 + (d_j-1)^2 + m_i*m_j - d_i*d_j))

If the SDP relaxation cannot violate the bound, then discrete violations
are impossible (since discrete graphs are a subset of the relaxed space).

Methods:
1. Fixed-topology weight optimization: optimize edge weights on known graphs
2. Free-topology SDP: optimize both topology and weights simultaneously
3. Analytical verification on known graph families
"""

import sys
import time

import cvxpy as cp
import networkx as nx
import numpy as np

sys.path.insert(0, 'D:/Public Analysis/src')
from exhaustive_bound_search import (
    laplacian_spectral_radius,
    compute_dv_mv,
    compute_edge_bounds,
)


# =====================================================================
# Section 1: Numerical computation of Bound 44 (non-CVXPY, for eval)
# =====================================================================

def compute_bound44_numpy(W):
    """Compute Bound 44 value from a weighted adjacency matrix W.

    Args:
        W: n x n symmetric matrix with 0 <= W_ij <= 1, W_ii = 0

    Returns:
        mu: largest Laplacian eigenvalue
        bound44: the Bound 44 value (max over edges with W_ij > threshold)
        gap: bound44 - mu (positive = bound holds)
    """
    n = W.shape[0]
    d = W.sum(axis=1)  # weighted degree
    L = np.diag(d) - W  # Laplacian
    eigvals = np.linalg.eigvalsh(L)
    mu = float(eigvals[-1])

    # Average neighbor degree: m_i = (W^2)_{ii} / d_i = (W @ d)_i / d_i
    # Actually m_i = sum_j W_ij * d_j / d_i (weighted version)
    neighbor_deg_sum = W @ d
    m = np.zeros(n)
    nonzero = d > 1e-12
    m[nonzero] = neighbor_deg_sum[nonzero] / d[nonzero]

    # Compute Bound 44 over all potential edges (pairs with W_ij > threshold)
    threshold = 1e-6
    best_b44 = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] > threshold:
                di, dj = d[i], d[j]
                mi, mj = m[i], m[j]
                inner = 2.0 * ((di - 1)**2 + (dj - 1)**2 + mi * mj - di * dj)
                if inner < 0:
                    inner = 0.0
                b44 = 2.0 + np.sqrt(inner)
                if b44 > best_b44:
                    best_b44 = b44

    gap = best_b44 - mu
    return mu, best_b44, gap


# =====================================================================
# Section 2: Fixed-topology weight optimization
# =====================================================================

def optimize_weights_fixed_topology(G, verbose=True):
    """Optimize edge weights on a fixed graph topology to minimize Bound 44 gap.

    Given a graph G, optimize w_e in [epsilon, 1] for each edge e to minimize
    (bound44_value - mu). If this reaches 0, a weighted counterexample exists.

    We use a gradient-free approach (scipy.optimize) since the problem involves
    eigenvalues and non-smooth operations (max, sqrt).

    Args:
        G: NetworkX graph
        verbose: print progress

    Returns:
        dict with best weights, mu, bound44, gap
    """
    from scipy.optimize import minimize, differential_evolution

    edges = list(G.edges())
    n = G.number_of_nodes()
    m = len(edges)

    # Map nodes to 0..n-1
    node_map = {v: i for i, v in enumerate(G.nodes())}

    def weights_to_adjacency(weights):
        """Convert edge weight vector to adjacency matrix."""
        W = np.zeros((n, n))
        for k, (u, v) in enumerate(edges):
            i, j = node_map[u], node_map[v]
            W[i, j] = weights[k]
            W[j, i] = weights[k]
        return W

    def objective(weights):
        """Minimize gap = bound44 - mu (want gap -> 0 or negative)."""
        W = weights_to_adjacency(weights)
        mu, b44, gap = compute_bound44_numpy(W)
        return gap  # minimize gap

    # Bounds: each weight in [0.01, 1.0]
    bounds = [(0.01, 1.0)] * m

    if verbose:
        print(f"  Optimizing {m} edge weights on {n}-vertex graph...")

    # Try differential evolution (global optimizer)
    result = differential_evolution(
        objective, bounds, maxiter=500, seed=42, tol=1e-10,
        polish=True, workers=1
    )

    best_weights = result.x
    W_best = weights_to_adjacency(best_weights)
    mu, b44, gap = compute_bound44_numpy(W_best)

    if verbose:
        print(f"  Result: mu={mu:.8f}, B44={b44:.8f}, gap={gap:+.10f}")
        print(f"  Weight range: [{best_weights.min():.4f}, {best_weights.max():.4f}]")
        print(f"  Optimizer: {result.message}")

    return {
        'mu': mu, 'bound44': b44, 'gap': gap,
        'weights': best_weights, 'W': W_best,
        'edges': edges, 'n': n
    }


# =====================================================================
# Section 3: SDP relaxation — maximize spectral radius
# =====================================================================

def sdp_max_spectral_radius(n, solver='CLARABEL', verbose=True):
    """Find the maximum Laplacian spectral radius over all n-vertex weighted graphs.

    Solves: maximize t
    subject to:
        W >= 0, W <= 1, diag(W) = 0, W = W^T
        t*I - L >> 0  (i.e., L << t*I, so mu(L) <= t)

    Wait — this formulation gives t >= mu, so maximizing t is unbounded.
    We need to flip: minimize t subject to L << t*I AND some lower bound on mu.

    Actually, to maximize mu:
    maximize t
    subject to:
        W >= 0, W <= 1, diag(W) = 0, W = W^T
        L - t*I << 0  (equivalent to t >= lambda_max(L))

    But we want to MAXIMIZE t, and t >= mu always, so t can be anything >= mu.
    That's unbounded above.

    The correct approach: maximize lambda_max(L) directly.
    lambda_max(L) = max_x x^T L x / x^T x

    SDP formulation to maximize mu:
    maximize t
    subject to:
        L - t*I + S = 0, S >> 0  [means L << t*I]

    No — this still minimizes t (tightest upper bound on mu).

    Correct: we want the max of lambda_max over choice of W.
    Reformulate as:
    maximize t
    subject to:
        exists unit vector v: v^T L v >= t
        W >= 0, W <= 1, diag(W) = 0, W = W^T

    Equivalently: there exists V >> 0 with trace(V) = 1 such that trace(LV) >= t.
    So: maximize trace(LV)
    subject to: V >> 0, trace(V) = 1, W in [0,1]^{n x n} symmetric, diag(W)=0.

    But trace(LV) = trace((D-W)V) = trace(DV) - trace(WV).
    D = diag(W@1), so trace(DV) = sum_i (W@1)_i * V_ii = sum_i V_ii * sum_j W_ij.
    This is bilinear in W and V — not convex.

    Alternative approach: enumerate over the SDP dual or use alternating optimization.

    Practical: fix V, optimize W. Fix W, optimize V. Alternate.

    Args:
        n: number of vertices
        solver: CVXPY solver name
        verbose: print progress

    Returns:
        dict with best mu, W, details
    """
    if verbose:
        print(f"\n  Alternating optimization for max mu(L) on n={n}...")

    best_mu = 0.0
    best_W = None

    # Multiple random starts
    for trial in range(5):
        # Initialize W randomly
        W_val = np.random.rand(n, n) * 0.5
        W_val = (W_val + W_val.T) / 2
        np.fill_diagonal(W_val, 0)

        for iteration in range(30):
            # Step 1: Given W, find eigenvector for largest eigenvalue
            d = W_val.sum(axis=1)
            L_val = np.diag(d) - W_val
            eigvals, eigvecs = np.linalg.eigh(L_val)
            mu = eigvals[-1]
            v = eigvecs[:, -1]

            # V = v @ v^T (rank-1)
            V_val = np.outer(v, v)

            # Step 2: Given V (rank-1), optimize W to maximize trace(LV)
            # trace(LV) = trace((D-W)V) = sum_i V_ii * d_i - trace(WV)
            # = sum_i V_ii * sum_j W_ij - sum_ij W_ij V_ij
            # = sum_ij W_ij (V_ii - V_ij)
            # Maximize by setting W_ij = 1 if V_ii - V_ij > 0, else 0
            # (since W_ij in [0,1])

            gradient = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    gradient[i, j] = V_val[i, i] - V_val[i, j]

            # Make symmetric: gradient for edge (i,j) is grad[i,j] + grad[j,i]
            sym_gradient = gradient + gradient.T
            np.fill_diagonal(sym_gradient, 0)

            # Set W based on gradient sign
            W_new = np.where(sym_gradient > 0, 1.0, 0.0)
            W_new = (W_new + W_new.T) / 2  # ensure symmetric
            np.fill_diagonal(W_new, 0)

            # Soft update (prevent oscillation)
            alpha = 0.5
            W_val = alpha * W_new + (1 - alpha) * W_val
            W_val = np.clip(W_val, 0, 1)
            W_val = (W_val + W_val.T) / 2
            np.fill_diagonal(W_val, 0)

        # Final evaluation
        d = W_val.sum(axis=1)
        L_val = np.diag(d) - W_val
        eigvals = np.linalg.eigvalsh(L_val)
        mu_final = eigvals[-1]

        if mu_final > best_mu:
            best_mu = mu_final
            best_W = W_val.copy()

        if verbose:
            print(f"    Trial {trial+1}: mu={mu_final:.6f}")

    if verbose:
        print(f"  Best mu = {best_mu:.6f}")

    return {'mu': best_mu, 'W': best_W, 'n': n}


# =====================================================================
# Section 4: SDP for Bound 44 gap minimization (free topology)
# =====================================================================

def sdp_bound44_free_topology(n, n_trials=10, n_iters=50, verbose=True):
    """Search for n-vertex weighted graphs minimizing Bound 44 gap.

    Uses alternating optimization:
    1. Fix W, compute mu and bound44 (and their gradients)
    2. Update W using gradient descent on the gap

    This is heuristic but explores the continuous relaxation space.

    Args:
        n: number of vertices
        n_trials: number of random restarts
        n_iters: iterations per trial
        verbose: print progress

    Returns:
        dict with best gap, mu, bound44, W
    """
    if verbose:
        print(f"\n  Free-topology search for n={n}, {n_trials} trials x {n_iters} iters...")

    best_gap = float('inf')
    best_result = None

    for trial in range(n_trials):
        # Random initialization
        np.random.seed(trial * 137 + 42)
        W = np.random.rand(n, n) * 0.8
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0)

        lr = 0.05  # learning rate for gradient update

        for it in range(n_iters):
            # Compute mu and bound44
            mu, b44, gap = compute_bound44_numpy(W)

            if gap < best_gap:
                best_gap = gap
                best_result = {
                    'mu': mu, 'bound44': b44, 'gap': gap,
                    'W': W.copy(), 'n': n, 'trial': trial, 'iter': it
                }

            # Numerical gradient of gap w.r.t. W
            eps = 1e-5
            grad = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    W[i, j] += eps
                    W[j, i] += eps
                    _, _, gap_plus = compute_bound44_numpy(W)
                    W[i, j] -= 2 * eps
                    W[j, i] -= 2 * eps
                    _, _, gap_minus = compute_bound44_numpy(W)
                    W[i, j] += eps
                    W[j, i] += eps
                    grad[i, j] = (gap_plus - gap_minus) / (2 * eps)
                    grad[j, i] = grad[i, j]

            # Gradient descent step (minimize gap)
            W -= lr * grad
            W = np.clip(W, 0, 1)
            W = (W + W.T) / 2
            np.fill_diagonal(W, 0)

        # Final evaluation
        mu, b44, gap = compute_bound44_numpy(W)
        if verbose and (trial < 3 or trial == n_trials - 1 or gap < best_gap + 0.01):
            print(f"    Trial {trial+1:2d}: mu={mu:.6f}, B44={b44:.6f}, gap={gap:+.8f}")

        if gap < best_gap:
            best_gap = gap
            best_result = {
                'mu': mu, 'bound44': b44, 'gap': gap,
                'W': W.copy(), 'n': n, 'trial': trial, 'iter': n_iters
            }

    if verbose:
        print(f"  Best gap = {best_gap:+.10f}")

    return best_result


# =====================================================================
# Section 5: CVXPY SDP — maximize mu with bound constraint
# =====================================================================

def cvxpy_sdp_check(n, solver='CLARABEL', verbose=True):
    """Use CVXPY SDP to find max mu for n-vertex weighted graphs.

    Formulation:
        maximize t
        subject to:
            W >= 0, W <= 1, W symmetric, diag(W) = 0
            V >> 0, trace(V) = 1
            trace((diag(W@1) - W) @ V) >= t

    The bilinear term trace(D*V) - trace(W*V) is handled by fixing V
    and solving for W (LP), then fixing W and solving for V (SDP),
    alternating.

    But we can do a simpler thing: just maximize the trace of L against
    the top eigenvector space.

    Even simpler: for the SDP formulation, note that for a complete graph
    K_n with all weights = 1, mu = n. So the max mu over weighted graphs
    is n (achieved by K_n). The question is whether Bound 44 value for K_n
    is >= n.

    Args:
        n: number of vertices
        solver: SDP solver
        verbose: print progress

    Returns:
        dict with results
    """
    if verbose:
        print(f"\n  CVXPY SDP check for n={n}...")

    # Approach: use SDP to maximize trace(L @ V) where V >> 0, tr(V)=1
    # and L depends on W. Since it's bilinear, we alternate.

    # But first, let's just check specific known graphs analytically.

    # Complete graph K_n: all weights = 1
    W_kn = np.ones((n, n)) - np.eye(n)
    mu_kn, b44_kn, gap_kn = compute_bound44_numpy(W_kn)

    if verbose:
        print(f"    K_{n}: mu={mu_kn:.6f}, B44={b44_kn:.6f}, gap={gap_kn:+.8f}")

    # Star graph S_n
    W_star = np.zeros((n, n))
    for i in range(1, n):
        W_star[0, i] = 1.0
        W_star[i, 0] = 1.0
    mu_star, b44_star, gap_star = compute_bound44_numpy(W_star)

    if verbose:
        print(f"    S_{n}: mu={mu_star:.6f}, B44={b44_star:.6f}, gap={gap_star:+.8f}")

    # Cycle C_n
    W_cycle = np.zeros((n, n))
    for i in range(n):
        W_cycle[i, (i + 1) % n] = 1.0
        W_cycle[(i + 1) % n, i] = 1.0
    mu_cycle, b44_cycle, gap_cycle = compute_bound44_numpy(W_cycle)

    if verbose:
        print(f"    C_{n}: mu={mu_cycle:.6f}, B44={b44_cycle:.6f}, gap={gap_cycle:+.8f}")

    # Path P_n
    W_path = np.zeros((n, n))
    for i in range(n - 1):
        W_path[i, i + 1] = 1.0
        W_path[i + 1, i] = 1.0
    mu_path, b44_path, gap_path = compute_bound44_numpy(W_path)

    if verbose:
        print(f"    P_{n}: mu={mu_path:.6f}, B44={b44_path:.6f}, gap={gap_path:+.8f}")

    # Now use CVXPY for the actual SDP: maximize lambda_max(L) subject to
    # trace(L) <= bound44 constraint. But since bound44 depends on W nonlinearly,
    # we'll try a linearized approach.

    # Approach: SDP relaxation of "maximize mu subject to mu <= bound44(W)"
    # We solve: maximize t subject to L << t*I, t <= bound44_linearized(W)

    # Since bound44 is hard to express in CVXPY, we instead just find max mu
    # and compare post-hoc.

    # SDP: maximize t
    #   s.t. W >= 0, W <= 1, W symmetric, diag(W) = 0
    #        t*I - (diag(W@1) - W) >> 0

    # Note: D = diag(W@1) depends linearly on W.
    # L = D - W also depends linearly on W.
    # t*I - L >> 0 means t >= lambda_max(L).
    # Maximizing t while t >= lambda_max(L) is unbounded.

    # We need the MINIMUM t such that t >= lambda_max(L) = lambda_max(L).
    # But we want to maximize lambda_max(L) over W.
    # This is: max_W min_t {t : t*I - L >> 0}
    #        = max_W lambda_max(L(W))

    # As a single SDP: max_W lambda_max(L) is not standard.
    # Instead: max_W,t t s.t. t*I - L >> 0  →  unbounded (just increase t)

    # Need to flip: find the W that makes lambda_max(L) largest.
    # lambda_max(L) = max_{||x||=1} x^T L x
    # = max_{V>>0, tr(V)=1} trace(L @ V)

    # So: max_{W, V} trace(L(W) @ V)
    #     s.t. V >> 0, tr(V) = 1, W in [0,1], symmetric, diag 0

    # This is bilinear (L(W) = D(W) - W, and trace(D(W) V) = sum_i V_ii (W 1)_i
    # = sum_{i,j} W_ij V_ii; trace(W V) = sum_{i,j} W_ij V_ij)
    # So trace(LV) = sum_{i,j} W_ij (V_ii - V_ij)

    # Bilinear in W and V → not jointly convex.
    # But for FIXED V, it's linear in W. For FIXED W, it's linear in V.
    # → Alternating maximization.

    # Let's implement this with CVXPY.

    best_mu_sdp = 0.0
    best_W_sdp = None

    for start in range(3):
        # Initialize
        np.random.seed(start * 42 + 7)
        W_cur = np.random.rand(n, n) * 0.5
        W_cur = (W_cur + W_cur.T) / 2
        np.fill_diagonal(W_cur, 0)

        for alt_iter in range(20):
            # Step A: Given W_cur, find V >> 0, tr(V)=1 maximizing trace(L @ V)
            # = find top eigenvector of L
            d_cur = W_cur.sum(axis=1)
            L_cur = np.diag(d_cur) - W_cur
            eigvals_cur, eigvecs_cur = np.linalg.eigh(L_cur)
            mu_cur = eigvals_cur[-1]
            v_top = eigvecs_cur[:, -1]
            V_cur = np.outer(v_top, v_top)  # rank-1 optimal V

            # Step B: Given V_cur, find W in [0,1] symmetric, diag=0
            # maximizing trace(L(W) @ V_cur)
            # = sum_{i,j} W_ij (V_cur[i,i] - V_cur[i,j])

            # Use CVXPY for this step (LP)
            W_var = cp.Variable((n, n), symmetric=True)
            constraints_w = [
                W_var >= 0,
                W_var <= 1,
                cp.diag(W_var) == 0,
            ]

            # Coefficient matrix for W_ij
            C = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    C[i, j] = V_cur[i, i] - V_cur[i, j]
            # Symmetrize
            C = (C + C.T) / 2

            objective_w = cp.Maximize(cp.trace(C @ W_var))
            prob_w = cp.Problem(objective_w, constraints_w)
            try:
                prob_w.solve(solver=solver, verbose=False)
                if prob_w.status in ['optimal', 'optimal_inaccurate']:
                    W_cur = W_var.value.copy()
                    W_cur = (W_cur + W_cur.T) / 2
                    np.fill_diagonal(W_cur, 0)
                    W_cur = np.clip(W_cur, 0, 1)
            except Exception:
                pass

        # Final evaluation
        d_final = W_cur.sum(axis=1)
        L_final = np.diag(d_final) - W_cur
        eigvals_final = np.linalg.eigvalsh(L_final)
        mu_final = eigvals_final[-1]

        if mu_final > best_mu_sdp:
            best_mu_sdp = mu_final
            best_W_sdp = W_cur.copy()

        if verbose:
            print(f"    SDP alt. start {start+1}: mu={mu_final:.6f}")

    # Evaluate Bound 44 on the best SDP result
    if best_W_sdp is not None:
        mu_sdp, b44_sdp, gap_sdp = compute_bound44_numpy(best_W_sdp)
        if verbose:
            print(f"    SDP best: mu={mu_sdp:.6f}, B44={b44_sdp:.6f}, gap={gap_sdp:+.8f}")

        return {
            'mu': mu_sdp, 'bound44': b44_sdp, 'gap': gap_sdp,
            'W': best_W_sdp, 'n': n,
            'known_graphs': {
                'K_n': {'mu': mu_kn, 'b44': b44_kn, 'gap': gap_kn},
                'S_n': {'mu': mu_star, 'b44': b44_star, 'gap': gap_star},
                'C_n': {'mu': mu_cycle, 'b44': b44_cycle, 'gap': gap_cycle},
                'P_n': {'mu': mu_path, 'b44': b44_path, 'gap': gap_path},
            }
        }

    return None


# =====================================================================
# Section 6: SDP with CVXPY — direct mu maximization with bound44 eval
# =====================================================================

def sdp_maximize_mu_minus_bound(n, solver='SCS', verbose=True):
    """Attempt to directly maximize mu - bound44 using SDP + post-hoc eval.

    Since Bound 44 involves square roots and max-over-edges (non-convex),
    we cannot directly encode it in an SDP. Instead:

    Strategy: Fix a target structure (which edges are "active"), then
    solve an SDP to maximize mu while bound44 is evaluated post-hoc.

    For each possible edge subset of reasonable size, we:
    1. Fix which edges exist (W_ij > 0 for edge (i,j))
    2. Optimize weights using SDP to maximize mu
    3. Evaluate bound44 on the result

    For small n this is tractable.

    Args:
        n: number of vertices
        solver: SDP solver
        verbose: print progress

    Returns:
        dict with results
    """
    if verbose:
        print(f"\n  Direct mu maximization for n={n}...")

    # For n <= 6, we can try all possible graph structures (2^(n*(n-1)/2))
    # But that's too many. Instead, focus on key topologies.

    topologies = {
        'complete': [(i, j) for i in range(n) for j in range(i+1, n)],
        'path': [(i, i+1) for i in range(n-1)],
        'cycle': [(i, (i+1) % n) for i in range(n)],
        'star': [(0, i) for i in range(1, n)],
    }

    # Add a few random topologies
    np.random.seed(42)
    for t in range(5):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if np.random.rand() > 0.5:
                    edges.append((i, j))
        if len(edges) >= n - 1:  # at least spanning
            topologies[f'random_{t}'] = edges

    best_overall = None

    for topo_name, edge_list in topologies.items():
        if not edge_list:
            continue

        m = len(edge_list)

        # Optimize weights: maximize mu = lambda_max(L(w))
        # Using alternating: fix eigenvector, optimize weights (LP)

        best_mu_topo = 0.0
        best_weights_topo = None

        for start in range(3):
            np.random.seed(start * 13 + 99)
            weights = np.random.rand(m) * 0.5 + 0.25

            for alt in range(30):
                # Build W
                W = np.zeros((n, n))
                for k, (i, j) in enumerate(edge_list):
                    W[i, j] = weights[k]
                    W[j, i] = weights[k]

                # Eigen decomposition
                d = W.sum(axis=1)
                L = np.diag(d) - W
                eigvals, eigvecs = np.linalg.eigh(L)
                mu = eigvals[-1]
                v = eigvecs[:, -1]
                V = np.outer(v, v)

                # Gradient of trace(L @ V) w.r.t. each weight
                grad = np.zeros(m)
                for k, (i, j) in enumerate(edge_list):
                    # d(trace(LV))/dw_k = V[i,i] + V[j,j] - 2*V[i,j]
                    # (from D contribution + W contribution)
                    grad[k] = V[i, i] + V[j, j] - 2 * V[i, j]

                # Gradient ascent on weights
                weights = weights + 0.3 * grad
                weights = np.clip(weights, 0.01, 1.0)

            # Final eval
            W = np.zeros((n, n))
            for k, (i, j) in enumerate(edge_list):
                W[i, j] = weights[k]
                W[j, i] = weights[k]

            d = W.sum(axis=1)
            L = np.diag(d) - W
            mu = float(np.linalg.eigvalsh(L)[-1])

            if mu > best_mu_topo:
                best_mu_topo = mu
                best_weights_topo = weights.copy()

        # Evaluate Bound 44
        if best_weights_topo is not None:
            W = np.zeros((n, n))
            for k, (i, j) in enumerate(edge_list):
                W[i, j] = best_weights_topo[k]
                W[j, i] = best_weights_topo[k]

            mu, b44, gap = compute_bound44_numpy(W)

            if verbose:
                print(f"    {topo_name:15s}: mu={mu:.6f}, B44={b44:.6f}, gap={gap:+.8f}")

            if best_overall is None or gap < best_overall['gap']:
                best_overall = {
                    'topology': topo_name, 'mu': mu, 'bound44': b44,
                    'gap': gap, 'W': W, 'weights': best_weights_topo,
                    'edges': edge_list, 'n': n
                }

    if verbose and best_overall:
        print(f"  Best: {best_overall['topology']} gap={best_overall['gap']:+.10f}")

    return best_overall


# =====================================================================
# Section 7: Analytical analysis — paths and cycles
# =====================================================================

def analytical_paths_cycles():
    """Analytical verification that Bound 44 holds on all paths and cycles.

    For P_n (path graph):
        mu = 2 + 2*cos(pi/n) ≈ 4 - pi^2/n^2 as n → infinity
        All internal edges: di = dj = 2, mi = mj = 2 (for internal-internal)
        B44 = 2 + sqrt(2*(1 + 1 + 4 - 4)) = 2 + sqrt(4) = 4
        So gap ≈ pi^2/n^2 > 0 for all finite n.

    For C_n (cycle graph):
        mu = 2 + 2*cos(2*pi/n) = 4*cos^2(pi/n) → 4 as n → infinity
        All edges: di = dj = 2, mi = mj = 2
        B44 = 2 + sqrt(2*(1 + 1 + 4 - 4)) = 4
        gap = 4 - mu = 4 - 4*cos^2(pi/n) = 4*sin^2(pi/n) > 0
    """
    print("\n" + "=" * 70)
    print("ANALYTICAL VERIFICATION: Paths and Cycles")
    print("=" * 70)

    print("\n  Path P_n analysis:")
    print("  " + "-" * 50)
    print("  Internal edges (deg-2 to deg-2):")
    print("    di = dj = 2, mi = mj = 2")
    print("    inner = 2*((2-1)^2 + (2-1)^2 + 2*2 - 2*2) = 2*(1+1+4-4) = 4")
    print("    B44 = 2 + sqrt(4) = 4")
    print()
    print("  mu(P_n) = 2 + 2*cos(pi/n)")
    print("  gap = 4 - mu = 2 - 2*cos(pi/n) = 4*sin^2(pi/(2n))")
    print("  gap > 0 for all finite n. QED for paths.")
    print()

    # Numerical verification
    print("  Numerical verification:")
    for n in [5, 10, 20, 50, 100, 500, 1000]:
        mu_exact = 2 + 2 * np.cos(np.pi / n)
        gap_exact = 4 - mu_exact
        gap_formula = 4 * np.sin(np.pi / (2 * n))**2
        print(f"    P_{n:4d}: mu={mu_exact:.10f}, gap={gap_exact:.10f}, "
              f"formula={gap_formula:.10f}, match={abs(gap_exact - gap_formula) < 1e-12}")

    print()
    print("  Cycle C_n analysis:")
    print("  " + "-" * 50)
    print("  All edges: di = dj = 2, mi = mj = 2")
    print("  B44 = 4 (same as paths)")
    print("  mu(C_n) = 2 + 2*cos(2*pi/n) = 4*cos^2(pi/n)")
    print("  gap = 4 - 4*cos^2(pi/n) = 4*sin^2(pi/n)")
    print("  gap > 0 for all finite n. QED for cycles.")
    print()

    print("  Numerical verification:")
    for n in [5, 10, 20, 50, 100, 500, 1000]:
        mu_exact = 2 + 2 * np.cos(2 * np.pi / n)
        gap_exact = 4 - mu_exact
        gap_formula = 4 * np.sin(np.pi / n)**2
        print(f"    C_{n:4d}: mu={mu_exact:.10f}, gap={gap_exact:.10f}, "
              f"formula={gap_formula:.10f}, match={abs(gap_exact - gap_formula) < 1e-12}")

    # Key insight: for regular graphs, bound 44 simplifies
    print("\n  KEY INSIGHT: For k-regular graphs:")
    print("  " + "-" * 50)
    print("  di = dj = k, mi = mj = k (if also neighbor-regular)")
    print("  inner = 2*((k-1)^2 + (k-1)^2 + k^2 - k^2) = 4*(k-1)^2")
    print("  B44 = 2 + 2*(k-1) = 2k")
    print("  mu(K_n) = n for complete graph (which is (n-1)-regular)")
    print("  B44(K_n) = 2(n-1)")
    print("  gap = 2(n-1) - n = n - 2 > 0 for n >= 3")
    print()
    print("  For 2-regular (cycles): B44 = 4, mu < 4. Always holds.")
    print("  For 3-regular: B44 = 6, mu <= 6. Petersen: mu=5, gap=1.")


# =====================================================================
# Section 8: Comprehensive bound tightness analysis
# =====================================================================

def bound_tightness_analysis(n_range=(5, 9)):
    """For each n, find the graph with smallest Bound 44 gap.

    Uses the free-topology gradient optimization from Section 4.

    Args:
        n_range: (min_n, max_n) range of graph sizes

    Returns:
        dict of results per n
    """
    print("\n" + "=" * 70)
    print("COMPREHENSIVE BOUND 44 TIGHTNESS ANALYSIS")
    print("=" * 70)

    results = {}

    for n in range(n_range[0], n_range[1]):
        print(f"\n  n = {n}")
        print(f"  {'-' * 60}")
        t0 = time.time()

        # Method 1: Known graph families
        print(f"  Known graph families:")
        families = {}

        # Complete
        W_kn = np.ones((n, n)) - np.eye(n)
        mu, b44, gap = compute_bound44_numpy(W_kn)
        families['K_n'] = (mu, b44, gap)
        print(f"    K_{n}: mu={mu:.6f}, B44={b44:.6f}, gap={gap:+.8f}")

        # Path
        W_path = np.zeros((n, n))
        for i in range(n - 1):
            W_path[i, i+1] = 1.0
            W_path[i+1, i] = 1.0
        mu, b44, gap = compute_bound44_numpy(W_path)
        families['P_n'] = (mu, b44, gap)
        print(f"    P_{n}: mu={mu:.6f}, B44={b44:.6f}, gap={gap:+.8f}")

        # Cycle
        W_cycle = np.zeros((n, n))
        for i in range(n):
            W_cycle[i, (i+1)%n] = 1.0
            W_cycle[(i+1)%n, i] = 1.0
        mu, b44, gap = compute_bound44_numpy(W_cycle)
        families['C_n'] = (mu, b44, gap)
        print(f"    C_{n}: mu={mu:.6f}, B44={b44:.6f}, gap={gap:+.8f}")

        # Star
        W_star = np.zeros((n, n))
        for i in range(1, n):
            W_star[0, i] = 1.0
            W_star[i, 0] = 1.0
        mu, b44, gap = compute_bound44_numpy(W_star)
        families['S_n'] = (mu, b44, gap)
        print(f"    S_{n}: mu={mu:.6f}, B44={b44:.6f}, gap={gap:+.8f}")

        # Method 2: Free-topology gradient search
        result_free = sdp_bound44_free_topology(
            n, n_trials=10, n_iters=40, verbose=False
        )
        if result_free:
            print(f"    Free-topo: mu={result_free['mu']:.6f}, "
                  f"B44={result_free['bound44']:.6f}, gap={result_free['gap']:+.8f}")

        # Method 3: Weight optimization on path (usually tightest)
        G_path = nx.path_graph(n)
        result_path_opt = optimize_weights_fixed_topology(G_path, verbose=False)
        print(f"    Path-weighted: mu={result_path_opt['mu']:.6f}, "
              f"B44={result_path_opt['bound44']:.6f}, gap={result_path_opt['gap']:+.8f}")

        # Find minimum gap across all methods
        all_gaps = {}
        for name, (_, _, g) in families.items():
            all_gaps[name] = g
        if result_free:
            all_gaps['free_topo'] = result_free['gap']
        all_gaps['path_weighted'] = result_path_opt['gap']

        min_gap_name = min(all_gaps, key=all_gaps.get)
        min_gap_val = all_gaps[min_gap_name]

        elapsed = time.time() - t0
        print(f"    TIGHTEST: {min_gap_name} with gap={min_gap_val:+.10f} ({elapsed:.1f}s)")

        results[n] = {
            'families': families,
            'free_topo': result_free,
            'path_weighted': result_path_opt,
            'min_gap': min_gap_val,
            'min_gap_source': min_gap_name,
        }

    return results


# =====================================================================
# Section 9: Weighted graph counterexample search (targeted)
# =====================================================================

def targeted_weighted_search(n, max_time=60, verbose=True):
    """Targeted search for weighted graph counterexamples to Bound 44.

    Strategy: start from near-miss discrete graphs and optimize weights.
    Also try "anti-regular" structures that might break the bound.

    Args:
        n: number of vertices
        max_time: time budget in seconds
        verbose: print progress

    Returns:
        dict with best result
    """
    from scipy.optimize import differential_evolution

    if verbose:
        print(f"\n  Targeted weighted search for n={n}...")

    best_gap = float('inf')
    best_result = None
    t0 = time.time()

    # Strategy 1: Optimize weights on path (known near-miss)
    if verbose:
        print(f"    Strategy 1: Weighted path P_{n}")
    G = nx.path_graph(n)
    result = optimize_weights_fixed_topology(G, verbose=False)
    if result['gap'] < best_gap:
        best_gap = result['gap']
        best_result = {**result, 'strategy': f'weighted_path_{n}'}
    if verbose:
        print(f"      gap = {result['gap']:+.10f}")

    # Strategy 2: Path with one extra edge (near-cycle)
    if n >= 4 and time.time() - t0 < max_time:
        if verbose:
            print(f"    Strategy 2: Path + extra edge")
        G = nx.path_graph(n)
        # Add edge connecting vertices at distance 3
        for skip in range(3, min(n, 8)):
            if time.time() - t0 > max_time:
                break
            G2 = G.copy()
            mid = n // 2
            if mid + skip < n:
                G2.add_edge(mid, mid + skip)
                result = optimize_weights_fixed_topology(G2, verbose=False)
                if result['gap'] < best_gap:
                    best_gap = result['gap']
                    best_result = {**result, 'strategy': f'path+edge(skip={skip})'}
                if verbose:
                    print(f"      skip={skip}: gap = {result['gap']:+.10f}")

    # Strategy 3: Caterpillar variants
    if n >= 6 and time.time() - t0 < max_time:
        if verbose:
            print(f"    Strategy 3: Caterpillar variants")
        spine = max(3, n // 2)
        G = nx.path_graph(spine)
        leaves_added = 0
        node_id = spine
        for v in range(1, spine - 1):
            if node_id >= n:
                break
            G.add_edge(v, node_id)
            node_id += 1
            leaves_added += 1

        if G.number_of_nodes() >= 4:
            result = optimize_weights_fixed_topology(G, verbose=False)
            if result['gap'] < best_gap:
                best_gap = result['gap']
                best_result = {**result, 'strategy': 'caterpillar'}
            if verbose:
                print(f"      gap = {result['gap']:+.10f}")

    # Strategy 4: Bipartite graphs (known attractors in search)
    if n >= 4 and time.time() - t0 < max_time:
        if verbose:
            print(f"    Strategy 4: Complete bipartite")
        n1 = n // 2
        n2 = n - n1
        G = nx.complete_bipartite_graph(n1, n2)
        if G.number_of_nodes() == n:
            result = optimize_weights_fixed_topology(G, verbose=False)
            if result['gap'] < best_gap:
                best_gap = result['gap']
                best_result = {**result, 'strategy': f'K_{n1},{n2}'}
            if verbose:
                print(f"      K_{n1},{n2}: gap = {result['gap']:+.10f}")

    # Strategy 5: Full free-topology gradient descent
    if time.time() - t0 < max_time:
        remaining = max_time - (time.time() - t0)
        n_trials = max(3, int(remaining / 5))
        if verbose:
            print(f"    Strategy 5: Free topology ({n_trials} trials)")
        result = sdp_bound44_free_topology(n, n_trials=n_trials, n_iters=50, verbose=False)
        if result and result['gap'] < best_gap:
            best_gap = result['gap']
            best_result = {**result, 'strategy': 'free_topology'}
        if verbose and result:
            print(f"      gap = {result['gap']:+.10f}")

    elapsed = time.time() - t0
    if verbose:
        print(f"    BEST: gap={best_gap:+.10f} ({elapsed:.1f}s)")
        if best_result:
            print(f"    Strategy: {best_result.get('strategy', 'unknown')}")

    return best_result


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 70)
    print("SDP EXISTENCE ANALYSIS FOR BOUND 44")
    print("Cross-domain approach: continuous relaxation of discrete graph search")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────
    # Part 1: Analytical results
    # ─────────────────────────────────────────────────────────────────
    analytical_paths_cycles()

    # ─────────────────────────────────────────────────────────────────
    # Part 2: Comprehensive tightness analysis for n=5..8
    # ─────────────────────────────────────────────────────────────────
    tightness_results = bound_tightness_analysis(n_range=(5, 9))

    # ─────────────────────────────────────────────────────────────────
    # Part 3: CVXPY SDP-based analysis
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CVXPY SDP ANALYSIS")
    print("=" * 70)

    sdp_results = {}
    for n in [5, 6, 7, 8]:
        sdp_results[n] = cvxpy_sdp_check(n, verbose=True)

    # ─────────────────────────────────────────────────────────────────
    # Part 4: Targeted weighted search
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TARGETED WEIGHTED COUNTEREXAMPLE SEARCH")
    print("=" * 70)

    targeted_results = {}
    for n in [5, 6, 7, 8]:
        targeted_results[n] = targeted_weighted_search(n, max_time=30, verbose=True)

    # ─────────────────────────────────────────────────────────────────
    # Part 5: Summary
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print("\n  Minimum gaps found across all methods:")
    print(f"  {'n':>3} {'Min Gap':>14} {'Source':>20} {'Implication':>30}")
    print(f"  {'-'*70}")

    all_violated = False
    for n in [5, 6, 7, 8]:
        gaps = []

        if n in tightness_results:
            gaps.append((tightness_results[n]['min_gap'],
                        tightness_results[n]['min_gap_source']))

        if n in targeted_results and targeted_results[n]:
            gaps.append((targeted_results[n]['gap'],
                        targeted_results[n].get('strategy', 'targeted')))

        if n in sdp_results and sdp_results[n]:
            gaps.append((sdp_results[n]['gap'], 'SDP'))

        if gaps:
            min_gap, min_src = min(gaps, key=lambda x: x[0])
            violated = min_gap < -1e-6
            if violated:
                all_violated = True
            impl = "*** VIOLATED ***" if violated else "Bound holds"
            print(f"  {n:3d} {min_gap:+14.10f} {min_src:>20} {impl:>30}")

    print()
    if all_violated:
        print("  CONCLUSION: Counterexample(s) FOUND in the weighted relaxation!")
        print("  The bound can be violated on weighted graphs.")
        print("  This suggests discrete counterexamples MAY exist.")
    else:
        print("  CONCLUSION: No violations found in the weighted relaxation.")
        print("  Bound 44 appears to hold even on weighted graphs for n=5..8.")
        print("  This provides evidence (not proof) that the bound is valid.")
        print()
        print("  Theoretical support:")
        print("  - Paths: gap = 4*sin^2(pi/(2n)) > 0 for all finite n")
        print("  - Cycles: gap = 4*sin^2(pi/n) > 0 for all finite n")
        print("  - k-regular: B44 = 2k, mu <= 2k (equality at K_2). Gap >= 0")
        print("  - Complete K_n: gap = n-2 (grows with n)")
        print("  - The relaxation to weighted graphs did not create new violations")


if __name__ == '__main__':
    main()
