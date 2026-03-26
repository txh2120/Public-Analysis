#!/usr/bin/env python3
"""RL Graph Conjecture CEM — Ghebleh(2024) settings v2"""

# === Code Cell 0 ===
"""Cell 2: Imports and Configuration"""

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# ── Hyperparameters ──────────────────────────────────────────────
N = 20                      # Number of vertices (overridden per run)
DECISIONS = N * (N - 1) // 2
OBSERVATION_SPACE = 2 * DECISIONS

N_SESSIONS = 200            # Sessions per generation
PERCENTILE = 90             # Elite threshold (top 10%) — Ghebleh(2024)
SUPER_PERCENTILE = 97.5     # Super-elite threshold (top 2.5%) — Ghebleh(2024)
LEARNING_RATE = 0.003       # Adam learning rate — Ghebleh(2024)
MAX_ITERATIONS = 5000       # Default max iterations — Ghebleh(2024)

DISCONNECTED_PENALTY = -1_000_000  # Reward for disconnected graphs

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device (CPU — GPU unnecessary for this small MLP)
DEVICE = torch.device('cpu')

print(f"PyTorch version: {torch.__version__}")
print(f"NetworkX version: {nx.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Device: {DEVICE}")
print(f"Default graph size N={N}, decisions={DECISIONS}, state_dim={OBSERVATION_SPACE}")

# -- Time Budget Manager -----------------------------------------------
from datetime import datetime

TOTAL_TIME_BUDGET = 11.5 * 3600  # 11.5 hours (30min buffer for 12h Kaggle limit)
NOTEBOOK_START_TIME = time.time()

class TimeManager:
    """Global time budget manager for Kaggle 12-hour sessions.

    Allocates time adaptively:
      - Benchmark runs: ~1.5h total
      - Primary attack (Bound 1, n=20): ~6h
      - Secondary attacks: ~3h
      - Buffer: ~1h (plus any saved time from early completions)
    """

    ALLOCATIONS = {
        'benchmark': 1.5 * 3600,    # 1.5h for benchmark runs
        'primary':   6.0 * 3600,    # 6h for primary attack
        'secondary': 3.0 * 3600,    # 3h for secondary attacks
    }

    def __init__(self, start_time, total_budget):
        self.start_time = start_time
        self.total_budget = total_budget
        self.phase_start_times = {}
        self.phase_elapsed = {}

    def elapsed(self):
        """Total elapsed time since notebook start (seconds)."""
        return time.time() - self.start_time

    def remaining(self):
        """Remaining time in total budget (seconds)."""
        return max(0, self.total_budget - self.elapsed())

    def can_start(self, phase, min_useful_time=300):
        """Check if there's enough time to start a phase.

        Args:
            phase: 'benchmark', 'primary', or 'secondary'
            min_useful_time: minimum seconds needed for a useful run (default 5min)

        Returns:
            (can_start: bool, available_seconds: float)
        """
        remaining = self.remaining()
        if remaining < min_useful_time:
            return False, remaining
        return True, remaining

    def get_time_limit(self, phase, run_label=''):
        """Get time limit for a specific run within a phase.

        Returns seconds available, respecting both phase allocation
        and total remaining budget.
        """
        allocation = self.ALLOCATIONS.get(phase, 3600)
        phase_used = self.phase_elapsed.get(phase, 0)
        phase_remaining = max(0, allocation - phase_used)
        total_remaining = self.remaining()

        # Use the smaller of phase allocation and total remaining
        limit = min(phase_remaining, total_remaining)
        return max(0, limit)

    def start_phase(self, phase):
        """Mark the start of a phase."""
        self.phase_start_times[phase] = time.time()

    def end_phase(self, phase):
        """Mark the end of a phase and record elapsed time."""
        if phase in self.phase_start_times:
            elapsed = time.time() - self.phase_start_times[phase]
            self.phase_elapsed[phase] = self.phase_elapsed.get(phase, 0) + elapsed

    def status(self):
        """Print current time budget status."""
        elapsed = self.elapsed()
        remaining = self.remaining()
        print(f"\n{'---'*17}")
        print(f"TIME BUDGET STATUS")
        print(f"  Elapsed:   {elapsed/3600:.2f}h ({elapsed:.0f}s)")
        print(f"  Remaining: {remaining/3600:.2f}h ({remaining:.0f}s)")
        print(f"  Budget:    {self.total_budget/3600:.1f}h")
        for phase, alloc in self.ALLOCATIONS.items():
            used = self.phase_elapsed.get(phase, 0)
            print(f"  {phase:12s}: {used/3600:.2f}h used / {alloc/3600:.1f}h allocated")
        print(f"{'---'*17}\n")

# Create global time manager
timer = TimeManager(NOTEBOOK_START_TIME, TOTAL_TIME_BUDGET)
print(f"TimeManager initialized: {TOTAL_TIME_BUDGET/3600:.1f}h budget")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# === Code Cell 1 ===
"""Cell 3: CEM Model — Configurable MLP with 2-output logits (Ghebleh 2024)"""

class CEMModel(nn.Module):
    """
    Ghebleh(2024) Deep Cross-Entropy Method neural network.

    Architecture: Configurable hidden layers with Dropout after each ReLU.
    Default: Linear(input_dim, 128) -> ReLU -> Dropout -> Linear(128, 64) -> ReLU
             -> Dropout -> Linear(64, 4) -> ReLU -> Dropout -> Linear(4, 2)

    Input: state vector of dimension 2 * N*(N-1)/2
           (adjacency upper triangle + one-hot position marker)
    Output: 2 raw logits (for CrossEntropyLoss) — NO Sigmoid/Softmax
    """
    def __init__(self, input_dim, hidden_layers=None, dropout=0.2):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [128, 64, 4]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 2))  # 2 raw logits

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def create_model(n_vertices, hidden_layers=None):
    """Create a new CEM model for graphs with n_vertices nodes."""
    decisions = n_vertices * (n_vertices - 1) // 2
    input_dim = 2 * decisions
    model = CEMModel(input_dim, hidden_layers=hidden_layers).to(DEVICE)
    return model


# Quick architecture check — 2-output shape assertion
_test_model = create_model(10)
_test_input = torch.zeros(1, 2 * 45)  # n=10 -> 45 decisions -> dim=90
_test_output = _test_model(_test_input)
assert _test_output.shape == (1, 2), f"Expected (1, 2), got {_test_output.shape}"
print(f"Model test (n=10): input_dim=90, output_shape={_test_output.shape} (2 logits)")
n_params = sum(p.numel() for p in _test_model.parameters())
print(f"Default [128,64,4] architecture — Total parameters: {n_params}")

# Test [72, 12] architecture
_test_model_72 = create_model(10, hidden_layers=[72, 12])
_test_output_72 = _test_model_72(_test_input)
assert _test_output_72.shape == (1, 2), f"Expected (1, 2), got {_test_output_72.shape}"
n_params_72 = sum(p.numel() for p in _test_model_72.parameters())
print(f"Alt [72,12] architecture — Total parameters: {n_params_72}")
del _test_model, _test_model_72, _test_input, _test_output, _test_output_72

# === Code Cell 2 ===
"""Cell 4: Graph Utility Functions"""

def edge_index_to_pair(idx, n):
    """Convert flat upper-triangle index to (i, j) vertex pair.

    Edge ordering: (0,1), (0,2), ..., (0,n-1), (1,2), ..., (n-2,n-1)
    """
    # Row i where the edge falls
    i = 0
    cumulative = 0
    for row in range(n - 1):
        row_edges = n - 1 - row
        if cumulative + row_edges > idx:
            i = row
            j = i + 1 + (idx - cumulative)
            return (i, j)
        cumulative += row_edges
    raise ValueError(f"Invalid edge index {idx} for n={n}")


def build_graph(state, n):
    """Build a NetworkX graph from the binary adjacency vector.

    Args:
        state: binary vector of length >= n*(n-1)/2 (first half of full state)
        n: number of vertices

    Returns:
        NetworkX Graph
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))
    decisions = n * (n - 1) // 2
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if state[count] == 1:
                G.add_edge(i, j)
            count += 1
    return G


def compute_laplacian_spectral_radius(G):
    """Compute the largest eigenvalue of the Laplacian matrix L = D - A.

    Args:
        G: NetworkX graph (must have at least 1 edge)

    Returns:
        mu: largest Laplacian eigenvalue (float)
    """
    if G.number_of_edges() == 0:
        return 0.0
    L = nx.laplacian_matrix(G).toarray().astype(np.float64)
    eigenvalues = np.linalg.eigvalsh(L)
    return float(eigenvalues[-1])


def compute_dv_mv(G, n):
    """Compute degree (d_v) and average neighbor degree (m_v) for all vertices.

    Args:
        G: NetworkX graph
        n: number of vertices

    Returns:
        dv: array of degrees, shape (n,)
        mv: array of average neighbor degrees, shape (n,)
             (m_v = 0 for isolated vertices)
    """
    dv = np.zeros(n, dtype=np.float64)
    mv = np.zeros(n, dtype=np.float64)

    for v in range(n):
        d = G.degree(v)
        dv[v] = d
        if d > 0:
            neighbor_degree_sum = sum(G.degree(u) for u in G.neighbors(v))
            mv[v] = neighbor_degree_sum / d

    return dv, mv


# Precompute edge pairs lookup for efficiency
def precompute_edge_pairs(n):
    """Precompute the mapping from flat index to (i, j) pairs."""
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    return pairs


# ── Numpy bypass functions ───────────────────────────────────────
# These replace NetworkX in the CEM hot loop for ~5-10x speedup.
# verify_counterexample() still uses NetworkX for independent cross-verification.

def precompute_triu_indices(n):
    """Precompute upper triangle indices for vectorized adjacency matrix construction."""
    rows, cols = [], []
    for i in range(n):
        for j in range(i + 1, n):
            rows.append(i)
            cols.append(j)
    return np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)


def compute_graph_properties_numpy(state, n, triu_rows, triu_cols):
    """All-numpy graph property computation (no NetworkX).

    Replaces build_graph + compute_laplacian_spectral_radius + compute_dv_mv
    in a single function for maximum performance.

    Args:
        state: binary adjacency vector, length >= n*(n-1)/2
        n: number of vertices
        triu_rows: precomputed row indices from precompute_triu_indices
        triu_cols: precomputed col indices from precompute_triu_indices

    Returns:
        tuple (mu, dv, mv, A) or None if graph is disconnected
        mu: largest Laplacian eigenvalue (float)
        dv: degree vector, shape (n,)
        mv: average neighbor degree vector, shape (n,)
        A: adjacency matrix, shape (n, n)
    """
    decisions = n * (n - 1) // 2

    # Vectorized adjacency matrix construction
    A = np.zeros((n, n), dtype=np.float64)
    mask = state[:decisions] == 1
    A[triu_rows[mask], triu_cols[mask]] = 1.0
    A[triu_cols[mask], triu_rows[mask]] = 1.0

    # Degree vector
    dv = A.sum(axis=1)

    # Quick disconnection check: isolated vertices
    if np.any(dv == 0):
        return None

    # DFS connectivity check (numpy-assisted)
    visited = np.zeros(n, dtype=bool)
    stack = [0]
    visited[0] = True
    count = 1
    while stack:
        v = stack.pop()
        for u in np.nonzero(A[v])[0]:
            if not visited[u]:
                visited[u] = True
                stack.append(int(u))
                count += 1
    if count < n:
        return None

    # Laplacian spectral radius via eigvalsh
    L = np.diag(dv) - A
    mu = float(np.linalg.eigvalsh(L)[-1])

    # Average neighbor degree via matrix multiply: (A @ dv)[v] / dv[v]
    mv = np.zeros(n, dtype=np.float64)
    nonzero = dv > 0
    mv[nonzero] = (A @ dv)[nonzero] / dv[nonzero]

    return mu, dv, mv, A


print("Graph utility functions defined (NetworkX + numpy bypass).")

# === Code Cell 3 ===
"""Cell 5: Unit Tests — Laplacian Spectral Radius"""

def run_unit_tests():
    """Test compute_laplacian_spectral_radius on known graphs.

    Test cases (regular):
    1. Complete graph K_5: mu = 5.0
    2. Star graph S_5 (1 center + 4 leaves): mu = 5.0
    3. Cycle graph C_6: mu = 4.0

    Test cases (non-regular):
    4. Path graph P_6: mu = 2 + sqrt(3) ≈ 3.732051
    5. Wheel graph W_6: mu = 6.0
    6. Barbell graph B(4,1): mu via eigvalsh reference
    """
    results = []
    tol = 1e-6

    # Test 1: Complete graph K_5
    G_k5 = nx.complete_graph(5)
    mu_k5 = compute_laplacian_spectral_radius(G_k5)
    expected_k5 = 5.0
    pass_k5 = abs(mu_k5 - expected_k5) < tol
    results.append(('K_5', expected_k5, mu_k5, pass_k5))

    # Test 2: Star graph S_5 (1 center + 4 leaves = 5 nodes)
    G_s5 = nx.star_graph(4)
    mu_s5 = compute_laplacian_spectral_radius(G_s5)
    expected_s5 = 5.0
    pass_s5 = abs(mu_s5 - expected_s5) < tol
    results.append(('S_5', expected_s5, mu_s5, pass_s5))

    # Test 3: Cycle graph C_6
    G_c6 = nx.cycle_graph(6)
    mu_c6 = compute_laplacian_spectral_radius(G_c6)
    expected_c6 = 4.0
    pass_c6 = abs(mu_c6 - expected_c6) < tol
    results.append(('C_6', expected_c6, mu_c6, pass_c6))

    # Test 4: Path graph P_6 (non-regular)
    # Laplacian eigenvalues of P_n: 2 - 2*cos(pi*k/n) for k=0..n-1
    # P_6 largest: 2 - 2*cos(5*pi/6) = 2 + sqrt(3)
    G_p6 = nx.path_graph(6)
    mu_p6 = compute_laplacian_spectral_radius(G_p6)
    expected_p6 = 2.0 + np.sqrt(3.0)
    pass_p6 = abs(mu_p6 - expected_p6) < tol
    results.append(('P_6 (non-reg)', expected_p6, mu_p6, pass_p6))

    # Test 5: Wheel graph W_6 (non-regular, 6 vertices)
    # W_n = K_1 + C_{n-1}. For W_6: center degree=5, rim degree=3.
    # Largest Laplacian eigenvalue of W_6 = 6.0
    G_w6 = nx.wheel_graph(6)
    mu_w6 = compute_laplacian_spectral_radius(G_w6)
    expected_w6 = 6.0
    pass_w6 = abs(mu_w6 - expected_w6) < tol
    results.append(('W_6 (non-reg)', expected_w6, mu_w6, pass_w6))

    # Test 6: Barbell graph B(4,1) (non-regular, 9 vertices)
    # Two K_4 cliques connected by a single bridge vertex
    G_barbell = nx.barbell_graph(4, 1)
    mu_barbell = compute_laplacian_spectral_radius(G_barbell)
    L_barbell = nx.laplacian_matrix(G_barbell).toarray().astype(np.float64)
    expected_barbell = float(np.linalg.eigvalsh(L_barbell)[-1])
    pass_barbell = abs(mu_barbell - expected_barbell) < tol
    results.append(('Barbell(4,1) (non-reg)', expected_barbell, mu_barbell, pass_barbell))

    # Print results
    print("=" * 65)
    print("UNIT TESTS: Laplacian Spectral Radius")
    print("=" * 65)
    all_pass = True
    for name, expected, actual, passed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {name}: expected={expected:.6f}, got={actual:.6f} -> [{status}]")
    print("=" * 65)

    if all_pass:
        print("All unit tests PASSED.")
    else:
        print("WARNING: Some tests FAILED!")

    # Build-graph roundtrip test
    print("\nBuild-graph roundtrip test:")
    n_test = 5
    decisions_test = n_test * (n_test - 1) // 2
    state_k5 = np.ones(decisions_test, dtype=np.int32)
    G_rebuilt = build_graph(state_k5, n_test)
    assert G_rebuilt.number_of_edges() == 10, f"Expected 10 edges, got {G_rebuilt.number_of_edges()}"
    mu_rebuilt = compute_laplacian_spectral_radius(G_rebuilt)
    assert abs(mu_rebuilt - 5.0) < tol, f"Expected mu=5.0, got {mu_rebuilt}"
    print(f"  K_5 from state vector: edges={G_rebuilt.number_of_edges()}, mu={mu_rebuilt:.6f} -> [PASS]")

    dv, mv = compute_dv_mv(G_rebuilt, n_test)
    assert np.all(dv == 4), f"K_5 degrees should all be 4, got {dv}"
    assert np.all(mv == 4), f"K_5 avg neighbor degree should all be 4, got {mv}"
    print(f"  K_5 dv={dv}, mv={mv} -> [PASS]")

    dv_s, mv_s = compute_dv_mv(G_s5, 5)
    print(f"  S_5 dv={dv_s}, mv={mv_s}")
    assert dv_s[0] == 4, f"Star center degree should be 4, got {dv_s[0]}"
    assert np.all(dv_s[1:] == 1), f"Star leaf degrees should be 1"
    print(f"  S_5 dv/mv check -> [PASS]")

    # ── Numpy bypass cross-verification ──────────────────────────
    print("\n" + "=" * 65)
    print("CROSS-VERIFICATION: NetworkX vs Numpy bypass")
    print("=" * 65)

    numpy_pass = True

    test_graphs = {
        'K_5': (G_k5, 5),
        'S_5': (G_s5, 5),
        'C_6': (G_c6, 6),
        'P_6': (G_p6, 6),
        'W_6': (G_w6, 6),
        'Barbell(4,1)': (G_barbell, 9),
    }

    for name, (G, n_v) in test_graphs.items():
        decisions = n_v * (n_v - 1) // 2
        state = np.zeros(decisions, dtype=np.int32)
        idx = 0
        for i in range(n_v):
            for j in range(i + 1, n_v):
                if G.has_edge(i, j):
                    state[idx] = 1
                idx += 1

        triu_r, triu_c = precompute_triu_indices(n_v)
        result_np = compute_graph_properties_numpy(state, n_v, triu_r, triu_c)

        mu_nx = compute_laplacian_spectral_radius(G)
        dv_nx, mv_nx = compute_dv_mv(G, n_v)

        if result_np is None:
            print(f"  {name}: numpy returned None (disconnected) -> [FAIL]")
            numpy_pass = False
        else:
            mu_np, dv_np, mv_np, A_np = result_np
            mu_match = abs(mu_np - mu_nx) < 1e-10
            dv_match = np.allclose(dv_np, dv_nx, atol=1e-10)
            mv_match = np.allclose(mv_np, mv_nx, atol=1e-10)
            all_match = mu_match and dv_match and mv_match
            status = "PASS" if all_match else "FAIL"
            if not all_match:
                numpy_pass = False
            print(f"  {name}: mu_diff={abs(mu_np-mu_nx):.2e}, "
                  f"dv_match={dv_match}, mv_match={mv_match} -> [{status}]")

    # Random graph stress test: n=15, 50 random graphs
    print("\n  Random graph stress test (n=15, 50 graphs):")
    n_stress = 15
    n_tests = 50
    triu_r15, triu_c15 = precompute_triu_indices(n_stress)
    decisions_15 = n_stress * (n_stress - 1) // 2
    stress_pass = 0
    stress_fail = 0

    rng = np.random.RandomState(42)
    for t in range(n_tests):
        state_rand = rng.randint(0, 2, size=decisions_15).astype(np.int32)

        G_rand = build_graph(state_rand, n_stress)
        if not nx.is_connected(G_rand):
            result_np = compute_graph_properties_numpy(state_rand, n_stress, triu_r15, triu_c15)
            if result_np is None:
                stress_pass += 1
            else:
                stress_fail += 1
            continue

        mu_nx = compute_laplacian_spectral_radius(G_rand)
        dv_nx, mv_nx = compute_dv_mv(G_rand, n_stress)

        result_np = compute_graph_properties_numpy(state_rand, n_stress, triu_r15, triu_c15)
        if result_np is None:
            stress_fail += 1
            continue

        mu_np, dv_np, mv_np, A_np = result_np
        if (abs(mu_np - mu_nx) < 1e-10 and
            np.allclose(dv_np, dv_nx, atol=1e-10) and
            np.allclose(mv_np, mv_nx, atol=1e-10)):
            stress_pass += 1
        else:
            stress_fail += 1
            print(f"    Test {t}: MISMATCH mu_diff={abs(mu_np-mu_nx):.2e}")

    stress_ok = stress_fail == 0
    if not stress_ok:
        numpy_pass = False
    print(f"    Results: {stress_pass}/{n_tests} passed, {stress_fail} failed -> "
          f"[{'PASS' if stress_ok else 'FAIL'}]")

    print("=" * 65)
    if numpy_pass:
        print("All numpy cross-verification tests PASSED.")
    else:
        print("WARNING: Some numpy tests FAILED!")

    return all_pass and numpy_pass

unit_tests_passed = run_unit_tests()

# === Code Cell 4 ===
"""Cell 5.5: Performance Comparison — NetworkX vs Numpy Bypass"""

def benchmark_performance(n=15, n_trials=200):
    """Time NetworkX vs numpy reward computation.

    Args:
        n: number of vertices
        n_trials: number of reward computations to time
    """
    decisions = n * (n - 1) // 2
    triu_r, triu_c = precompute_triu_indices(n)

    # Generate random connected graph states
    rng = np.random.RandomState(123)
    states = []
    while len(states) < n_trials:
        s = rng.randint(0, 2, size=decisions).astype(np.int32)
        # Only use connected graphs for fair comparison
        G = build_graph(s, n)
        if nx.is_connected(G):
            states.append(s)

    print(f"Performance benchmark: n={n}, {n_trials} connected graphs")
    print("-" * 55)

    # ── NetworkX timing ──
    t0 = time.perf_counter()
    nx_results = []
    for s in states:
        G = build_graph(s, n)
        mu = compute_laplacian_spectral_radius(G)
        dv, mv = compute_dv_mv(G, n)
        valid = dv > 0
        bound_vals = 2.0 * mv[valid]**2 / dv[valid]
        nx_results.append(mu - np.max(bound_vals))
    t_nx = time.perf_counter() - t0

    # ── Numpy timing ──
    t0 = time.perf_counter()
    np_results = []
    for s in states:
        result = compute_graph_properties_numpy(s, n, triu_r, triu_c)
        if result is None:
            np_results.append(DISCONNECTED_PENALTY)
        else:
            mu, dv, mv, A = result
            valid = dv > 0
            bound_vals = 2.0 * mv[valid]**2 / dv[valid]
            np_results.append(mu - np.max(bound_vals))
    t_np = time.perf_counter() - t0

    speedup = t_nx / t_np if t_np > 0 else float('inf')

    print(f"  NetworkX: {t_nx:.3f}s ({t_nx/n_trials*1000:.2f} ms/graph)")
    print(f"  Numpy:    {t_np:.3f}s ({t_np/n_trials*1000:.2f} ms/graph)")
    print(f"  Speedup:  {speedup:.1f}x")

    # Verify results match
    max_diff = max(abs(a - b) for a, b in zip(nx_results, np_results))
    print(f"  Max result difference: {max_diff:.2e}")
    print(f"  Results match: {'YES' if max_diff < 1e-10 else 'NO'}")
    print("-" * 55)

    return speedup

perf_speedup = benchmark_performance(n=15, n_trials=200)

# === Code Cell 5 ===
"""Cell 6: Reward Functions — Laplacian Spectral Radius Bounds"""

from functools import partial

# ── NetworkX versions (retained for verify_counterexample) ───────

def calc_score_bound2_nx(state, n):
    """Benchmark bound (DISPROVED): mu <= max_v (2 * m_v^2 / d_v) [NetworkX version]"""
    G = build_graph(state, n)
    if not nx.is_connected(G):
        return DISCONNECTED_PENALTY
    mu = compute_laplacian_spectral_radius(G)
    dv, mv = compute_dv_mv(G, n)
    bound_values = []
    for v in range(n):
        if dv[v] > 0:
            bound_values.append(2.0 * mv[v]**2 / dv[v])
    if len(bound_values) == 0:
        return DISCONNECTED_PENALTY
    return mu - max(bound_values)


def calc_score_bound3_nx(state, n):
    """Bound 3 (DISPROVED, RL-findable): mu <= max_v (m_v^2/d_v + m_v) [NetworkX version]"""
    G = build_graph(state, n)
    if not nx.is_connected(G):
        return DISCONNECTED_PENALTY
    mu = compute_laplacian_spectral_radius(G)
    dv, mv = compute_dv_mv(G, n)
    bound_values = []
    for v in range(n):
        if dv[v] > 0:
            bound_values.append(mv[v]**2 / dv[v] + mv[v])
    if not bound_values:
        return DISCONNECTED_PENALTY
    return mu - max(bound_values)


def calc_score_bound1_nx(state, n):
    """Attack bound (OPEN): mu <= max_v sqrt(4 * d_v^3 / m_v) [NetworkX version]"""
    G = build_graph(state, n)
    if not nx.is_connected(G):
        return DISCONNECTED_PENALTY
    mu = compute_laplacian_spectral_radius(G)
    dv, mv = compute_dv_mv(G, n)
    bound_values = []
    for v in range(n):
        if dv[v] > 0 and mv[v] > 0:
            bound_values.append(np.sqrt(4.0 * dv[v]**3 / mv[v]))
    if len(bound_values) == 0:
        return DISCONNECTED_PENALTY
    return mu - max(bound_values)


def calc_score_bound4_nx(state, n):
    """Secondary attack (OPEN): mu <= max_v (2 * d_v^2 / m_v) [NetworkX version]"""
    G = build_graph(state, n)
    if not nx.is_connected(G):
        return DISCONNECTED_PENALTY
    mu = compute_laplacian_spectral_radius(G)
    dv, mv = compute_dv_mv(G, n)
    bound_values = []
    for v in range(n):
        if dv[v] > 0 and mv[v] > 0:
            bound_values.append(2.0 * dv[v]**2 / mv[v])
    if len(bound_values) == 0:
        return DISCONNECTED_PENALTY
    return mu - max(bound_values)


def calc_score_bound9_nx(state, n):
    """Secondary attack (OPEN): mu <= max_v (m_v + 3*d_v) / 2 [NetworkX version]"""
    G = build_graph(state, n)
    if not nx.is_connected(G):
        return DISCONNECTED_PENALTY
    mu = compute_laplacian_spectral_radius(G)
    dv, mv = compute_dv_mv(G, n)
    bound_values = []
    for v in range(n):
        if dv[v] > 0:
            bound_values.append((mv[v] + 3.0 * dv[v]) / 2.0)
    if len(bound_values) == 0:
        return DISCONNECTED_PENALTY
    return mu - max(bound_values)


def calc_score_bound33_nx(state, n):
    """Secondary attack (OPEN, edge-max): mu <= max_{vi~vj} (2*(d_i+d_j)-(m_i+m_j)) [NetworkX]"""
    G = build_graph(state, n)
    if not nx.is_connected(G):
        return DISCONNECTED_PENALTY
    mu = compute_laplacian_spectral_radius(G)
    dv, mv = compute_dv_mv(G, n)
    bound_values = []
    for (i, j) in G.edges():
        val = 2.0 * (dv[i] + dv[j]) - (mv[i] + mv[j])
        bound_values.append(val)
    if len(bound_values) == 0:
        return DISCONNECTED_PENALTY
    return mu - max(bound_values)


# ── Numpy bypass versions (used in CEM hot loop) ────────────────

def calc_score_bound2_numpy(state, n, triu_rows, triu_cols):
    """Benchmark bound (DISPROVED): mu <= max_v (2 * m_v^2 / d_v) [numpy version]"""
    result = compute_graph_properties_numpy(state, n, triu_rows, triu_cols)
    if result is None:
        return DISCONNECTED_PENALTY
    mu, dv, mv, A = result
    valid = dv > 0
    if not np.any(valid):
        return DISCONNECTED_PENALTY
    bound_values = np.where(valid, 2.0 * mv**2 / np.maximum(dv, 1e-300), 0.0)
    return mu - np.max(bound_values[valid])


def calc_score_bound3_numpy(state, n, triu_rows, triu_cols):
    """Bound 3 (DISPROVED, RL-findable): mu <= max_v (m_v^2/d_v + m_v) [numpy version]"""
    result = compute_graph_properties_numpy(state, n, triu_rows, triu_cols)
    if result is None:
        return DISCONNECTED_PENALTY
    mu, dv, mv, A = result
    valid = dv > 0
    if not np.any(valid):
        return DISCONNECTED_PENALTY
    bound_values = np.where(valid, mv**2 / np.maximum(dv, 1e-300) + mv, 0.0)
    return mu - np.max(bound_values[valid])


def calc_score_bound1_numpy(state, n, triu_rows, triu_cols):
    """Attack bound (OPEN): mu <= max_v sqrt(4 * d_v^3 / m_v) [numpy version]"""
    result = compute_graph_properties_numpy(state, n, triu_rows, triu_cols)
    if result is None:
        return DISCONNECTED_PENALTY
    mu, dv, mv, A = result
    valid = (dv > 0) & (mv > 0)
    if not np.any(valid):
        return DISCONNECTED_PENALTY
    bound_values = np.where(valid, np.sqrt(4.0 * dv**3 / np.maximum(mv, 1e-300)), 0.0)
    return mu - np.max(bound_values[valid])


def calc_score_bound4_numpy(state, n, triu_rows, triu_cols):
    """Secondary attack (OPEN): mu <= max_v (2 * d_v^2 / m_v) [numpy version]"""
    result = compute_graph_properties_numpy(state, n, triu_rows, triu_cols)
    if result is None:
        return DISCONNECTED_PENALTY
    mu, dv, mv, A = result
    valid = (dv > 0) & (mv > 0)
    if not np.any(valid):
        return DISCONNECTED_PENALTY
    bound_values = np.where(valid, 2.0 * dv**2 / np.maximum(mv, 1e-300), 0.0)
    return mu - np.max(bound_values[valid])


def calc_score_bound9_numpy(state, n, triu_rows, triu_cols):
    """Secondary attack (OPEN): mu <= max_v (m_v + 3*d_v) / 2 [numpy version]"""
    result = compute_graph_properties_numpy(state, n, triu_rows, triu_cols)
    if result is None:
        return DISCONNECTED_PENALTY
    mu, dv, mv, A = result
    valid = dv > 0
    if not np.any(valid):
        return DISCONNECTED_PENALTY
    bound_values = np.where(valid, (mv + 3.0 * dv) / 2.0, 0.0)
    return mu - np.max(bound_values[valid])


def calc_score_bound33_numpy(state, n, triu_rows, triu_cols):
    """Secondary attack (OPEN, edge-max): mu <= max_{vi~vj} (2*(d_i+d_j)-(m_i+m_j)) [numpy]"""
    result = compute_graph_properties_numpy(state, n, triu_rows, triu_cols)
    if result is None:
        return DISCONNECTED_PENALTY
    mu, dv, mv, A = result
    edge_rows, edge_cols = np.nonzero(np.triu(A))
    if len(edge_rows) == 0:
        return DISCONNECTED_PENALTY
    d_i, d_j = dv[edge_rows], dv[edge_cols]
    m_i, m_j = mv[edge_rows], mv[edge_cols]
    bound_values = 2.0 * (d_i + d_j) - (m_i + m_j)
    return mu - np.max(bound_values)


# ── Reward function registry ────────────────────────────────────
# fn: NetworkX version (for verify_counterexample and backward compat)
# fn_numpy: numpy bypass version (for CEM hot loop via functools.partial)

BOUND_REGISTRY = {
    'bound1': {'fn': calc_score_bound1_nx, 'fn_numpy': calc_score_bound1_numpy,
               'name': 'Bound 1 (OPEN)',
               'formula': 'mu <= max_v sqrt(4*d_v^3/m_v)'},
    'bound2': {'fn': calc_score_bound2_nx, 'fn_numpy': calc_score_bound2_numpy,
               'name': 'Bound 2 (DISPROVED)',
               'formula': 'mu <= max_v (2*m_v^2/d_v)'},
    'bound3': {'fn': calc_score_bound3_nx, 'fn_numpy': calc_score_bound3_numpy,
               'name': 'Bound 3 (DISPROVED, RL-findable)',
               'formula': 'mu <= max_v (m_v^2/d_v + m_v)'},
    'bound4': {'fn': calc_score_bound4_nx, 'fn_numpy': calc_score_bound4_numpy,
               'name': 'Bound 4 (OPEN)',
               'formula': 'mu <= max_v (2*d_v^2/m_v)'},
    'bound9': {'fn': calc_score_bound9_nx, 'fn_numpy': calc_score_bound9_numpy,
               'name': 'Bound 9 (OPEN)',
               'formula': 'mu <= max_v (m_v+3*d_v)/2'},
    'bound33': {'fn': calc_score_bound33_nx, 'fn_numpy': calc_score_bound33_numpy,
                'name': 'Bound 33 (OPEN, edge-max)',
                'formula': 'mu <= max_{vi~vj} (2*(d_i+d_j)-(m_i+m_j))'},
}

# Quick test: compare NetworkX vs numpy on K_5
print("Reward function test on K_5 (n=5, complete graph):")
state_k5 = np.ones(10, dtype=np.int32)
triu_r5, triu_c5 = precompute_triu_indices(5)
for key, info in BOUND_REGISTRY.items():
    score_nx = info['fn'](state_k5, 5)
    score_np = info['fn_numpy'](state_k5, 5, triu_r5, triu_c5)
    match = abs(score_nx - score_np) < 1e-10
    tag = "MATCH" if match else "MISMATCH"
    print(f"  {info['name']}: nx={score_nx:.6f}, np={score_np:.6f} -> [{tag}]")
print("\n(Negative rewards on K_5 are expected — regular graphs satisfy most bounds.)")

# === Code Cell 6 ===
"""Cell 7: CEM Training Engine (Ghebleh 2024 settings)"""

def generate_sessions(model, n_sessions, n, calc_score_fn, randomness=0.0):
    """Generate n_sessions graph construction episodes in parallel.

    At each of DECISIONS steps:
      1. Model outputs 2 logits per session
      2. Apply softmax to get edge probability (class 1 = add edge)
      3. Sample binary action (add edge or skip)
      4. When randomness > 0, override some actions with random choices

    After all decisions, compute reward for each complete graph.

    Args:
        model: CEMModel instance (2-output logits)
        n_sessions: number of parallel sessions
        n: number of vertices
        calc_score_fn: reward function (state, n) -> float
        randomness: fraction of actions to override with random (0.0 = none)

    Returns:
        states_record: numpy array (n_sessions, DECISIONS, 2*DECISIONS)
                       — state at each step for each session
        actions: numpy array (n_sessions, DECISIONS) — action at each step
        rewards: numpy array (n_sessions,) — final reward per session
    """
    decisions = n * (n - 1) // 2
    obs_dim = 2 * decisions

    # Initialize states: all zeros
    current_states = np.zeros((n_sessions, obs_dim), dtype=np.float32)
    current_states[:, decisions] = 1.0

    # Storage for training data
    states_record = np.zeros((n_sessions, decisions, obs_dim), dtype=np.float32)
    actions = np.zeros((n_sessions, decisions), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for step in range(decisions):
            states_record[:, step, :] = current_states

            state_tensor = torch.from_numpy(current_states).to(DEVICE)
            logits = model(state_tensor).cpu().numpy()  # (n_sessions, 2)

            # Softmax to get edge probabilities (class 1 = add edge)
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            edge_probs = softmax_probs[:, 1]  # P(add edge)

            random_vals = np.random.random(n_sessions).astype(np.float32)
            action = (random_vals < edge_probs).astype(np.float32)

            # Adaptive action randomness (Ghebleh 2024)
            if randomness > 0:
                random_mask = np.random.random(n_sessions) < randomness
                random_actions = np.random.randint(0, 2, size=n_sessions).astype(np.float32)
                action[random_mask] = random_actions[random_mask]

            actions[:, step] = action

            current_states[action == 1, step] = 1.0

            current_states[:, decisions + step] = 0.0
            if step + 1 < decisions:
                current_states[:, decisions + step + 1] = 1.0

    # Compute rewards for all complete graphs
    rewards = np.zeros(n_sessions, dtype=np.float64)
    for i in range(n_sessions):
        adj_state = current_states[i, :decisions].astype(np.int32)
        rewards[i] = calc_score_fn(adj_state, n)

    return states_record, actions, rewards, current_states[:, :decisions].astype(np.int32).copy()


def select_elites(states_record, actions, rewards, percentile):
    """Select elite sessions and flatten into (state, action) training pairs."""
    threshold = np.percentile(rewards, percentile)
    elite_mask = rewards >= threshold

    elite_states = states_record[elite_mask]
    elite_actions = actions[elite_mask]

    n_elite = elite_states.shape[0]
    decisions = elite_states.shape[1]
    obs_dim = elite_states.shape[2]

    elite_states_flat = elite_states.reshape(-1, obs_dim)
    elite_actions_flat = elite_actions.reshape(-1)

    return elite_states_flat, elite_actions_flat, threshold


def select_super_sessions(states_record, actions, rewards, super_percentile, n_sessions):
    """Select super-elite sessions to carry forward to next generation."""
    threshold = np.percentile(rewards, super_percentile)
    super_mask = rewards >= threshold

    super_states = states_record[super_mask]
    super_actions = actions[super_mask]
    super_rewards = rewards[super_mask]

    sort_idx = np.argsort(-super_rewards)[:n_sessions]

    return super_states[sort_idx], super_actions[sort_idx], super_rewards[sort_idx]


def verify_counterexample(state, n, bound_key):
    """Independently verify a potential counterexample.

    NOTE: This intentionally uses NetworkX (not numpy bypass) for independent
    cross-verification. The CEM loop uses numpy for speed, but verification
    must use a separate code path to catch implementation bugs.
    """
    info = BOUND_REGISTRY[bound_key]
    G = build_graph(state, n)

    if not nx.is_connected(G):
        return False, "Graph is disconnected"

    mu = compute_laplacian_spectral_radius(G)
    dv, mv = compute_dv_mv(G, n)

    # Use NetworkX version for verification
    reward = info['fn'](state, n)

    result = {
        'mu': mu,
        'reward': reward,
        'n_vertices': n,
        'n_edges': G.number_of_edges(),
        'degrees': dv.tolist(),
        'avg_neighbor_degrees': mv.tolist(),
        'is_counterexample': reward > 0,
        'graph': G,
    }

    return reward > 0, result


def train_cem(n, bound_key, max_iter=5000, n_sessions=200,
              percentile=90, super_percentile=97.5, lr=0.003,
              log_interval=20, early_stop_reward=0.0,
              time_limit_seconds=None):
    """Main CEM training loop (Ghebleh 2024 settings).

    Uses Adam optimizer + CrossEntropyLoss with 2-output logits.
    Implements Ghebleh adaptive action randomness.

    Args:
        n: number of vertices
        bound_key: key into BOUND_REGISTRY
        max_iter: maximum training iterations
        n_sessions: sessions per generation
        percentile: elite selection threshold
        super_percentile: super-elite carry-forward threshold
        lr: Adam learning rate
        log_interval: print every N iterations
        early_stop_reward: stop if best reward exceeds this
        time_limit_seconds: optional time limit

    Returns:
        best_state: adjacency vector of best graph found
        best_reward: reward of best graph
        history: dict with training history
    """
    decisions = n * (n - 1) // 2
    obs_dim = 2 * decisions
    info = BOUND_REGISTRY[bound_key]

    # ── Numpy bypass: precompute triu indices and bind via partial ──
    triu_rows, triu_cols = precompute_triu_indices(n)
    calc_score_fn = partial(info['fn_numpy'], triu_rows=triu_rows, triu_cols=triu_cols)

    print(f"\n{'='*70}")
    print(f"CEM Training: {info['name']}")
    print(f"Formula: {info['formula']}")
    print(f"N={n}, decisions={decisions}, state_dim={obs_dim}")
    print(f"Sessions={n_sessions}, elite={percentile}%, super={super_percentile}%")
    print(f"Max iterations={max_iter}, LR={lr} (Adam)")
    print(f"Loss: CrossEntropyLoss, Output: 2 logits")
    print(f"Adaptive randomness: Ghebleh(2024) — init=0.005, wait=10, mult=1.1, cap=0.025")
    print(f"Reward computation: numpy bypass (triu indices precomputed)")
    print(f"{'='*70}\n")

    # Create model and optimizer
    model = create_model(n)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Initialize super-session storage (empty)
    super_states = np.empty((0, decisions, obs_dim), dtype=np.float32)
    super_actions = np.empty((0, decisions), dtype=np.float32)
    super_rewards = np.empty((0,), dtype=np.float64)
    super_final_adj = np.empty((0, decisions), dtype=np.int32)

    # Tracking
    best_reward = -float('inf')
    best_state = None
    history = {
        'iteration': [],
        'best_reward': [],
        'mean_elite_reward': [],
        'mean_reward': [],
        'elite_threshold': [],
        'time_elapsed': [],
    }

    # Ghebleh adaptive randomness
    randomness = 0.005  # initial randomness
    no_improve_count = 0

    start_time = time.time()
    counterexample_found = False

    for iteration in range(1, max_iter + 1):
        # Check time limit
        elapsed = time.time() - start_time
        if time_limit_seconds and elapsed > time_limit_seconds:
            print(f"\nTime limit reached ({elapsed:.0f}s / {time_limit_seconds}s). Stopping.")
            break

        # 1. Generate sessions (uses numpy bypass calc_score_fn)
        sess_states, sess_actions, sess_rewards, sess_final_adj = generate_sessions(
            model, n_sessions, n, calc_score_fn, randomness=randomness
        )

        # 2. Combine with super-sessions from previous generation
        if len(super_rewards) > 0:
            all_states = np.concatenate([sess_states, super_states], axis=0)
            all_actions = np.concatenate([sess_actions, super_actions], axis=0)
            all_rewards = np.concatenate([sess_rewards, super_rewards], axis=0)
            all_final_adj = np.concatenate([sess_final_adj, super_final_adj], axis=0)
        else:
            all_states = sess_states
            all_actions = sess_actions
            all_rewards = sess_rewards
            all_final_adj = sess_final_adj

        # 3. Select elites for training
        elite_states_flat, elite_actions_flat, elite_threshold = select_elites(
            all_states, all_actions, all_rewards, percentile
        )

        # 4. Select super-sessions to carry forward
        super_states, super_actions, super_rewards = select_super_sessions(
            all_states, all_actions, all_rewards, super_percentile, n_sessions
        )
        super_threshold = np.percentile(all_rewards, super_percentile)
        super_mask_adj = all_rewards >= super_threshold
        super_final_adj_candidates = all_final_adj[super_mask_adj]
        super_sort_idx = np.argsort(-all_rewards[super_mask_adj])[:n_sessions]
        super_final_adj = super_final_adj_candidates[super_sort_idx]

        # 5. Train model on elite data
        model.train()
        elite_s_tensor = torch.from_numpy(elite_states_flat).to(DEVICE)
        elite_a_tensor = torch.from_numpy(elite_actions_flat).long().to(DEVICE)

        batch_size = 2048
        n_samples = elite_s_tensor.shape[0]
        indices = torch.randperm(n_samples)
        total_loss = 0.0
        n_batches = 0

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_idx = indices[start_idx:end_idx]

            batch_states = elite_s_tensor[batch_idx]
            batch_actions = elite_a_tensor[batch_idx]

            optimizer.zero_grad()
            predictions = model(batch_states)
            loss = loss_fn(predictions, batch_actions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # 6. Track best result + adaptive randomness
        iter_best_idx = np.argmax(all_rewards)
        iter_best_reward = all_rewards[iter_best_idx]

        if iter_best_reward > best_reward:
            best_reward = iter_best_reward
            best_state = all_final_adj[iter_best_idx].copy()
            no_improve_count = 0
            randomness = 0.005  # reset to initial on improvement
        else:
            no_improve_count += 1
            if no_improve_count >= 10:  # Ghebleh: wait=10
                randomness = min(randomness * 1.1, 0.025)  # mult=1.1, cap=2.5%
                no_improve_count = 0

        # Record history
        elite_mask = all_rewards >= elite_threshold
        mean_elite = np.mean(all_rewards[elite_mask])

        history['iteration'].append(iteration)
        history['best_reward'].append(best_reward)
        history['mean_elite_reward'].append(mean_elite)
        history['mean_reward'].append(np.mean(sess_rewards))
        history['elite_threshold'].append(elite_threshold)
        history['time_elapsed'].append(elapsed)

        # 7. Log progress
        if iteration % log_interval == 0 or iteration == 1:
            print(f"Iter {iteration:5d} | best={best_reward:+.6f} | "
                  f"elite_mean={mean_elite:+.6f} | loss={avg_loss:.4f} | "
                  f"threshold={elite_threshold:+.4f} | "
                  f"rand={randomness:.4f} | time={elapsed:.1f}s")

        # 8. Check for counterexample
        if best_reward > early_stop_reward + 1e-6:  # +1e-6 to filter floating-point equality
            elapsed = time.time() - start_time
            print(f"\n*** COUNTEREXAMPLE FOUND at iteration {iteration}! ***")
            print(f"    Reward = {best_reward:.6f}")
            print(f"    Time = {elapsed:.1f}s")

            # Verify using NetworkX (independent cross-verification)
            is_valid, details = verify_counterexample(best_state, n, bound_key)
            if is_valid:
                print(f"    VERIFIED: mu={details['mu']:.6f}, reward={details['reward']:.6f}")
                print(f"    Vertices={details['n_vertices']}, Edges={details['n_edges']}")
                print(f"    Degrees: {details['degrees']}")
                counterexample_found = True
            else:
                print(f"    WARNING: Verification failed! {details}")
            break

    elapsed = time.time() - start_time
    print(f"\nTraining complete. Total time: {elapsed:.1f}s")
    print(f"Best reward: {best_reward:+.6f}")

    if not counterexample_found and best_reward > -100:
        print(f"\nNear-miss analysis (best graph, reward={best_reward:.6f}):")
        if best_state is not None:
            G_best = build_graph(best_state, n)
            if nx.is_connected(G_best):
                mu = compute_laplacian_spectral_radius(G_best)
                dv, mv = compute_dv_mv(G_best, n)
                print(f"  mu (spectral radius) = {mu:.6f}")
                print(f"  Edges = {G_best.number_of_edges()}")
                print(f"  Max degree = {max(dv):.0f}, Min degree = {min(dv):.0f}")
                print(f"  Degree sequence: {sorted([int(d) for d in dv], reverse=True)}")

    return best_state, best_reward, history, counterexample_found


print("CEM training engine defined (Ghebleh 2024 settings: Adam + CrossEntropyLoss + adaptive randomness).")

# === Code Cell 7 ===
"""Cell 8: Benchmark Phase -- Bound 3 n-sweep + Attack Phase -- Open bounds n-sweep"""

# Store all results for summary
ALL_RESULTS = {}

# ── n-sweep configuration ──────────────────────────────────────
N_SWEEP = [8, 10, 12, 14, 16, 18, 20]
BENCHMARK_BOUND = 'bound3'
ATTACK_BOUNDS = ['bound1', 'bound4', 'bound9', 'bound33']

# ── Phase 1: Benchmark — Bound 3 across N_SWEEP ────────────────
timer.start_phase('benchmark')
timer.status()

print("=" * 70)
print(f"BENCHMARK PHASE: {BOUND_REGISTRY[BENCHMARK_BOUND]['name']}")
print(f"Formula: {BOUND_REGISTRY[BENCHMARK_BOUND]['formula']}")
print(f"N-sweep: {N_SWEEP}")
print("Expected: Counterexample should be found (bound disproved)")
print("=" * 70)

for n_val in N_SWEEP:
    can_run, avail = timer.can_start('benchmark', min_useful_time=60)
    if not can_run:
        print(f"\nSKIP: bench_b3_n{n_val} -- insufficient time ({avail:.0f}s remaining)")
        continue

    time_limit = timer.get_time_limit('benchmark', f'bench_b3_n{n_val}')
    # Distribute remaining benchmark time among remaining n values
    remaining_ns = [nv for nv in N_SWEEP if f'bench_b3_n{nv}' not in ALL_RESULTS]
    if len(remaining_ns) > 0:
        time_limit = min(time_limit, timer.remaining() / len(remaining_ns))
    if time_limit < 60:
        print(f"\nSKIP: bench_b3_n{n_val} -- only {time_limit:.0f}s available")
        continue

    print(f"\n--- Benchmark: {BENCHMARK_BOUND}, n={n_val} (limit {time_limit/60:.1f}min) ---")

    b_state, b_reward, b_history, b_found = train_cem(
        n=n_val,
        bound_key=BENCHMARK_BOUND,
        max_iter=5000,
        n_sessions=200,
        percentile=90,
        super_percentile=97.5,
        lr=0.003,
        log_interval=50,
        early_stop_reward=0.0,
        time_limit_seconds=time_limit,
    )

    result_key = f'bench_b3_n{n_val}'
    ALL_RESULTS[result_key] = {
        'state': b_state,
        'reward': b_reward,
        'history': b_history,
        'found': b_found,
        'n': n_val,
        'bound': BENCHMARK_BOUND,
    }

    if b_found:
        print(f"  bench_b3_n{n_val}: COUNTEREXAMPLE FOUND (reward={b_reward:+.6f})")
    else:
        print(f"  bench_b3_n{n_val}: best reward={b_reward:+.6f}")

timer.end_phase('benchmark')
print("\nBenchmark phase complete.")
timer.status()

# ── Phase 2: Attack — Open bounds across N_SWEEP ───────────────
timer.start_phase('secondary')
timer.status()

print("=" * 70)
print("ATTACK PHASE: Open bounds n-sweep")
print(f"Bounds: {ATTACK_BOUNDS}")
print(f"N-sweep: {N_SWEEP}")
print("=" * 70)

for attack_bound in ATTACK_BOUNDS:
    bound_info = BOUND_REGISTRY[attack_bound]
    print(f"\n{'='*60}")
    print(f"ATTACK: {bound_info['name']}")
    print(f"Formula: {bound_info['formula']}")
    print(f"{'='*60}")

    for n_val in N_SWEEP:
        result_key = f'{attack_bound}_n{n_val}'

        can_run, avail = timer.can_start('secondary', min_useful_time=60)
        if not can_run:
            print(f"\nSKIP: {result_key} -- insufficient time ({avail:.0f}s remaining)")
            continue

        time_limit = timer.get_time_limit('secondary', result_key)
        # Distribute remaining time among remaining attack runs
        remaining_runs = sum(
            1 for ab in ATTACK_BOUNDS for nv in N_SWEEP
            if f'{ab}_n{nv}' not in ALL_RESULTS
        )
        if remaining_runs > 0:
            time_limit = min(time_limit, timer.remaining() / remaining_runs)
        if time_limit < 60:
            print(f"\nSKIP: {result_key} -- only {time_limit:.0f}s available")
            continue

        print(f"\n--- Attack: {attack_bound}, n={n_val} (limit {time_limit/60:.1f}min) ---")

        a_state, a_reward, a_history, a_found = train_cem(
            n=n_val,
            bound_key=attack_bound,
            max_iter=5000,
            n_sessions=200,
            percentile=90,
            super_percentile=97.5,
            lr=0.003,
            log_interval=50,
            early_stop_reward=0.0,
            time_limit_seconds=time_limit,
        )

        ALL_RESULTS[result_key] = {
            'state': a_state,
            'reward': a_reward,
            'history': a_history,
            'found': a_found,
            'n': n_val,
            'bound': attack_bound,
        }

        if a_found:
            print(f"  {result_key}: *** COUNTEREXAMPLE FOUND *** (reward={a_reward:+.6f})")
        else:
            print(f"  {result_key}: best reward={a_reward:+.6f}")

timer.end_phase('secondary')
print("\nAttack phase complete.")
timer.status()

# === Code Cell 12 ===
"""Cell 13: Results Summary + Visualization"""

timer.status()

# ==================================================================
# AUTO-SUMMARY TABLE
# ==================================================================
print("=" * 90)
print("RESULTS SUMMARY")
print("=" * 90)
print(f"{'Run':<20} {'N':>3} {'Bound':<20} {'Best Reward':>12} {'Found':>6} {'Iters':>6} {'Time':>8}")
print("-" * 90)

for key, result in ALL_RESULTS.items():
    n = result['n']
    bound = result['bound']
    reward = result['reward']
    found = "YES" if result['found'] else "no"
    iters = len(result['history']['iteration']) if result['history']['iteration'] else 0
    elapsed = result['history']['time_elapsed'][-1] if result['history']['time_elapsed'] else 0
    bound_name = BOUND_REGISTRY[bound]['name']

    print(f"{key:<20} {n:>3} {bound_name:<20} {reward:>+12.6f} {found:>6} {iters:>6} {elapsed:>7.1f}s")

print("=" * 90)

n_found = sum(1 for r in ALL_RESULTS.values() if r['found'])
total_time = timer.elapsed()
print(f"\nCounterexamples found: {n_found} / {len(ALL_RESULTS)}")
print(f"Total notebook time: {total_time/3600:.2f}h")

# ==================================================================
# REWARD CURVES (enhanced: 4-panel with time axis + exploration quality)
# ==================================================================
if ALL_RESULTS:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Best reward over iterations
    ax1 = axes[0, 0]
    for key, result in ALL_RESULTS.items():
        h = result['history']
        if h['iteration']:
            label = f"{key} (n={result['n']})"
            ax1.plot(h['iteration'], h['best_reward'], label=label, alpha=0.8, linewidth=1.5)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Best Reward')
    ax1.set_title('Best Reward over Training')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Counterexample threshold')
    ax1.legend(fontsize=7, loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Mean elite reward over iterations
    ax2 = axes[0, 1]
    for key, result in ALL_RESULTS.items():
        h = result['history']
        if h['iteration']:
            label = f"{key} (n={result['n']})"
            ax2.plot(h['iteration'], h['mean_elite_reward'], label=label, alpha=0.8, linewidth=1.5)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Mean Elite Reward')
    ax2.set_title('Mean Elite Reward over Training')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Counterexample threshold')
    ax2.legend(fontsize=7, loc='lower right')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Best reward over WALL TIME
    ax3 = axes[1, 0]
    for key, result in ALL_RESULTS.items():
        h = result['history']
        if h['time_elapsed']:
            label = f"{key} (n={result['n']})"
            time_hours = [t / 3600 for t in h['time_elapsed']]
            ax3.plot(time_hours, h['best_reward'], label=label, alpha=0.8, linewidth=1.5)
    ax3.set_xlabel('Wall Time (hours)')
    ax3.set_ylabel('Best Reward')
    ax3.set_title('Best Reward over Wall Time')
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Counterexample threshold')
    ax3.legend(fontsize=7, loc='lower right')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Mean reward (all sessions) -- exploration quality
    ax4 = axes[1, 1]
    for key, result in ALL_RESULTS.items():
        h = result['history']
        if h['iteration']:
            label = f"{key} (n={result['n']})"
            ax4.plot(h['iteration'], h['mean_reward'], label=label, alpha=0.8, linewidth=1.5)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Mean Reward (all sessions)')
    ax4.set_title('Mean Reward over Training (Exploration Quality)')
    ax4.legend(fontsize=7, loc='lower right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reward_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: reward_curves.png")

# ==================================================================
# COUNTEREXAMPLE GRAPH VISUALIZATION (color by degree, with colorbar)
# ==================================================================
counterexample_graphs = []
nearmiss_graphs = []

for key, result in ALL_RESULTS.items():
    if result['state'] is not None and result['reward'] > DISCONNECTED_PENALTY:
        G = build_graph(result['state'], result['n'])
        if nx.is_connected(G):
            if result['found']:
                counterexample_graphs.append((key, G, result))
            else:
                nearmiss_graphs.append((key, G, result))

# Draw counterexamples with detailed annotations
if counterexample_graphs:
    n_ce = len(counterexample_graphs)
    fig, axes = plt.subplots(1, n_ce, figsize=(6 * n_ce, 6))
    if n_ce == 1:
        axes = [axes]

    for idx, (key, G, result) in enumerate(counterexample_graphs):
        ax = axes[idx]
        pos = nx.spring_layout(G, seed=42)
        degrees = [G.degree(v) for v in G.nodes()]

        node_colors = nx.draw_networkx_nodes(G, pos, ax=ax, node_color=degrees,
                                              cmap=plt.cm.YlOrRd, node_size=300, alpha=0.9)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.5)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight='bold')
        plt.colorbar(node_colors, ax=ax, label='Degree', shrink=0.8)

        n_v = result['n']
        bound_name = BOUND_REGISTRY[result['bound']]['name']
        ax.set_title(f"COUNTEREXAMPLE: {key}\n{bound_name}\n"
                     f"n={n_v}, edges={G.number_of_edges()}, reward={result['reward']:+.4f}",
                     fontsize=10, fontweight='bold', color='darkgreen')

    plt.tight_layout()
    plt.savefig('counterexample_graphs.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: counterexample_graphs.png")

# Draw top near-miss graphs
if nearmiss_graphs:
    nearmiss_graphs.sort(key=lambda x: -x[2]['reward'])
    top_nearmiss = nearmiss_graphs[:4]
    n_nm = len(top_nearmiss)

    fig, axes = plt.subplots(1, n_nm, figsize=(5 * n_nm, 5))
    if n_nm == 1:
        axes = [axes]

    for idx, (key, G, result) in enumerate(top_nearmiss):
        ax = axes[idx]
        pos = nx.spring_layout(G, seed=42)
        degrees = [G.degree(v) for v in G.nodes()]

        nx.draw(G, pos, ax=ax, node_color=degrees, cmap=plt.cm.viridis,
                node_size=200, font_size=8, with_labels=True,
                edge_color='gray', alpha=0.8)

        ax.set_title(f"{key} (n={result['n']})\nreward={result['reward']:+.4f}\nNear-miss",
                     fontsize=10)

    plt.tight_layout()
    plt.savefig('nearmiss_graphs.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: nearmiss_graphs.png")

# ==================================================================
# NEAR-MISS ANALYSIS (enhanced: structural properties + tightest vertex)
# ==================================================================
print("\n" + "=" * 80)
print("NEAR-MISS ANALYSIS -- Structural Properties")
print("=" * 80)

for key, result in ALL_RESULTS.items():
    if result['state'] is not None and not result['found'] and result['reward'] > DISCONNECTED_PENALTY:
        n = result['n']
        state = result['state']
        G = build_graph(state, n)

        if not nx.is_connected(G):
            continue

        mu = compute_laplacian_spectral_radius(G)
        dv, mv = compute_dv_mv(G, n)
        degree_seq = sorted([int(d) for d in dv], reverse=True)

        print(f"\n{'---'*24}")
        print(f"  {key} (n={n}, bound={BOUND_REGISTRY[result['bound']]['name']})")
        print(f"{'---'*24}")
        print(f"  Best reward:        {result['reward']:+.6f}")
        print(f"  Gap to violation:   {abs(result['reward']):.6f}")
        print(f"  mu (spectral rad.): {mu:.6f}")
        print(f"  Bound formula:      {BOUND_REGISTRY[result['bound']]['formula']}")
        print(f"")
        print(f"  --- Graph Structure ---")
        print(f"  Vertices:           {n}")
        print(f"  Edges:              {G.number_of_edges()}")
        print(f"  Density:            {nx.density(G):.4f}")
        print(f"  Diameter:           {nx.diameter(G)}")
        print(f"  Radius:             {nx.radius(G)}")
        print(f"  Avg clustering:     {nx.average_clustering(G):.4f}")
        print(f"  Transitivity:       {nx.transitivity(G):.4f}")
        print(f"")
        print(f"  --- Degree Stats ---")
        print(f"  Degree sequence:    {degree_seq}")
        print(f"  Max degree:         {max(dv):.0f}")
        print(f"  Min degree:         {min(dv):.0f}")
        print(f"  Mean degree:        {np.mean(dv):.2f}")
        print(f"  Degree std:         {np.std(dv):.2f}")
        print(f"")
        print(f"  --- Avg Neighbor Degree ---")
        print(f"  Max m_v:            {max(mv):.4f}")
        print(f"  Min m_v:            {min(mv[dv > 0]):.4f}")
        print(f"  Mean m_v:           {np.mean(mv[dv > 0]):.4f}")

        # Identify the vertex closest to violating the bound
        bound_key_for_result = result['bound']
        per_vertex = None
        if bound_key_for_result != 'bound33':  # vertex-max bounds
            if bound_key_for_result == 'bound1':
                per_vertex = np.where(
                    (dv > 0) & (mv > 0),
                    np.sqrt(4.0 * dv**3 / np.maximum(mv, 1e-300)),
                    float('inf')
                )
            elif bound_key_for_result == 'bound2':
                per_vertex = np.where(
                    dv > 0,
                    2.0 * mv**2 / np.maximum(dv, 1e-300),
                    float('inf')
                )
            elif bound_key_for_result == 'bound4':
                per_vertex = np.where(
                    (dv > 0) & (mv > 0),
                    2.0 * dv**2 / np.maximum(mv, 1e-300),
                    float('inf')
                )
            elif bound_key_for_result == 'bound3':
                per_vertex = np.where(
                    dv > 0,
                    mv**2 / np.maximum(dv, 1e-300) + mv,
                    float('inf')
                )
            elif bound_key_for_result == 'bound9':
                per_vertex = np.where(
                    dv > 0,
                    (mv + 3.0 * dv) / 2.0,
                    float('inf')
                )

            if per_vertex is not None:
                tightest_v = np.argmin(per_vertex)
                print(f"\n  --- Tightest Bound Vertex ---")
                print(f"  Vertex {tightest_v}: bound_value={per_vertex[tightest_v]:.6f}, "
                      f"d_v={dv[tightest_v]:.0f}, m_v={mv[tightest_v]:.4f}")
                print(f"  mu - bound_at_v = {mu - per_vertex[tightest_v]:+.6f}")

print("\n" + "=" * 80)
print("Analysis complete.")
print("=" * 80)
