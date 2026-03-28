#!/usr/bin/env python3
"""Cycle 2 Analysis Phase 2: Deep investigation of tight bounds + creative families."""

import sys
sys.path.insert(0, 'src')

import numpy as np
from exhaustive_bound_search import (
    evaluate_all_bounds, compute_dv_mv, laplacian_spectral_radius,
    compute_vertex_bounds, compute_edge_bounds
)

remaining_vertex = [1,4,5,6,7,8,9,10,12,14,16,18,19,20,21,22,23,24,25,26,27,30]
remaining_edge = [33,34,35,37,38,39,42,44,46,47,56]
remaining_all = remaining_vertex + remaining_edge

# =====================================================================
# Verify "violations" are just floating point noise
# =====================================================================
print("VERIFICATION: Bounds 33, 42, 47 on Star(k)")
print("=" * 70)

def make_star(k):
    n = k + 1
    A = np.zeros((n, n))
    for i in range(1, n):
        A[0, i] = A[i, 0] = 1.0
    return A

for k in [5, 10, 14, 15, 20, 100]:
    A = make_star(k)
    mu, bvals, gaps = evaluate_all_bounds(A)
    print(f"Star({k}): mu={mu:.15f}")
    for bid in [33, 42, 47]:
        print(f"  Bound {bid}: val={bvals[bid]:.15f}, gap={gaps[bid]:.2e}")

print()
print("Conclusion: gaps are -0.0 or +0.0 (floating point). Star gives EXACT equality, not violation.")

# =====================================================================
# DS(1,1) = P3 is the tightest for many vertex bounds
# Why? Let's analyze P3 specifically
# =====================================================================
print()
print("DS(1,1) = P3 analysis")
print("=" * 70)
A_p3 = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=float)
mu, bvals, gaps = evaluate_all_bounds(A_p3)
dv, mv = compute_dv_mv(A_p3)
print(f"P3: vertices degrees = {dv}, m-values = {mv}")
print(f"P3: mu = {mu:.6f}")
for bid in remaining_all:
    if gaps[bid] < 1.0:
        print(f"  Bound {bid}: val={bvals[bid]:.6f}, gap={gaps[bid]:.6f}")

# =====================================================================
# Creative graph families targeting Bound 42
# Key: Bound 42 = max_(i~j) sqrt(di^2 + dj^2 + 2*mi*mj)
# Need: large mu but small mi*mj on high-degree edge
# Strategy: maximize spectral radius while keeping neighbor-degree products low
# =====================================================================
print()
print("CREATIVE FAMILIES TARGETING BOUND 42")
print("=" * 70)

# Family A: Bridged BiStar -- two hubs connected, each hub connected to
# k cliques of size c (not just leaves)
print()
print("--- Family A: HubWithCliques(k, c) ---")
print("Hub connected to k disjoint K_c cliques (hub-to-one-vertex-in-clique)")

def make_hub_with_cliques(k, c):
    """Hub connected to one vertex in each of k disjoint K_c cliques."""
    n = 1 + k * c
    A = np.zeros((n, n))
    for i in range(k):
        base = 1 + i * c
        # Clique among base..base+c-1
        for a in range(c):
            for b in range(a+1, c):
                A[base+a, base+b] = A[base+b, base+a] = 1.0
        # Hub connected to base vertex
        A[0, base] = A[base, 0] = 1.0
    return A

for k in [5, 10, 20, 50]:
    for c in [2, 3, 4, 5]:
        A = make_hub_with_cliques(k, c)
        mu, bvals, gaps = evaluate_all_bounds(A)
        tight = [(bid, gaps[bid]) for bid in remaining_all if gaps[bid] < 0.5]
        violations = [(bid, gaps[bid]) for bid in remaining_all if gaps[bid] < -1e-10]
        if tight:
            print(f"  HWC({k},{c}): mu={mu:.4f}, tight: ", end="")
            for bid, gap in sorted(tight, key=lambda x: x[1]):
                viol = " **VIOL**" if gap < -1e-10 else ""
                print(f"B{bid}({gap:.4f}){viol} ", end="")
            print()

# Family B: Sun graph -- cycle with pendant at each vertex
print()
print("--- Family B: Sun(k) ---")
print("Cycle C_k with one pendant leaf at each vertex")

def make_sun(k):
    """Cycle of k vertices, each with one pendant leaf."""
    n = 2 * k
    A = np.zeros((n, n))
    # Cycle among 0..k-1
    for i in range(k):
        A[i, (i+1) % k] = A[(i+1) % k, i] = 1.0
    # Pendant leaves k..2k-1
    for i in range(k):
        A[i, k+i] = A[k+i, i] = 1.0
    return A

for k in [3, 4, 5, 10, 20]:
    A = make_sun(k)
    mu, bvals, gaps = evaluate_all_bounds(A)
    tight = [(bid, gaps[bid]) for bid in remaining_all if gaps[bid] < 1.0]
    if tight:
        print(f"  Sun({k}): mu={mu:.4f}, tight: ", end="")
        for bid, gap in sorted(tight, key=lambda x: x[1]):
            print(f"B{bid}({gap:.4f}) ", end="")
        print()

# Family C: Complete bipartite K_{a,b}
print()
print("--- Family C: Complete Bipartite K_{a,b} ---")

def make_complete_bipartite(a, b):
    n = a + b
    A = np.zeros((n, n))
    for i in range(a):
        for j in range(a, a+b):
            A[i,j] = A[j,i] = 1.0
    return A

for a in [1, 2, 3, 5, 10]:
    for b in [a, a+1, 2*a, 5*a, 10*a, 50]:
        if b < a:
            continue
        A = make_complete_bipartite(a, b)
        mu, bvals, gaps = evaluate_all_bounds(A)
        tight = [(bid, gaps[bid]) for bid in remaining_all if gaps[bid] < 0.5]
        violations = [(bid, gaps[bid]) for bid in remaining_all if gaps[bid] < -1e-10]
        if tight:
            print(f"  K_{{{a},{b}}}: mu={mu:.4f}, tight: ", end="")
            for bid, gap in sorted(tight, key=lambda x: x[1]):
                viol = " **VIOL**" if gap < -1e-10 else ""
                print(f"B{bid}({gap:.4f}){viol} ", end="")
            print()

# Family D: Friendship graph + extra structure
# Friendship(k) = k triangles sharing a common vertex
# This is StarOfCliques variant -- test for remaining bounds
print()
print("--- Family D: Friendship(k) ---")

def make_friendship(k):
    """k triangles sharing a common vertex (hub)."""
    n = 1 + 2*k
    A = np.zeros((n, n))
    for i in range(k):
        v1 = 1 + 2*i
        v2 = 2 + 2*i
        A[0, v1] = A[v1, 0] = 1.0
        A[0, v2] = A[v2, 0] = 1.0
        A[v1, v2] = A[v2, v1] = 1.0
    return A

for k in [3, 5, 10, 20, 50]:
    A = make_friendship(k)
    mu, bvals, gaps = evaluate_all_bounds(A)
    tight = [(bid, gaps[bid]) for bid in remaining_all if gaps[bid] < 1.0]
    violations = [(bid, gaps[bid]) for bid in remaining_all if gaps[bid] < -1e-10]
    if tight:
        print(f"  Friendship({k}): mu={mu:.4f}, tight: ", end="")
        for bid, gap in sorted(tight, key=lambda x: x[1]):
            viol = " **VIOL**" if gap < -1e-10 else ""
            print(f"B{bid}({gap:.4f}){viol} ", end="")
        print()

# Family E: Path attached to high-degree hub (asymmetric perturbation)
# Intent: longer path from hub creates higher mu via asymmetric Fiedler vector
print()
print("--- Family E: TadpoleVariant(k, p) ---")
print("Hub with k leaves + one long path of length p, where last vertex")
print("connects back to a clique of size c")

def make_tadpole_clique(k, p, c):
    """Hub with k leaves, path of length p, ending in K_c clique."""
    n = 1 + k + p + c
    A = np.zeros((n, n))
    # Hub = 0, leaves = 1..k
    for i in range(1, k+1):
        A[0,i] = A[i,0] = 1.0
    # Path from hub: k+1, ..., k+p
    prev = 0
    for j in range(k+1, k+1+p):
        A[prev, j] = A[j, prev] = 1.0
        prev = j
    # Clique at the end: k+p+1, ..., k+p+c
    clique_start = k + 1 + p
    # Connect last path vertex to clique
    A[prev, clique_start] = A[clique_start, prev] = 1.0
    for a in range(clique_start, clique_start + c):
        for b in range(a+1, clique_start + c):
            A[a,b] = A[b,a] = 1.0
    return A

for k in [10, 20, 50]:
    for p in [1, 2, 5]:
        for c in [3, 5, 10]:
            A = make_tadpole_clique(k, p, c)
            mu, bvals, gaps = evaluate_all_bounds(A)
            tight = [(bid, gaps[bid]) for bid in remaining_all if gaps[bid] < 0.5]
            violations = [(bid, gaps[bid]) for bid in remaining_all if gaps[bid] < -1e-10]
            if violations:
                print(f"  TadCliq({k},{p},{c}): mu={mu:.4f} **VIOLATIONS**: ", end="")
                for bid, gap in violations:
                    print(f"B{bid}({gap:.4f}) ", end="")
                print()
            elif tight:
                tight_str = " ".join(f"B{bid}({gap:.3f})" for bid, gap in sorted(tight, key=lambda x: x[1])[:5])
                print(f"  TadCliq({k},{p},{c}): mu={mu:.4f}, tight: {tight_str}")

# Family F: ThickStar -- hub connected to k vertices, those k vertices
# also form a cycle or path among themselves
print()
print("--- Family F: ThickStar(k) (star + cycle on leaves) = Wheel ---")

def make_wheel(k):
    """Wheel: hub + cycle of k vertices."""
    n = k + 1
    A = np.zeros((n, n))
    for i in range(1, k+1):
        A[0, i] = A[i, 0] = 1.0
    for i in range(1, k):
        A[i, i+1] = A[i+1, i] = 1.0
    A[1, k] = A[k, 1] = 1.0
    return A

for k in [4, 5, 10, 20, 50, 100]:
    A = make_wheel(k)
    mu, bvals, gaps = evaluate_all_bounds(A)
    tight = [(bid, gaps[bid]) for bid in remaining_all if gaps[bid] < 1.0]
    violations = [(bid, gaps[bid]) for bid in remaining_all if gaps[bid] < -1e-10]
    if tight:
        print(f"  Wheel({k}): mu={mu:.4f}, tight: ", end="")
        for bid, gap in sorted(tight, key=lambda x: x[1]):
            viol = " **VIOL**" if gap < -1e-10 else ""
            print(f"B{bid}({gap:.4f}){viol} ", end="")
        print()

# Family G: Barbell(k1, k2, p) -- K_k1 connected to K_k2 via path of length p
print()
print("--- Family G: Barbell(k1, k2, p) ---")

def make_barbell(k1, k2, p):
    """Two cliques K_k1 and K_k2 connected by a path of length p."""
    n = k1 + k2 + p
    A = np.zeros((n, n))
    # Clique 1: 0..k1-1
    for i in range(k1):
        for j in range(i+1, k1):
            A[i,j] = A[j,i] = 1.0
    # Path: k1..k1+p-1
    prev = k1 - 1  # last vertex of clique 1
    for i in range(k1, k1 + p):
        A[prev, i] = A[i, prev] = 1.0
        prev = i
    # Clique 2: k1+p..k1+p+k2-1
    base2 = k1 + p
    A[prev, base2] = A[base2, prev] = 1.0
    for i in range(base2, base2 + k2):
        for j in range(i+1, base2 + k2):
            A[i,j] = A[j,i] = 1.0
    return A

for k1 in [3, 5, 10]:
    for k2 in [3, 5, 10, 20]:
        for p in [0, 1, 2, 5]:
            A = make_barbell(k1, k2, p)
            mu, bvals, gaps = evaluate_all_bounds(A)
            violations = [(bid, gaps[bid]) for bid in remaining_all if gaps[bid] < -1e-10]
            tight = [(bid, gaps[bid]) for bid in remaining_all if abs(gaps[bid]) < 0.3]
            if violations:
                print(f"  Barbell({k1},{k2},{p}): mu={mu:.4f} **VIOLATIONS**: ", end="")
                for bid, gap in violations:
                    print(f"B{bid}({gap:.4f}) ", end="")
                print()
            elif tight:
                tight_str = " ".join(f"B{bid}({gap:.3f})" for bid, gap in sorted(tight, key=lambda x: x[1])[:5])
                print(f"  Barbell({k1},{k2},{p}): mu={mu:.4f}, tight: {tight_str}")

# =====================================================================
# Cross-domain insight: What structure maximizes mu while minimizing
# the bound? The bound is local (evaluated per vertex/edge), but mu is
# global. We need a graph where the global structure (eigenvector) creates
# high mu but the local parameters (d, m) at the maximizing edge/vertex
# don't reflect this.
#
# This is a SPECTRAL vs LOCAL mismatch problem.
# Random graphs might have this property -- high connectivity but
# locally "boring" parameters.
# =====================================================================
print()
print("--- Family H: Random-like structured graphs ---")
print("Erdos-Renyi-like but deterministic: Paley graphs (not easily constructable)")
print("Instead: Circulant graphs C(n, {1,2,...,r})")

def make_circulant(n, r):
    """Circulant graph: vertex i connected to i+1,...,i+r and i-1,...,i-r (mod n)."""
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(1, r+1):
            A[i, (i+j) % n] = A[(i+j) % n, i] = 1.0
            A[i, (i-j) % n] = A[(i-j) % n, i] = 1.0
    return A

for n in [10, 20, 30, 50]:
    for r in [1, 2, 3, n//4, n//3]:
        A = make_circulant(n, r)
        mu, bvals, gaps = evaluate_all_bounds(A)
        tight = [(bid, gaps[bid]) for bid in remaining_all if gaps[bid] < 0.5]
        violations = [(bid, gaps[bid]) for bid in remaining_all if gaps[bid] < -1e-10]
        if violations:
            print(f"  Circ({n},{r}): mu={mu:.4f} **VIOLATIONS**: ", end="")
            for bid, gap in violations:
                print(f"B{bid}({gap:.4f}) ", end="")
            print()
        elif tight:
            tight_str = " ".join(f"B{bid}({gap:.3f})" for bid, gap in sorted(tight, key=lambda x: x[1])[:3])
            # Only print if interesting
            if any(g < 0.2 for _, g in tight):
                print(f"  Circ({n},{r}): mu={mu:.4f}, tight: {tight_str}")

# Family I: Kite graph variants -- clique with a tail
print()
print("--- Family I: Kite(c, t) = K_c with tail of length t ---")

def make_kite(c, t):
    """Complete graph K_c with a path of length t attached."""
    n = c + t
    A = np.zeros((n, n))
    for i in range(c):
        for j in range(i+1, c):
            A[i,j] = A[j,i] = 1.0
    prev = c - 1
    for i in range(c, c + t):
        A[prev, i] = A[i, prev] = 1.0
        prev = i
    return A

for c in [3, 5, 10, 15, 20]:
    for t in [1, 2, 5, 10]:
        A = make_kite(c, t)
        mu, bvals, gaps = evaluate_all_bounds(A)
        tight = [(bid, gaps[bid]) for bid in remaining_all if gaps[bid] < 0.3]
        violations = [(bid, gaps[bid]) for bid in remaining_all if gaps[bid] < -1e-10]
        if violations:
            print(f"  Kite({c},{t}): mu={mu:.4f} **VIOLATIONS**: ", end="")
            for bid, gap in violations:
                print(f"B{bid}({gap:.4f}) ", end="")
            print()
        elif tight:
            tight_str = " ".join(f"B{bid}({gap:.3f})" for bid, gap in sorted(tight, key=lambda x: x[1])[:3])
            print(f"  Kite({c},{t}): mu={mu:.4f}, tight: {tight_str}")

# =====================================================================
# Specific focus: Can we break Bound 10 or Bound 23?
# Bound 10: max_v sqrt(d*(d + 3*m))
# Bound 23: max_v sqrt(d^2 + 3*d*m)
# These are actually THE SAME formula! sqrt(d^2 + 3dm)
# On star hub: sqrt(k^2 + 3k) ~ k + 3/2
# mu = k+1, so gap ~ 1/2 (confirmed numerically)
# Can we make gap negative? Need graph where mu > sqrt(d^2 + 3dm) for ALL vertices
# =====================================================================
print()
print("FOCUSED: Bounds 10 = 23 (same formula: sqrt(d^2 + 3dm))")
print("=" * 70)
print("These are identical. On star, gap -> 0.5. Can we push below 0?")
print()

# The bound evaluated at a vertex v gives sqrt(d_v^2 + 3*d_v*m_v)
# = sqrt(d_v * (d_v + 3*m_v))
# For this to be < mu, we need d_v*(d_v + 3*m_v) < mu^2 for ALL v
#
# On a k-regular graph: d=m=k, bound = sqrt(k^2+3k^2) = 2k, mu = 2k. Exact.
# But that's for complete regular graphs. Non-complete regular: mu < 2k but bound = 2k. Not helpful.
#
# We need IRREGULAR graphs where mu is large relative to local d,m values.
#
# Key insight: In any graph, mu >= max_v (d_v + 1) [Grone-Merris bound from below]
# Actually: mu >= Delta + 1 for trees (star achieves this)
# For non-trees: mu can be larger.
#
# So for Bound 10/23 to be violated on vertex v:
# sqrt(d_v * (d_v + 3*m_v)) < mu
# If v is the hub with d=k, m=1: sqrt(k^2+3k) < mu
# Need mu > k + 3/2 (approx). Stars give mu = k+1.
# Can we get mu > k + 3/2 with d_hub = k, m_hub = 1?
# That means hub has k leaf neighbors, and mu > k + 3/2.
#
# If we add edges among non-hub vertices (but NOT to hub neighbors),
# mu could increase.

print("Strategy: Star(k) + extra edges among leaves (or between non-hub vertices)")
print("to increase mu while keeping hub's d=k, m=1")
print()

def make_star_plus_leaf_edges(k, extra_edges):
    """Star(k) with additional edges among leaves."""
    n = k + 1
    A = np.zeros((n, n))
    for i in range(1, n):
        A[0, i] = A[i, 0] = 1.0
    # Add edges among leaves
    for i, j in extra_edges:
        if 1 <= i < n and 1 <= j < n and i != j:
            A[i, j] = A[j, i] = 1.0
    return A

# Star(10) + edges among leaves
k = 10
print(f"Star({k}): mu={k+1}")

# Add one edge: leaf1-leaf2
edges_1 = [(1, 2)]
A = make_star_plus_leaf_edges(k, edges_1)
mu, bvals, gaps = evaluate_all_bounds(A)
dv, mv = compute_dv_mv(A)
print(f"  +1 leaf edge: mu={mu:.4f}, hub: d={dv[0]:.0f}, m={mv[0]:.4f}")
print(f"    Bound 10 at hub: {np.sqrt(dv[0]*(dv[0]+3*mv[0])):.4f}, gap={gaps[10]:.4f}")
print(f"    Bound 42 gap: {gaps[42]:.4f}")
print(f"    Bound 33 gap: {gaps[33]:.4f}")
print(f"    Bound 47 gap: {gaps[47]:.4f}")

# Add a matching among leaves (5 edges for k=10)
edges_match = [(1,2),(3,4),(5,6),(7,8),(9,10)]
A = make_star_plus_leaf_edges(k, edges_match)
mu, bvals, gaps = evaluate_all_bounds(A)
dv, mv = compute_dv_mv(A)
print(f"  +5 leaf edges (matching): mu={mu:.4f}, hub: d={dv[0]:.0f}, m={mv[0]:.4f}")
print(f"    Bound 10 at hub: {np.sqrt(dv[0]*(dv[0]+3*mv[0])):.4f}, gap={gaps[10]:.4f}")
print(f"    Bound 42 gap: {gaps[42]:.4f}")
print(f"    Bound 33 gap: {gaps[33]:.4f}")
print(f"    Bound 47 gap: {gaps[47]:.4f}")

# Add a path among all leaves
edges_path = [(i, i+1) for i in range(1, k)]
A = make_star_plus_leaf_edges(k, edges_path)
mu, bvals, gaps = evaluate_all_bounds(A)
dv, mv = compute_dv_mv(A)
print(f"  +path among leaves: mu={mu:.4f}, hub: d={dv[0]:.0f}, m={mv[0]:.4f}")
print(f"    Bound 10 at hub: {np.sqrt(dv[0]*(dv[0]+3*mv[0])):.4f}, gap={gaps[10]:.4f}")
for bid in remaining_all:
    if gaps[bid] < 0:
        print(f"    !! VIOLATION Bound {bid}: gap={gaps[bid]:.6f}")

# Add complete graph among leaves (= K_{1,k} + K_k on the k side)
# This is the complete graph K_{k+1} minus nothing... actually this is the wheel.
# No, wheel has cycle; this has all leaf edges.
# K_{k+1} has all edges. Star + all leaf edges = K_{k+1}
# Let's try partial clique on leaves
for frac in [0.1, 0.2, 0.3, 0.5]:
    np.random.seed(42)
    all_leaf_pairs = [(i, j) for i in range(1, k+1) for j in range(i+1, k+1)]
    n_edges = int(frac * len(all_leaf_pairs))
    idx = np.random.choice(len(all_leaf_pairs), n_edges, replace=False)
    extra = [all_leaf_pairs[ii] for ii in idx]
    A = make_star_plus_leaf_edges(k, extra)
    mu, bvals, gaps = evaluate_all_bounds(A)
    dv, mv = compute_dv_mv(A)
    print(f"  +{frac*100:.0f}% random leaf edges: mu={mu:.4f}, hub: d={dv[0]:.0f}, m={mv[0]:.4f}")
    for bid in [10, 23, 33, 42, 47]:
        if gaps[bid] < 0.3:
            print(f"    Bound {bid}: gap={gaps[bid]:.4f}")

print()
print("Insight: Adding leaf-leaf edges INCREASES m_hub (hub neighbors now have higher degree)")
print("This increases the bound proportionally, preventing violation.")
print("The Star topology is actually the WORST case for these bounds because m_hub=1 is minimal.")
print()

# =====================================================================
# What about Bound 19 and 20? They were very tight on DS(1,1)
# =====================================================================
print("Bounds 19, 20 tightness on small graphs")
print("=" * 70)

# DS(1,1) = P3
A = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=float)
mu, bvals, gaps = evaluate_all_bounds(A)
print(f"P3: mu = {mu:.6f}")
print(f"  Bound 19: {bvals[19]:.6f}, gap={gaps[19]:.6f}")
print(f"  Bound 20: {bvals[20]:.6f}, gap={gaps[20]:.6f}")

# P4
A = np.array([[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]], dtype=float)
mu, bvals, gaps = evaluate_all_bounds(A)
print(f"P4: mu = {mu:.6f}")
print(f"  Bound 19: {bvals[19]:.6f}, gap={gaps[19]:.6f}")
print(f"  Bound 20: {bvals[20]:.6f}, gap={gaps[20]:.6f}")

# P5
n = 5
A = np.zeros((n,n))
for i in range(n-1):
    A[i,i+1] = A[i+1,i] = 1.0
mu, bvals, gaps = evaluate_all_bounds(A)
print(f"P5: mu = {mu:.6f}")
print(f"  Bound 19: {bvals[19]:.6f}, gap={gaps[19]:.6f}")
print(f"  Bound 20: {bvals[20]:.6f}, gap={gaps[20]:.6f}")

# =====================================================================
# Final comprehensive sweep: test many more families
# =====================================================================
print()
print("FINAL COMPREHENSIVE SWEEP")
print("=" * 70)

min_gaps = {bid: (float('inf'), "none") for bid in remaining_all}

def test_graph(name, A):
    if A.shape[0] < 2:
        return
    mu, bvals, gaps = evaluate_all_bounds(A)
    for bid in remaining_all:
        if gaps[bid] < min_gaps[bid][0]:
            min_gaps[bid] = (gaps[bid], name)

def make_double_star(a, b):
    n = 2 + a + b
    A = np.zeros((n, n))
    A[0,1] = A[1,0] = 1.0
    for i in range(2, 2+a):
        A[0,i] = A[i,0] = 1.0
    for i in range(2+a, 2+a+b):
        A[1,i] = A[i,1] = 1.0
    return A

# All small connected graphs (paths, cycles, small special)
for n in range(3, 8):
    A_path = np.zeros((n,n))
    for i in range(n-1):
        A_path[i,i+1] = A_path[i+1,i] = 1.0
    test_graph(f"P{n}", A_path)

for n in range(3, 20):
    A_cyc = np.zeros((n,n))
    for i in range(n):
        A_cyc[i,(i+1)%n] = A_cyc[(i+1)%n,i] = 1.0
    test_graph(f"C{n}", A_cyc)

# Complete graphs
for n in range(3, 20):
    A_kn = np.ones((n,n)) - np.eye(n)
    test_graph(f"K{n}", A_kn)

# Complete bipartite
for a in range(1, 15):
    for b in range(a, min(100, 10*a+1)):
        test_graph(f"K{a},{b}", make_complete_bipartite(a, b))

# Stars
for k in range(2, 300):
    test_graph(f"Star({k})", make_star(k))

# Double stars
for a in range(1, 20):
    for b in range(a, min(300, 20*a)):
        test_graph(f"DS({a},{b})", make_double_star(a, b))

# Wheels
for k in range(3, 100):
    test_graph(f"Wheel({k})", make_wheel(k))

# Friendships
for k in range(2, 50):
    test_graph(f"Friendship({k})", make_friendship(k))

# Hub with cliques
for k in [3,5,10,20,50]:
    for c in [2,3,4,5]:
        test_graph(f"HWC({k},{c})", make_hub_with_cliques(k, c))

# Kites
for c in range(3, 20):
    for t in range(1, 10):
        test_graph(f"Kite({c},{t})", make_kite(c, t))

# Barbells
for k1 in range(3, 15):
    for k2 in range(k1, 20):
        for p in range(0, 5):
            test_graph(f"Barbell({k1},{k2},{p})", make_barbell(k1, k2, p))

# Circulants
for n in range(5, 50):
    for r in range(1, n//2):
        test_graph(f"Circ({n},{r})", make_circulant(n, r))

# Suns
for k in range(3, 30):
    test_graph(f"Sun({k})", make_sun(k))

# Fans
for k in range(3, 100):
    A_fan = np.zeros((k+1, k+1))
    for i in range(1, k+1):
        A_fan[0,i] = A_fan[i,0] = 1.0
    for i in range(1, k):
        A_fan[i,i+1] = A_fan[i+1,i] = 1.0
    test_graph(f"Fan({k})", A_fan)

print(f"\n{'Bound':>7s} | {'Min Gap':>12s} | {'Achieved by':>25s} | Status")
print("-" * 70)
for bid in sorted(remaining_all):
    gap, name = min_gaps[bid]
    if gap < -1e-10:
        status = "*** VIOLATION ***"
    elif gap < 0.1:
        status = "EXTREMELY TIGHT"
    elif gap < 0.5:
        status = "VERY TIGHT"
    elif gap < 2.0:
        status = "TIGHT"
    else:
        status = "robust"
    print(f"  Bound {bid:2d} | {gap:12.6f} | {name:>25s} | {status}")

violations = [(bid, min_gaps[bid]) for bid in remaining_all if min_gaps[bid][0] < -1e-10]
print()
if violations:
    print("!!! NEW VIOLATIONS FOUND !!!")
    for bid, (gap, name) in violations:
        print(f"  Bound {bid}: gap={gap:.6f} on {name}")
else:
    print("No genuine violations found among 33 remaining bounds.")
    print()
    print("Top 10 tightest bounds:")
    sorted_bounds = sorted(remaining_all, key=lambda b: min_gaps[b][0])
    for bid in sorted_bounds[:10]:
        gap, name = min_gaps[bid]
        print(f"  Bound {bid}: gap={gap:.8f} on {name}")
