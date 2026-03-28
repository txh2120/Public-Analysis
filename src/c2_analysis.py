#!/usr/bin/env python3
"""Cycle 2 Analysis: Hub-periphery classification of 33 remaining BHS bounds."""

import sys
sys.path.insert(0, 'src')

import numpy as np
from exhaustive_bound_search import (
    evaluate_all_bounds, compute_dv_mv, laplacian_spectral_radius,
    compute_vertex_bounds, compute_edge_bounds
)

# =====================================================================
# PART 1: Catalog all 33 remaining bound formulas
# =====================================================================
remaining_vertex = [1,4,5,6,7,8,9,10,12,14,16,18,19,20,21,22,23,24,25,26,27,30]
remaining_edge = [33,34,35,37,38,39,42,44,46,47,56]
remaining_all = remaining_vertex + remaining_edge

formulas = {
    1:  'max_v sqrt(4*d^3/m)',
    4:  'max_v 2*d^2/m',
    5:  'max_v d^2/m + m',
    6:  'max_v sqrt(m^2 + 3*d^2)',
    7:  'max_v d^2/m + d',
    8:  'max_v sqrt(d*(m + 3*d))',
    9:  'max_v (m + 3*d)/2',
    10: 'max_v sqrt(d*(d + 3*m))',
    12: 'max_v sqrt(2*m^2 + 2*d^2)',
    14: 'max_v 2*d^3/m^2',
    16: 'max_v 2*d^4/m^3',
    18: 'max_v sqrt(2*m^3/d + 2*d^2)',
    19: 'max_v (4*d^4 + 12*d*m^3)^(1/4)',
    20: 'max_v sqrt(7*d^2 + 9*m^2)/2',
    21: 'max_v sqrt(d^3/m + 3*m^2)',
    22: 'max_v (2*d^4 + 14*d^2*m^2)^(1/4)',
    23: 'max_v sqrt(d^2 + 3*d*m)',
    24: 'max_v (6*d^4 + 10*m^4)^(1/4)',
    25: 'max_v (3*d^4 + 13*d^2*m^2)^(1/4)',
    26: 'max_v sqrt(5*d^2 + 11*d*m)/2',
    27: 'max_v sqrt((3*d^2 + 5*d*m)/2)',
    30: 'max_v m^3/d^2 + d^2/m',
    33: 'max_(i~j) 2*(di+dj) - (mi+mj)',
    34: 'max_(i~j) 2*(di^2+dj^2)/(di+dj)',
    35: 'max_(i~j) 2*(di^2+dj^2)/(mi+mj)',
    37: 'max_(i~j) sqrt(2*(di^2+dj^2))',
    38: 'max_(i~j) 2 + sqrt(2*(di-1)^2 + 2*(dj-1)^2)',
    39: 'max_(i~j) 2 + sqrt(2*(di^2+dj^2) - 4*(mi+mj) + 4)',
    42: 'max_(i~j) sqrt(di^2 + dj^2 + 2*mi*mj)',
    44: 'max_(i~j) 2 + sqrt(2*((di-1)^2+(dj-1)^2 + mi*mj - di*dj))',
    46: 'max_(i~j) 2 + sqrt(2*(di^2+dj^2) - 16*di*dj/(mi+mj) + 4)',
    47: 'max_(i~j) (2*(di^2+dj^2) - (mi-mj)^2)/(di+dj)',
    56: 'max_(i~j) sqrt(2*(di^2+dj^2) + 4*mi*mj)',
}

print("PART 1: 33 Remaining Bound Formulas")
print("=" * 70)
for bid in sorted(remaining_all):
    btype = "V" if bid < 33 else "E"
    print(f"  Bound {bid:2d} [{btype}]: {formulas[bid]}")

# =====================================================================
# PART 2: Hub-periphery symbolic analysis
# Star(k): hub d=k, m=1; leaf d=1, m=k; mu = k+1
# Edge (hub~leaf): di=k, dj=1, mi=1, mj=k
# =====================================================================
print()
print("PART 2: Hub-Periphery Classification (Star(k) asymptotics)")
print("=" * 70)
print("Star(k): hub d=k, m=1; leaf d=1, m=k; mu = k+1")
print("Hub vertex: d=k, m=1; Leaf vertex: d=1, m=k")
print("Edge (hub~leaf): di=k, dj=1, mi=1, mj=k")
print()

# Vertex bounds: evaluate at BOTH hub and leaf, take max
print("--- Vertex Bounds ---")
print(f"{'Bound':>7s} | {'At hub (d=k, m=1)':>30s} | {'At leaf (d=1, m=k)':>25s} | {'Leading':>10s} | Status")
print("-" * 100)

vertex_hub_leaf = {
    # bid: (hub_expr, hub_growth, leaf_expr, leaf_growth)
    1:  ("sqrt(4*k^3)", "k^1.5", "sqrt(4/k)", "k^-0.5"),
    4:  ("2*k^2", "k^2", "2/k", "k^-1"),
    5:  ("k^2 + 1", "k^2", "1/k + k", "k"),
    6:  ("sqrt(1 + 3*k^2)", "sqrt(3)*k", "sqrt(k^2 + 3)", "k"),
    7:  ("k^2 + k", "k^2", "1/k + 1", "const"),
    8:  ("sqrt(k*(1+3k))", "sqrt(3)*k", "sqrt(1*(k+3))", "sqrt(k)"),
    9:  ("(1 + 3k)/2", "3k/2", "(k+3)/2", "k/2"),
    10: ("sqrt(k*(k+3))", "k", "sqrt(1*(1+3k))", "sqrt(3k)"),
    12: ("sqrt(2+2k^2)", "sqrt(2)*k", "sqrt(2k^2+2)", "sqrt(2)*k"),
    14: ("2*k^3", "k^3", "2/k^2", "k^-2"),
    16: ("2*k^4", "k^4", "2/k^3", "k^-3"),
    18: ("sqrt(2/k+2k^2)", "sqrt(2)*k", "sqrt(2k^3+2)", "k^1.5"),
    19: ("(4k^4+12k)^0.25", "sqrt(2)*k", "(4+12k^3)^0.25", "12^.25*k^.75"),
    20: ("sqrt(7k^2+9)/2", "sqrt(7)k/2", "sqrt(7+9k^2)/2", "3k/2"),
    21: ("sqrt(k^3+3)", "k^1.5", "sqrt(1/k+3k^2)", "sqrt(3)*k"),
    22: ("(2k^4+14k^2)^.25", "2^.25*k", "(2+14k^2)^.25", "14^.25*sqrt(k)"),
    23: ("sqrt(k^2+3k)", "k", "sqrt(1+3k)", "sqrt(3k)"),
    24: ("(6k^4+10)^.25", "6^.25*k", "(6+10k^4)^.25", "10^.25*k"),
    25: ("(3k^4+13k^2)^.25", "3^.25*k", "(3+13k^2)^.25", "13^.25*sqrt(k)"),
    26: ("sqrt(5k^2+11k)/2", "sqrt(5)k/2", "sqrt(5+11k)/2", "sqrt(11k)/2"),
    27: ("sqrt((3k^2+5k)/2)", "sqrt(3/2)*k", "sqrt((3+5k)/2)", "sqrt(5k/2)"),
    30: ("1/k^2 + k^2", "k^2", "k^3 + 1/k", "k^3"),
}

# Track which bounds are tight
tight_vertex = []
for bid in remaining_vertex:
    hub_expr, hub_growth, leaf_expr, leaf_growth = vertex_hub_leaf[bid]
    # Determine the dominant term (max of hub and leaf)
    # For star(k), mu = k+1
    # We need to check if the dominant bound value grows slower than k
    print(f"  Bound {bid:2d} | {hub_expr:>30s} | {leaf_expr:>25s} | {hub_growth:>10s} | ", end="")
    # All vertex bounds on star(k) grow at least as fast as k
    # The question is: do any grow EXACTLY as k (not faster)?
    # Those with leading term = c*k where c could be close to 1
    if hub_growth in ("k", "k/2"):
        print("POTENTIALLY TIGHT")
        tight_vertex.append(bid)
    else:
        print("ROBUST (superlinear)")

print()
print("--- Edge Bounds (at hub~leaf: di=k, dj=1, mi=1, mj=k) ---")
print(f"{'Bound':>7s} | {'Formula value':>45s} | {'Leading':>15s} | Status")
print("-" * 95)

edge_hub_leaf = {
    33: ("2*(k+1) - (1+k) = k+1", "k+1", "EXACT MATCH"),
    34: ("2*(k^2+1)/(k+1) ~ 2k-2+4/(k+1)", "2k", "ROBUST"),
    35: ("2*(k^2+1)/(1+k) ~ 2k", "2k", "ROBUST"),
    37: ("sqrt(2*(k^2+1)) ~ sqrt(2)*k", "sqrt(2)*k", "ROBUST"),
    38: ("2+sqrt(2*(k-1)^2) = 2+sqrt(2)*(k-1)", "sqrt(2)*k", "ROBUST"),
    39: ("2+sqrt(2k^2-4k+2) ~ sqrt(2)*k", "sqrt(2)*k", "ROBUST"),
    42: ("sqrt(k^2+1+2k) = sqrt((k+1)^2) = k+1", "k+1", "EXACT MATCH"),
    44: ("2+sqrt(2*(k-1)^2+2*1*k-2*k*1) = 2+sqrt(2)*(k-1)", "sqrt(2)*k", "ROBUST"),
    46: ("2+sqrt(2k^2-16k/(k+1)+6) ~ sqrt(2)*k", "sqrt(2)*k", "ROBUST"),
    47: ("(2(k^2+1)-(k-1)^2)/(k+1) = (k^2+2k+1)/(k+1) = k+1", "k+1", "EXACT MATCH"),
    56: ("sqrt(2k^2+4k+2) = sqrt(2)*(k+1)", "sqrt(2)*(k+1)", "ROBUST"),
}

tight_edge = []
for bid in remaining_edge:
    expr, growth, status = edge_hub_leaf[bid]
    print(f"  Bound {bid:2d} | {expr:>45s} | {growth:>15s} | {status}")
    if status == "EXACT MATCH":
        tight_edge.append(bid)

# Also check leaf~leaf edges on star: no such edges exist (leaves not adjacent)
# Need to check other graph families

print()
print("=" * 70)
print("CLASSIFICATION RESULT")
print("=" * 70)
print()
print("Category A: EXACT MATCH on Star(k) (bound = mu = k+1) -- PRIMARY TARGETS")
print(f"  Edge bounds: {tight_edge}")
print()
print("Category B: Linear growth but c > 1 on star -- check if tightening possible")
# bounds where hub eval grows as c*k with 1 < c < 2
cat_b = []
for bid in remaining_vertex:
    hub_expr, hub_growth, leaf_expr, leaf_growth = vertex_hub_leaf[bid]
    if hub_growth == "k":
        cat_b.append(bid)
print(f"  Vertex bounds with ~k growth at hub: {cat_b}")
print("  These have constant factor > 1 on star, but other structures may reduce it.")
print()
print("Category C: Superlinear growth (k^2, k^1.5, etc.) -- PROVABLY ROBUST vs hub-periphery")
cat_c = [b for b in remaining_vertex if b not in cat_b]
print(f"  Vertex bounds: {cat_c}")
cat_c_edge = [b for b in remaining_edge if b not in tight_edge]
print(f"  Edge bounds: {cat_c_edge}")

# =====================================================================
# PART 2b: Numerical verification for small k
# =====================================================================
print()
print("PART 2b: Numerical Verification - Star(k) for k=5,10,20,50,100")
print("=" * 70)

def make_star(k):
    """Star graph with k leaves (k+1 vertices total)."""
    n = k + 1
    A = np.zeros((n, n))
    for i in range(1, n):
        A[0, i] = 1.0
        A[i, 0] = 1.0
    return A

for k in [5, 10, 20, 50, 100]:
    A = make_star(k)
    mu, bvals, gaps = evaluate_all_bounds(A)
    print(f"\nStar({k}): mu = {mu:.4f}")
    # Show only tight bounds (gap < 1.0)
    tight = [(bid, bvals[bid], gaps[bid]) for bid in remaining_all if gaps[bid] < 1.0]
    if tight:
        for bid, bv, gap in sorted(tight, key=lambda x: x[2]):
            print(f"  Bound {bid:2d}: value={bv:.4f}, gap={gap:.6f}")
    else:
        print("  No bounds with gap < 1.0")

# =====================================================================
# PART 3: Bound 42 deep analysis
# =====================================================================
print()
print("PART 3: Bound 42 Deep Analysis")
print("=" * 70)
print()
print("Bound 42: max_(i~j) sqrt(di^2 + dj^2 + 2*mi*mj)")
print()

# On Star(k): at hub~leaf edge:
# di=k, dj=1, mi=1, mj=k
# sqrt(k^2 + 1 + 2*1*k) = sqrt(k^2+2k+1) = sqrt((k+1)^2) = k+1 = mu
print("(a) Star(k): bound = sqrt(k^2+1+2k) = k+1 = mu  [EXACT]")
print()

# On DoubleStar(a,b): center1-center2 edge
# center1: d=a+1, m=(a + d_center2)/(a+1) = (a + b+1)/(a+1)
# center2: d=b+1, m=(b + d_center1)/(b+1) = (b + a+1)/(b+1)
# For DoubleStar(1,k): center1 d=2, center2 d=k+1
# center1 neighbors: leaf1 (d=1) + center2 (d=k+1) => m_c1 = (1+k+1)/2 = (k+2)/2
# center2 neighbors: k leaves (d=1 each) + center1 (d=2) => m_c2 = (k*1+2)/(k+1) = (k+2)/(k+1)
print("(b) DoubleStar(1,k):")
print("  center1: d1=2, m1=(k+2)/2")
print("  center2: d2=k+1, m2=(k+2)/(k+1)")
print()
for k in [5, 10, 20, 50, 100, 200, 500, 1000]:
    d1 = 2.0
    d2 = k + 1.0
    m1 = (k + 2.0) / 2.0
    m2 = (k + 2.0) / (k + 1.0)
    bound42 = np.sqrt(d1**2 + d2**2 + 2*m1*m2)
    # mu of DoubleStar(1,k):
    A = np.zeros((k+3, k+3))
    # vertex 0: center1 connected to vertex 1 (leaf) and vertex 2 (center2)
    A[0,1] = A[1,0] = 1.0
    A[0,2] = A[2,0] = 1.0
    # vertex 2: center2 connected to vertices 3..k+2 (k leaves) and vertex 0
    for i in range(3, k+3):
        A[2,i] = A[i,2] = 1.0
    mu = laplacian_spectral_radius(A)
    gap = bound42 - mu
    ratio = bound42 / mu
    print(f"  k={k:4d}: bound42={bound42:.4f}, mu={mu:.6f}, gap={gap:.6f}, ratio={ratio:.6f}")

print()
print("(c) Why DoubleStar(1,k) makes Bound 42 tight:")
print("  At c1~c2 edge: sqrt(4 + (k+1)^2 + 2*(k+2)/2*(k+2)/(k+1))")
print("  = sqrt(4 + k^2+2k+1 + (k+2)^2/(k+1))")
print("  As k->inf: sqrt(k^2 + 2k + 5 + (k+4+4/k+...)) ~ sqrt(k^2 + 3k + 9) ~ k + 3/2")
print("  mu(DoubleStar(1,k)) -> k + 2 + O(1/k)  (approaches k+2)")
print("  So ratio = (k + 3/2) / (k + 2) -> 1 from below. Tight but never violates!")

# Check: also at c2~leaf edge
print()
print("  Also check c2~leaf edge:")
for k in [10, 50, 100, 500]:
    d_c2 = k + 1.0
    d_leaf = 1.0
    m_c2 = (k + 2.0) / (k + 1.0)
    m_leaf = k + 1.0  # leaf connected only to center2
    bound42_cl = np.sqrt(d_c2**2 + d_leaf**2 + 2*m_c2*m_leaf)
    # This is sqrt((k+1)^2 + 1 + 2*(k+2)/(k+1)*(k+1)) = sqrt((k+1)^2 + 1 + 2(k+2))
    # = sqrt(k^2 + 2k + 1 + 1 + 2k + 4) = sqrt(k^2 + 4k + 6)
    print(f"  k={k:4d}: c2~leaf bound42 = {bound42_cl:.4f}  (= sqrt(k^2+4k+6) ~ k+2)")

# =====================================================================
# PART 4: Propose new graph families targeting tight bounds (33, 42, 47)
# Also check: Can we make bounds 10, 23 tight?
# =====================================================================
print()
print("PART 4: New Graph Families Targeting Tight Bounds")
print("=" * 70)

# The 3 exact-match bounds on star: 33, 42, 47
# To violate, need bound < mu for some graph.
# Strategy: modify star to INCREASE mu while KEEPING bound the same or smaller.

# Family 1: Star with added pendant paths (Caterpillar)
# Attach a path of length L to each leaf of star
# This creates longer paths through hub, might increase mu

print()
print("--- Family 1: Caterpillar(k, L) ---")
print("Star(k) with a pendant path of length L at each leaf")

def make_caterpillar(k, L):
    """Star with k branches, each extended by a path of length L."""
    # Vertex 0: hub
    # Vertices 1..k: first-level (connected to hub)
    # Vertices k+1..k+k*L: pendant path vertices
    n = 1 + k + k * L
    A = np.zeros((n, n))
    for i in range(1, k+1):
        A[0, i] = A[i, 0] = 1.0
        prev = i
        for j in range(L):
            next_v = k + 1 + (i-1)*L + j
            A[prev, next_v] = A[next_v, prev] = 1.0
            prev = next_v
    return A

for k in [10, 20, 50]:
    for L in [1, 2, 3]:
        A = make_caterpillar(k, L)
        mu, bvals, gaps = evaluate_all_bounds(A)
        tight = [(bid, gaps[bid]) for bid in [33, 42, 47] if gaps[bid] < 2.0]
        print(f"  Caterpillar({k},{L}): mu={mu:.4f}, tight bounds: ", end="")
        if tight:
            for bid, gap in tight:
                print(f"B{bid}(gap={gap:.4f}) ", end="")
        else:
            print("none", end="")
        print()

# Family 2: Spider(k, L) -- hub connected to k paths of length L
print()
print("--- Family 2: Spider(k, L) ---")
print("Hub connected to k paths of length L (like caterpillar but path only, no extra leaves)")

def make_spider(k, L):
    """Hub connected to k disjoint paths of length L."""
    n = 1 + k * L
    A = np.zeros((n, n))
    for i in range(k):
        prev = 0
        for j in range(L):
            v = 1 + i * L + j
            A[prev, v] = A[v, prev] = 1.0
            prev = v
    return A

for k in [10, 20, 50]:
    for L in [2, 3, 5]:
        A = make_spider(k, L)
        mu, bvals, gaps = evaluate_all_bounds(A)
        tight = [(bid, gaps[bid]) for bid in [33, 42, 47] if gaps[bid] < 2.0]
        print(f"  Spider({k},{L}): mu={mu:.4f}, tight bounds: ", end="")
        if tight:
            for bid, gap in tight:
                print(f"B{bid}(gap={gap:.4f}) ", end="")
        else:
            print("none", end="")
        print()

# Family 3: Star + clique at hub (JellyFish)
# Hub has k leaves + connects to a clique of size c
print()
print("--- Family 3: Jellyfish(k, c) ---")
print("Hub + k leaves + complete graph K_c where hub is in the clique")

def make_jellyfish(k, c):
    """Hub vertex in a K_c clique, plus k pendant leaves."""
    n = c + k  # c clique vertices (0..c-1) + k leaves (c..c+k-1)
    A = np.zeros((n, n))
    # Clique among 0..c-1
    for i in range(c):
        for j in range(i+1, c):
            A[i,j] = A[j,i] = 1.0
    # Hub (vertex 0) connected to k leaves
    for i in range(c, c+k):
        A[0,i] = A[i,0] = 1.0
    return A

for k in [10, 20, 50]:
    for c in [2, 3, 5]:
        A = make_jellyfish(k, c)
        mu, bvals, gaps = evaluate_all_bounds(A)
        tight = [(bid, gaps[bid]) for bid in [33, 42, 47] if gaps[bid] < 2.0]
        # Also check ALL remaining bounds
        violations = [(bid, gaps[bid]) for bid in remaining_all if gaps[bid] < 0]
        print(f"  Jellyfish({k},{c}): mu={mu:.4f}, tight: ", end="")
        if tight:
            for bid, gap in tight:
                print(f"B{bid}(gap={gap:.4f}) ", end="")
        else:
            print("none", end="")
        if violations:
            print(f" VIOLATIONS: ", end="")
            for bid, gap in violations:
                print(f"B{bid}(gap={gap:.4f}) ", end="")
        print()

# Family 4: Hub with mixed leaves and triangles (BroomGraph)
# Hub + k1 leaves + k2 triangles (hub-v-w with v-w edge)
print()
print("--- Family 4: BroomTriangle(k_leaves, k_tri) ---")
print("Hub + k_leaves pendant leaves + k_tri triangles hanging off hub")

def make_broom_triangle(k_leaves, k_tri):
    """Hub with k_leaves leaves and k_tri triangles."""
    n = 1 + k_leaves + 2 * k_tri
    A = np.zeros((n, n))
    # Hub = vertex 0
    # Leaves = 1..k_leaves
    for i in range(1, k_leaves + 1):
        A[0, i] = A[i, 0] = 1.0
    # Triangles
    base = k_leaves + 1
    for t in range(k_tri):
        v1 = base + 2*t
        v2 = base + 2*t + 1
        A[0, v1] = A[v1, 0] = 1.0
        A[0, v2] = A[v2, 0] = 1.0
        A[v1, v2] = A[v2, v1] = 1.0
    return A

for kl in [0, 5, 10, 20]:
    for kt in [5, 10, 20]:
        A = make_broom_triangle(kl, kt)
        mu, bvals, gaps = evaluate_all_bounds(A)
        tight = [(bid, gaps[bid]) for bid in remaining_all if gaps[bid] < 1.0]
        violations = [(bid, gaps[bid]) for bid in remaining_all if gaps[bid] < 0]
        print(f"  BroomTri({kl},{kt}): mu={mu:.4f}, tight(<1): ", end="")
        if tight:
            for bid, gap in sorted(tight, key=lambda x: x[1]):
                print(f"B{bid}({gap:.3f}) ", end="")
        else:
            print("none", end="")
        if violations:
            print(f" **VIOLATIONS**: ", end="")
            for bid, gap in violations:
                print(f"B{bid}({gap:.3f}) ", end="")
        print()

# Family 5: Modified DoubleStar -- DoubleStar(a, b) with clique on one side
print()
print("--- Family 5: DoubleStar(a, b) systematic ---")

def make_double_star(a, b):
    """DoubleStar(a,b): two hubs connected, hub1 has a leaves, hub2 has b leaves."""
    n = 2 + a + b
    A = np.zeros((n, n))
    A[0,1] = A[1,0] = 1.0  # hub1-hub2
    for i in range(2, 2+a):
        A[0,i] = A[i,0] = 1.0
    for i in range(2+a, 2+a+b):
        A[1,i] = A[i,1] = 1.0
    return A

print("Testing DoubleStar(a,b) for various a,b:")
for a in [1, 2, 3, 5]:
    for b in [10, 20, 50, 100]:
        A = make_double_star(a, b)
        mu, bvals, gaps = evaluate_all_bounds(A)
        tight = [(bid, gaps[bid]) for bid in remaining_all if gaps[bid] < 1.0]
        violations = [(bid, gaps[bid]) for bid in remaining_all if gaps[bid] < 0]
        if tight or violations:
            print(f"  DS({a},{b}): mu={mu:.4f}, ", end="")
            if tight:
                for bid, gap in sorted(tight, key=lambda x: x[1]):
                    vstr = "**VIOL**" if gap < 0 else ""
                    print(f"B{bid}({gap:.4f}){vstr} ", end="")
            print()

# Family 6: Fan graph (hub connected to a path)
print()
print("--- Family 6: Fan(k) ---")
print("Hub connected to every vertex of P_k (path)")

def make_fan(k):
    """Fan: hub connected to all vertices of a path of k vertices."""
    n = k + 1
    A = np.zeros((n, n))
    # Hub = 0, path = 1..k
    for i in range(1, k+1):
        A[0,i] = A[i,0] = 1.0
    for i in range(1, k):
        A[i, i+1] = A[i+1, i] = 1.0
    return A

for k in [5, 10, 20, 50]:
    A = make_fan(k)
    mu, bvals, gaps = evaluate_all_bounds(A)
    tight = [(bid, gaps[bid]) for bid in remaining_all if gaps[bid] < 1.5]
    violations = [(bid, gaps[bid]) for bid in remaining_all if gaps[bid] < 0]
    print(f"  Fan({k}): mu={mu:.4f}, tight(<1.5): ", end="")
    if tight:
        for bid, gap in sorted(tight, key=lambda x: x[1]):
            print(f"B{bid}({gap:.4f}) ", end="")
    else:
        print("none", end="")
    if violations:
        print(f" **VIOLATIONS**: ", end="")
        for bid, gap in violations:
            print(f"B{bid}({gap:.4f})")
    print()

# Family 7: Subdivided star -- each edge of Star(k) subdivided s times
print()
print("--- Family 7: SubdividedStar(k, s) ---")
print("Star(k) with each hub-leaf edge subdivided s times")

def make_subdivided_star(k, s):
    """Star(k) with each edge subdivided s times."""
    n = 1 + k * (s + 1)  # hub + k paths of (s+1) vertices
    A = np.zeros((n, n))
    for i in range(k):
        prev = 0
        for j in range(s + 1):
            v = 1 + i * (s + 1) + j
            A[prev, v] = A[v, prev] = 1.0
            prev = v
    return A

for k in [10, 20, 50]:
    for s in [1, 2, 3]:
        A = make_subdivided_star(k, s)
        mu, bvals, gaps = evaluate_all_bounds(A)
        tight = [(bid, gaps[bid]) for bid in remaining_all if gaps[bid] < 1.0]
        violations = [(bid, gaps[bid]) for bid in remaining_all if gaps[bid] < 0]
        print(f"  SubStar({k},{s}): mu={mu:.4f}, tight(<1): ", end="")
        if tight:
            for bid, gap in sorted(tight, key=lambda x: x[1]):
                print(f"B{bid}({gap:.4f}) ", end="")
        else:
            print("none", end="")
        if violations:
            print(f" **VIOLATIONS**: ", end="")
            for bid, gap in violations:
                print(f"B{bid}({gap:.4f})")
        print()

# Family 8: Comet(k, p) -- Star(k) with one path of length p from hub
print()
print("--- Family 8: Comet(k, p) ---")
print("Star(k) + one path of length p from hub (asymmetric spider)")

def make_comet(k, p):
    """Star(k) with one extra path of length p from hub."""
    n = 1 + k + p
    A = np.zeros((n, n))
    # Hub = 0, leaves = 1..k
    for i in range(1, k+1):
        A[0,i] = A[i,0] = 1.0
    # Path from hub: k+1, k+2, ..., k+p
    prev = 0
    for j in range(k+1, k+1+p):
        A[prev, j] = A[j, prev] = 1.0
        prev = j
    return A

for k in [10, 20, 50]:
    for p in [2, 5, 10]:
        A = make_comet(k, p)
        mu, bvals, gaps = evaluate_all_bounds(A)
        tight = [(bid, gaps[bid]) for bid in remaining_all if gaps[bid] < 1.0]
        violations = [(bid, gaps[bid]) for bid in remaining_all if gaps[bid] < 0]
        print(f"  Comet({k},{p}): mu={mu:.4f}, tight(<1): ", end="")
        if tight:
            for bid, gap in sorted(tight, key=lambda x: x[1]):
                print(f"B{bid}({gap:.4f}) ", end="")
        else:
            print("none", end="")
        if violations:
            print(f" **VIOLATIONS**: ", end="")
            for bid, gap in violations:
                print(f"B{bid}({gap:.4f})")
        print()

# =====================================================================
# PART 5: Comprehensive test: ALL remaining bounds across ALL families
# =====================================================================
print()
print("PART 5: Comprehensive Minimum Gap Report")
print("=" * 70)
print("For each of the 33 remaining bounds, report the smallest gap found")
print("across ALL tested graph families")
print()

min_gaps = {bid: (float('inf'), "none") for bid in remaining_all}

def test_graph(name, A):
    mu, bvals, gaps = evaluate_all_bounds(A)
    for bid in remaining_all:
        if gaps[bid] < min_gaps[bid][0]:
            min_gaps[bid] = (gaps[bid], name)

# Test Stars
for k in range(3, 201):
    test_graph(f"Star({k})", make_star(k))

# Test DoubleStars
for a in range(1, 11):
    for b in range(a, 201):
        test_graph(f"DS({a},{b})", make_double_star(a, b))

# Test Caterpillars
for k in [5, 10, 20, 50, 100]:
    for L in [1, 2, 3, 5]:
        test_graph(f"Cat({k},{L})", make_caterpillar(k, L))

# Test Spiders
for k in [5, 10, 20, 50]:
    for L in [2, 3, 5, 10]:
        test_graph(f"Spider({k},{L})", make_spider(k, L))

# Test Jellyfish
for k in [5, 10, 20, 50, 100]:
    for c in [2, 3, 4, 5]:
        test_graph(f"Jelly({k},{c})", make_jellyfish(k, c))

# Test BroomTriangles
for kl in [0, 5, 10, 20, 50]:
    for kt in [5, 10, 20, 50]:
        test_graph(f"BroomTri({kl},{kt})", make_broom_triangle(kl, kt))

# Test Fans
for k in [3, 5, 10, 20, 50, 100]:
    test_graph(f"Fan({k})", make_fan(k))

# Test SubdividedStars
for k in [5, 10, 20, 50]:
    for s in [1, 2, 3]:
        test_graph(f"SubStar({k},{s})", make_subdivided_star(k, s))

# Test Comets
for k in [5, 10, 20, 50, 100]:
    for p in [2, 3, 5, 10]:
        test_graph(f"Comet({k},{p})", make_comet(k, p))

# Print results sorted by gap
print(f"{'Bound':>7s} | {'Min Gap':>12s} | {'Achieved by':>25s} | Status")
print("-" * 70)
for bid in sorted(remaining_all):
    gap, name = min_gaps[bid]
    if gap < 0:
        status = "*** VIOLATION ***"
    elif gap < 0.5:
        status = "VERY TIGHT"
    elif gap < 2.0:
        status = "TIGHT"
    else:
        status = "robust"
    print(f"  Bound {bid:2d} | {gap:12.6f} | {name:>25s} | {status}")

# Check for violations
violations = [(bid, min_gaps[bid]) for bid in remaining_all if min_gaps[bid][0] < 0]
print()
if violations:
    print("!!! NEW VIOLATIONS FOUND !!!")
    for bid, (gap, name) in violations:
        print(f"  Bound {bid}: gap={gap:.6f} on {name}")
else:
    print("No new violations found among the 33 remaining bounds.")
    print("Tightest bounds (gap < 1.0):")
    tight_final = [(bid, min_gaps[bid]) for bid in remaining_all if min_gaps[bid][0] < 1.0]
    for bid, (gap, name) in sorted(tight_final, key=lambda x: x[1][0]):
        print(f"  Bound {bid}: gap={gap:.6f} on {name}")
