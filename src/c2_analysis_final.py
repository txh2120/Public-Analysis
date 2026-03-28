#!/usr/bin/env python3
"""Cycle 2 Final Analysis: Non-trivial tightness ranking per bound."""

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
# Non-trivial tightness: exclude regular graphs (d=m for all vertices)
# =====================================================================

def make_star(k):
    n = k + 1
    A = np.zeros((n, n))
    for i in range(1, n):
        A[0, i] = A[i, 0] = 1.0
    return A

def make_double_star(a, b):
    n = 2 + a + b
    A = np.zeros((n, n))
    A[0,1] = A[1,0] = 1.0
    for i in range(2, 2+a):
        A[0,i] = A[i,0] = 1.0
    for i in range(2+a, 2+a+b):
        A[1,i] = A[i,1] = 1.0
    return A

def make_complete_bipartite(a, b):
    n = a + b
    A = np.zeros((n, n))
    for i in range(a):
        for j in range(a, a+b):
            A[i,j] = A[j,i] = 1.0
    return A

def make_friendship(k):
    n = 1 + 2*k
    A = np.zeros((n, n))
    for i in range(k):
        v1 = 1 + 2*i
        v2 = 2 + 2*i
        A[0, v1] = A[v1, 0] = 1.0
        A[0, v2] = A[v2, 0] = 1.0
        A[v1, v2] = A[v2, v1] = 1.0
    return A

def make_sun(k):
    n = 2 * k
    A = np.zeros((n, n))
    for i in range(k):
        A[i, (i+1) % k] = A[(i+1) % k, i] = 1.0
    for i in range(k):
        A[i, k+i] = A[k+i, i] = 1.0
    return A

def is_regular(A):
    dv = A.sum(axis=1)
    return np.allclose(dv, dv[0])

# Track per-bound: top 3 tightest non-regular graphs
top_tight = {bid: [] for bid in remaining_all}  # list of (gap, name)

def test_graph(name, A):
    if A.shape[0] < 2:
        return
    if is_regular(A):
        return  # skip regular graphs
    mu, bvals, gaps = evaluate_all_bounds(A)
    for bid in remaining_all:
        g = gaps[bid]
        if g > -1e-10:  # non-violation (allow tiny floating point)
            g = max(g, 0.0)
            top_tight[bid].append((g, name))
            # Keep only top 5
            top_tight[bid].sort(key=lambda x: x[0])
            if len(top_tight[bid]) > 5:
                top_tight[bid] = top_tight[bid][:5]

# Generate all families
print("Generating graphs...")

# Stars k=2..500
for k in range(2, 501):
    test_graph(f"Star({k})", make_star(k))

# Double stars
for a in range(1, 20):
    for b in range(a, min(300, 20*a)):
        test_graph(f"DS({a},{b})", make_double_star(a, b))

# K_{a,b} non-square
for a in range(1, 15):
    for b in range(a+1, min(200, 15*a)):
        test_graph(f"K{a},{b}", make_complete_bipartite(a, b))

# Suns
for k in range(3, 100):
    test_graph(f"Sun({k})", make_sun(k))

# Friendships
for k in range(2, 200):
    test_graph(f"Friend({k})", make_friendship(k))

# Paths
for n in range(3, 50):
    A = np.zeros((n,n))
    for i in range(n-1):
        A[i,i+1] = A[i+1,i] = 1.0
    test_graph(f"P{n}", A)

# Wheels
for k in range(3, 200):
    n = k + 1
    A = np.zeros((n, n))
    for i in range(1, n):
        A[0,i] = A[i,0] = 1.0
    for i in range(1, k):
        A[i,i+1] = A[i+1,i] = 1.0
    A[1,k] = A[k,1] = 1.0
    test_graph(f"Wheel({k})", A)

# Fans
for k in range(3, 200):
    n = k + 1
    A = np.zeros((n, n))
    for i in range(1, n):
        A[0,i] = A[i,0] = 1.0
    for i in range(1, k):
        A[i,i+1] = A[i+1,i] = 1.0
    test_graph(f"Fan({k})", A)

# HubWithCliques
for k in [3,5,10,20,50,100]:
    for c in [2,3,4,5]:
        n = 1 + k * c
        A = np.zeros((n, n))
        for i in range(k):
            base = 1 + i * c
            for a in range(c):
                for b in range(a+1, c):
                    A[base+a, base+b] = A[base+b, base+a] = 1.0
            A[0, base] = A[base, 0] = 1.0
        test_graph(f"HWC({k},{c})", A)

# BroomTriangles
for kl in [0, 5, 10, 20, 50]:
    for kt in [5, 10, 20, 50]:
        n = 1 + kl + 2 * kt
        A = np.zeros((n, n))
        for i in range(1, kl + 1):
            A[0, i] = A[i, 0] = 1.0
        base = kl + 1
        for t in range(kt):
            v1 = base + 2*t
            v2 = base + 2*t + 1
            A[0, v1] = A[v1, 0] = 1.0
            A[0, v2] = A[v2, 0] = 1.0
            A[v1, v2] = A[v2, v1] = 1.0
        test_graph(f"BroomTri({kl},{kt})", A)

# Kites
for c in range(3, 20):
    for t in range(1, 15):
        n = c + t
        A = np.zeros((n, n))
        for i in range(c):
            for j in range(i+1, c):
                A[i,j] = A[j,i] = 1.0
        prev = c - 1
        for i in range(c, c + t):
            A[prev, i] = A[i, prev] = 1.0
            prev = i
        test_graph(f"Kite({c},{t})", A)

# Barbells
for k1 in range(3, 12):
    for k2 in range(k1+1, 15):  # asymmetric only (non-regular)
        for p in range(0, 5):
            n = k1 + k2 + p
            A = np.zeros((n, n))
            for i in range(k1):
                for j in range(i+1, k1):
                    A[i,j] = A[j,i] = 1.0
            prev = k1 - 1
            for i in range(k1, k1 + p):
                A[prev, i] = A[i, prev] = 1.0
                prev = i
            base2 = k1 + p
            A[prev, base2] = A[base2, prev] = 1.0
            for i in range(base2, base2 + k2):
                for j in range(i+1, base2 + k2):
                    A[i,j] = A[j,i] = 1.0
            test_graph(f"Barbell({k1},{k2},{p})", A)

print("Done generating.")
print()

# =====================================================================
# Print results: per-bound tightness ranking
# =====================================================================
print("PER-BOUND TIGHTNESS RANKING (non-regular graphs only)")
print("=" * 80)
print()

# Group bounds by their tightest gap
bound_classes = {
    "EXACT (gap=0 on non-regular)": [],
    "ASYMPTOTICALLY TIGHT (gap->0)": [],
    "BOUNDED GAP (gap > c > 0)": [],
}

for bid in sorted(remaining_all):
    entries = top_tight[bid]
    if not entries:
        print(f"Bound {bid:2d}: No data")
        continue

    min_gap, min_graph = entries[0]

    btype = "V" if bid < 33 else "E"
    print(f"Bound {bid:2d} [{btype}]: min gap = {min_gap:.6f} on {min_graph}")

    # Show top 3
    for gap, name in entries[:3]:
        print(f"    gap={gap:.6f}  {name}")

    if min_gap < 1e-10:
        bound_classes["EXACT (gap=0 on non-regular)"].append(bid)
    elif min_gap < 0.02:
        bound_classes["ASYMPTOTICALLY TIGHT (gap->0)"].append(bid)
    else:
        bound_classes["BOUNDED GAP (gap > c > 0)"].append(bid)

print()
print("SUMMARY CLASSIFICATION:")
for cls, bids in bound_classes.items():
    print(f"  {cls}: {bids}")

# =====================================================================
# Bound 42 detailed convergence
# =====================================================================
print()
print("BOUND 42 CONVERGENCE RATE")
print("=" * 70)
print("DS(1,k): gap ~ 1/(2k) as k->inf")
for k in [10, 50, 100, 500, 1000, 5000]:
    A = make_double_star(1, k)
    mu, bvals, gaps = evaluate_all_bounds(A)
    predicted = 0.5 / k  # rough prediction
    print(f"  DS(1,{k:5d}): gap={gaps[42]:.8f}, 1/(2k)={predicted:.8f}, ratio={gaps[42]/predicted:.4f}")

print()
print("Friend(k): gap ~ 1/(6k) as k->inf")
for k in [10, 50, 100, 500, 1000]:
    A = make_friendship(k)
    mu, bvals, gaps = evaluate_all_bounds(A)
    predicted = 1.0 / (6*k)
    print(f"  Friend({k:4d}): gap={gaps[42]:.8f}, 1/(6k)={predicted:.8f}")

# =====================================================================
# Bound 33 and 47 on K_{a,b}: all exact
# Check tightest NON-EXACT case for 33, 47
# =====================================================================
print()
print("BOUNDS 33, 47: Tightest non-exact (non-regular, non-bipartite)")
print("=" * 70)
for bid in [33, 47]:
    entries = top_tight[bid]
    # Filter out exact matches
    non_exact = [(g, n) for g, n in entries if g > 1e-6]
    if non_exact:
        print(f"  Bound {bid}: tightest non-exact: gap={non_exact[0][0]:.6f} on {non_exact[0][1]}")
    else:
        print(f"  Bound {bid}: all top entries are exact (K_{{a,b}} family)")
