#!/usr/bin/env python3
"""WA2 Exploration: Systematic graph family search for BHS bound counterexamples."""

import sys
sys.path.insert(0, 'src')
from exhaustive_bound_search import evaluate_all_bounds, ALL_BOUND_IDS
import numpy as np
import networkx as nx
from collections import defaultdict
import time

# ============================================================
# Graph Family Generators using networkx
# ============================================================

def make_adj(G):
    """Convert networkx graph to numpy adjacency matrix."""
    return nx.to_numpy_array(G, dtype=np.float64)

# 1. Windmill variations: t copies of K_k sharing one vertex
def gen_windmill(k, t):
    n = 1 + t * (k - 1)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(t):
        start = 1 + i * (k - 1)
        verts = [0] + list(range(start, start + k - 1))
        for a in range(len(verts)):
            for b in range(a + 1, len(verts)):
                G.add_edge(verts[a], verts[b])
    return G

# 2. Spider graph: central vertex with legs of varying lengths
def gen_spider(legs):
    G = nx.Graph()
    G.add_node(0)
    node_id = 1
    for leg_len in legs:
        prev = 0
        for _ in range(leg_len):
            G.add_edge(prev, node_id)
            prev = node_id
            node_id += 1
    return G

# 3. Jellyfish: cycle with pendant paths
def gen_jellyfish(cycle_size, pendant_lengths):
    G = nx.cycle_graph(cycle_size)
    node_id = cycle_size
    for i, plen in enumerate(pendant_lengths):
        if i >= cycle_size:
            break
        prev = i
        for _ in range(plen):
            G.add_edge(prev, node_id)
            prev = node_id
            node_id += 1
    return G

# 4. Broom: path + star at one end
def gen_broom(path_len, star_leaves):
    G = nx.path_graph(path_len)
    node_id = path_len
    hub = path_len - 1
    for _ in range(star_leaves):
        G.add_edge(hub, node_id)
        node_id += 1
    return G

# 5. Firecracker: stars connected in a path
def gen_firecracker(num_stars, star_size):
    G = nx.Graph()
    node_id = 0
    centers = []
    for _ in range(num_stars):
        center = node_id
        centers.append(center)
        node_id += 1
        for _ in range(star_size):
            G.add_edge(center, node_id)
            node_id += 1
    for i in range(len(centers) - 1):
        G.add_edge(centers[i], centers[i + 1])
    return G

# 6. Comet: star + path from one leaf
def gen_comet(star_leaves, path_len):
    G = nx.star_graph(star_leaves)
    node_id = star_leaves + 1
    prev = 1
    for _ in range(path_len):
        G.add_edge(prev, node_id)
        prev = node_id
        node_id += 1
    return G

# 7. Lobster: caterpillar with pendant edges on leaves
def gen_lobster(spine_len, branches_per_node, sub_branches):
    G = nx.path_graph(spine_len)
    node_id = spine_len
    for i in range(spine_len):
        for _ in range(branches_per_node):
            branch_node = node_id
            G.add_edge(i, branch_node)
            node_id += 1
            for _ in range(sub_branches):
                G.add_edge(branch_node, node_id)
                node_id += 1
    return G

# 8. Double banana: two fans sharing endpoints
def gen_double_banana(fan_size):
    G = nx.Graph()
    u, v = 0, 1
    G.add_edge(u, v)
    node_id = 2
    for _ in range(fan_size):
        G.add_edge(u, node_id)
        G.add_edge(v, node_id)
        node_id += 1
    for _ in range(fan_size):
        G.add_edge(u, node_id)
        G.add_edge(v, node_id)
        node_id += 1
    return G

# 9. Barbell: two cliques connected by a path
def gen_barbell(clique_size, path_len):
    G1 = nx.complete_graph(clique_size)
    if path_len == 0:
        G2 = nx.complete_graph(range(clique_size, 2 * clique_size))
        G = nx.compose(G1, G2)
        G.add_edge(clique_size - 1, clique_size)
    else:
        path_start = clique_size
        path_nodes = list(range(path_start, path_start + path_len))
        G2_start = path_start + path_len
        G2 = nx.complete_graph(range(G2_start, G2_start + clique_size))
        G = nx.compose(G1, G2)
        G.add_nodes_from(path_nodes)
        G.add_edge(clique_size - 1, path_nodes[0])
        for i in range(len(path_nodes) - 1):
            G.add_edge(path_nodes[i], path_nodes[i + 1])
        G.add_edge(path_nodes[-1], G2_start)
    return G

# 10. Sun graph: cycle with pendant edges on each vertex
def gen_sun(cycle_size):
    G = nx.cycle_graph(cycle_size)
    node_id = cycle_size
    for i in range(cycle_size):
        G.add_edge(i, node_id)
        node_id += 1
    return G

# 11. Mixed friendship: copies of different cliques sharing vertex 0
def gen_mixed_friendship(clique_sizes):
    G = nx.Graph()
    G.add_node(0)
    node_id = 1
    for cs in clique_sizes:
        verts = [0] + list(range(node_id, node_id + cs - 1))
        for a in range(len(verts)):
            for b in range(a + 1, len(verts)):
                G.add_edge(verts[a], verts[b])
        node_id += cs - 1
    return G

# 12. Subdivided stars
def gen_subdivided_star(num_arms, subdivisions):
    G = nx.Graph()
    G.add_node(0)
    node_id = 1
    for _ in range(num_arms):
        prev = 0
        for _ in range(subdivisions + 1):
            G.add_edge(prev, node_id)
            prev = node_id
            node_id += 1
    return G

# 13. Generalized Petersen GP(n, k)
def gen_petersen(n, k):
    G = nx.Graph()
    for i in range(n):
        G.add_edge(i, (i + 1) % n)
        G.add_edge(i, i + n)
        G.add_edge(i + n, ((i + k) % n) + n)
    return G

# 14. Helm graph
def gen_helm(n):
    G = nx.wheel_graph(n + 1)
    node_id = n + 1
    for i in range(1, n + 1):
        G.add_edge(i, node_id)
        node_id += 1
    return G

# 15. Gear graph
def gen_gear(n):
    G = nx.Graph()
    G.add_node(0)
    for i in range(1, n + 1):
        G.add_edge(0, i)
    node_id = n + 1
    for i in range(1, n + 1):
        j = (i % n) + 1
        G.add_edge(i, node_id)
        G.add_edge(node_id, j)
        node_id += 1
    return G

# 16. DoubleStar
def gen_double_star(k1, k2):
    G = nx.Graph()
    G.add_edge(0, 1)
    node_id = 2
    for _ in range(k1):
        G.add_edge(0, node_id)
        node_id += 1
    for _ in range(k2):
        G.add_edge(1, node_id)
        node_id += 1
    return G

# 17. StarOfCliques
def gen_star_of_cliques(clique_size, num_cliques):
    G = nx.Graph()
    hub = 0
    G.add_node(hub)
    node_id = 1
    for _ in range(num_cliques):
        clique_nodes = list(range(node_id, node_id + clique_size))
        for a in range(len(clique_nodes)):
            for b in range(a + 1, len(clique_nodes)):
                G.add_edge(clique_nodes[a], clique_nodes[b])
        G.add_edge(hub, clique_nodes[0])
        node_id += clique_size
    return G

# 18. Caterpillar
def gen_caterpillar(spine_len, leaves_per_node):
    G = nx.path_graph(spine_len)
    node_id = spine_len
    for i in range(spine_len):
        for _ in range(leaves_per_node):
            G.add_edge(i, node_id)
            node_id += 1
    return G

# 19. Book graph
def gen_book(k):
    G = nx.Graph()
    G.add_edge(0, 1)
    node_id = 2
    for _ in range(k):
        G.add_edge(0, node_id)
        G.add_edge(1, node_id + 1)
        G.add_edge(node_id, node_id + 1)
        node_id += 2
    return G

# 20. MultiFan
def gen_multi_fan(path_len, num_fans):
    G = nx.path_graph(path_len)
    node_id = path_len
    for _ in range(num_fans):
        for i in range(path_len):
            G.add_edge(node_id, i)
        node_id += 1
    return G

# 21. Star with clique leaves
def gen_star_with_clique_leaves(total_leaves, clique_among_leaves):
    G = nx.star_graph(total_leaves)
    for a in range(1, clique_among_leaves + 1):
        for b in range(a + 1, clique_among_leaves + 1):
            G.add_edge(a, b)
    return G

# 22. Asymmetric barbell
def gen_asymmetric_barbell(a, b, path_len):
    G1 = nx.complete_graph(a)
    if path_len == 0:
        G2 = nx.complete_graph(range(a, a + b))
        G = nx.compose(G1, G2)
        G.add_edge(a - 1, a)
    else:
        pstart = a
        pnodes = list(range(pstart, pstart + path_len))
        G2_start = pstart + path_len
        G2 = nx.complete_graph(range(G2_start, G2_start + b))
        G = nx.compose(G1, G2)
        G.add_nodes_from(pnodes)
        G.add_edge(a - 1, pnodes[0])
        for i in range(len(pnodes) - 1):
            G.add_edge(pnodes[i], pnodes[i + 1])
        G.add_edge(pnodes[-1], G2_start)
    return G

# 23. Theta graph
def gen_theta(path_lengths):
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    node_id = 2
    for pl in path_lengths:
        if pl == 0:
            G.add_edge(0, 1)
        else:
            prev = 0
            for step in range(pl):
                G.add_edge(prev, node_id)
                prev = node_id
                node_id += 1
            G.add_edge(prev, 1)
    return G

# 24. Lollipop
def gen_lollipop(clique_size, tail_len):
    G = nx.complete_graph(clique_size)
    node_id = clique_size
    prev = clique_size - 1
    for _ in range(tail_len):
        G.add_edge(prev, node_id)
        prev = node_id
        node_id += 1
    return G

# 25. Tadpole
def gen_tadpole(cycle_size, tail_len):
    G = nx.cycle_graph(cycle_size)
    node_id = cycle_size
    prev = 0
    for _ in range(tail_len):
        G.add_edge(prev, node_id)
        prev = node_id
        node_id += 1
    return G

# ============================================================
# Main exploration
# ============================================================

known_violations = {11, 13, 45}
violations = []
near_misses = []
total_graphs = 0
families_tested = {}

def test_graph(G, family_name, params_str):
    global total_graphs
    n = G.number_of_nodes()
    if n < 3 or n > 150:
        return
    if not nx.is_connected(G):
        return
    total_graphs += 1
    A = make_adj(G)
    mu, bound_vals, gaps = evaluate_all_bounds(A)
    e = G.number_of_edges()
    for bid in ALL_BOUND_IDS:
        gap = gaps[bid]
        bv = bound_vals[bid]
        if gap < -1e-10:
            violations.append((bid, family_name, params_str, n, e, mu, bv, gap))
        elif bv > 0 and gap / bv < 0.05:
            near_misses.append((bid, family_name, params_str, n, e, mu, bv, gap))

start_time = time.time()

# 1. Windmill
print("Testing Windmill...")
for k in range(3, 12):
    for t in range(2, 30):
        n = 1 + t * (k - 1)
        if n > 150: break
        G = gen_windmill(k, t)
        test_graph(G, "Windmill", f"k={k},t={t}")
families_tested["Windmill"] = "k=3..11, t=2..29 (n<=150)"

# 2. Spider
print("Testing Spider...")
for num_legs in range(2, 10):
    for max_leg_len in range(1, 15):
        legs = [max_leg_len] * num_legs
        if sum(legs) + 1 <= 150:
            G = gen_spider(legs)
            test_graph(G, "Spider", f"legs={num_legs}x{max_leg_len}")
        legs2 = [1] * (num_legs - 1) + [max_leg_len]
        if sum(legs2) + 1 <= 150:
            G = gen_spider(legs2)
            test_graph(G, "Spider-asym", f"legs={num_legs-1}x1+1x{max_leg_len}")
        if num_legs >= 3:
            legs3 = [2] * (num_legs - 1) + [max_leg_len * 3]
            if sum(legs3) + 1 <= 150:
                G = gen_spider(legs3)
                test_graph(G, "Spider-vasym", f"legs={num_legs-1}x2+1x{max_leg_len*3}")
families_tested["Spider"] = "2-9 legs, lengths 1-14 (uniform, asym, v-asym)"

# 3. Jellyfish
print("Testing Jellyfish...")
for cs in range(3, 20):
    for plen in range(1, 10):
        pendants = [plen] * cs
        if cs + sum(pendants) <= 150:
            G = gen_jellyfish(cs, pendants)
            test_graph(G, "Jellyfish", f"cycle={cs},pendants={cs}x{plen}")
        pendants2 = [plen] * (cs // 2) + [0] * (cs - cs // 2)
        if cs + sum(pendants2) <= 150:
            G = gen_jellyfish(cs, pendants2)
            test_graph(G, "Jellyfish-half", f"cycle={cs},half_p={plen}")
families_tested["Jellyfish"] = "cycle=3..19, pendant 1..9 (full+half)"

# 4. Broom
print("Testing Broom...")
for path_len in range(2, 30):
    for star_leaves in range(2, 50):
        if path_len + star_leaves <= 150:
            G = gen_broom(path_len, star_leaves)
            test_graph(G, "Broom", f"path={path_len},leaves={star_leaves}")
families_tested["Broom"] = "path=2..29, leaves=2..49 (n<=150)"

# 5. Firecracker
print("Testing Firecracker...")
for num_stars in range(2, 20):
    for star_size in range(2, 20):
        if num_stars * (star_size + 1) <= 150:
            G = gen_firecracker(num_stars, star_size)
            test_graph(G, "Firecracker", f"stars={num_stars},size={star_size}")
families_tested["Firecracker"] = "stars=2..19, size=2..19 (n<=150)"

# 6. Comet
print("Testing Comet...")
for star_leaves in range(2, 50):
    for path_len in range(1, 50):
        if star_leaves + path_len + 1 <= 150:
            G = gen_comet(star_leaves, path_len)
            test_graph(G, "Comet", f"leaves={star_leaves},path={path_len}")
families_tested["Comet"] = "leaves=2..49, path=1..49 (n<=150)"

# 7. Lobster
print("Testing Lobster...")
for spine in range(2, 15):
    for branches in range(1, 5):
        for sub_br in range(1, 5):
            n = spine + spine * branches * (1 + sub_br)
            if n <= 150:
                G = gen_lobster(spine, branches, sub_br)
                test_graph(G, "Lobster", f"spine={spine},br={branches},sub={sub_br}")
families_tested["Lobster"] = "spine=2..14, branches=1..4, sub=1..4 (n<=150)"

# 8. Double banana
print("Testing DoubleBanana...")
for fan_size in range(2, 40):
    if 2 + 2 * fan_size <= 150:
        G = gen_double_banana(fan_size)
        test_graph(G, "DoubleBanana", f"fan={fan_size}")
families_tested["DoubleBanana"] = "fan=2..39"

# 9. Barbell
print("Testing Barbell...")
for clique in range(3, 20):
    for path in range(0, 30):
        if 2 * clique + path <= 150:
            G = gen_barbell(clique, path)
            test_graph(G, "Barbell", f"clique={clique},path={path}")
families_tested["Barbell"] = "clique=3..19, path=0..29 (n<=150)"

# 10. Sun
print("Testing Sun...")
for cs in range(3, 50):
    if 2 * cs <= 150:
        G = gen_sun(cs)
        test_graph(G, "Sun", f"cycle={cs}")
families_tested["Sun"] = "cycle=3..49"

# 11. Mixed friendship
print("Testing MixedFriendship...")
for k1 in range(3, 8):
    for k2 in range(k1, 10):
        for n1 in range(1, 8):
            for n2 in range(1, 8):
                total_n = 1 + n1 * (k1 - 1) + n2 * (k2 - 1)
                if total_n <= 100:
                    sizes = [k1] * n1 + [k2] * n2
                    G = gen_mixed_friendship(sizes)
                    test_graph(G, "MixedFriendship", f"k1={k1}x{n1},k2={k2}x{n2}")
families_tested["MixedFriendship"] = "k1=3..7x1..7, k2=k1..9x1..7 (n<=100)"

# 12. Subdivided stars
print("Testing SubdividedStar...")
for arms in range(2, 20):
    for subdiv in range(1, 10):
        n = 1 + arms * (subdiv + 1)
        if n <= 150:
            G = gen_subdivided_star(arms, subdiv)
            test_graph(G, "SubdividedStar", f"arms={arms},subdiv={subdiv}")
families_tested["SubdividedStar"] = "arms=2..19, subdiv=1..9 (n<=150)"

# 13. Generalized Petersen
print("Testing Petersen...")
for n in range(5, 50):
    for k in range(1, n // 2 + 1):
        if 2 * n <= 150:
            try:
                G = gen_petersen(n, k)
                if nx.is_connected(G):
                    test_graph(G, "Petersen", f"n={n},k={k}")
            except:
                pass
families_tested["Petersen"] = "n=5..49, k=1..n/2 (2n<=150)"

# 14. Helm
print("Testing Helm...")
for n in range(3, 50):
    if 2 * n + 1 <= 150:
        G = gen_helm(n)
        test_graph(G, "Helm", f"n={n}")
families_tested["Helm"] = "n=3..49"

# 15. Gear
print("Testing Gear...")
for n in range(3, 50):
    if 2 * n + 1 <= 150:
        G = gen_gear(n)
        test_graph(G, "Gear", f"n={n}")
families_tested["Gear"] = "n=3..49"

# 16. DoubleStar (extended range)
print("Testing DoubleStar...")
for k1 in range(1, 80):
    for k2 in range(k1, 80):
        if k1 + k2 + 2 <= 150:
            G = gen_double_star(k1, k2)
            test_graph(G, "DoubleStar", f"k1={k1},k2={k2}")
families_tested["DoubleStar"] = "k1=1..79, k2=k1..79 (n<=150)"

# 17. StarOfCliques
print("Testing StarOfCliques...")
for cs in range(3, 12):
    for nc in range(2, 30):
        if 1 + cs * nc <= 150:
            G = gen_star_of_cliques(cs, nc)
            test_graph(G, "StarOfCliques", f"clique={cs},num={nc}")
families_tested["StarOfCliques"] = "clique=3..11, num=2..29 (n<=150)"

# 18. Caterpillar
print("Testing Caterpillar...")
for spine in range(2, 30):
    for leaves in range(1, 15):
        if spine * (1 + leaves) <= 150:
            G = gen_caterpillar(spine, leaves)
            test_graph(G, "Caterpillar", f"spine={spine},leaves={leaves}")
families_tested["Caterpillar"] = "spine=2..29, leaves=1..14 (n<=150)"

# 19. Book
print("Testing Book...")
for k in range(2, 50):
    if 2 + 2 * k <= 150:
        G = gen_book(k)
        test_graph(G, "Book", f"k={k}")
families_tested["Book"] = "k=2..49"

# 20. MultiFan
print("Testing MultiFan...")
for path_len in range(3, 30):
    for num_fans in range(1, 10):
        if path_len + num_fans <= 150:
            G = gen_multi_fan(path_len, num_fans)
            test_graph(G, "MultiFan", f"path={path_len},fans={num_fans}")
families_tested["MultiFan"] = "path=3..29, fans=1..9 (n<=150)"

# 21. StarWithCliqueLeaves
print("Testing StarWithCliqueLeaves...")
for total in range(4, 40):
    for clique_k in range(2, min(total, 15)):
        G = gen_star_with_clique_leaves(total, clique_k)
        if G.number_of_nodes() <= 150:
            test_graph(G, "StarCliqueLeaves", f"total={total},clique={clique_k}")
families_tested["StarCliqueLeaves"] = "total=4..39, clique=2..14"

# 22. Asymmetric barbell
print("Testing AsymmetricBarbell...")
for a in range(3, 15):
    for b in range(3, 15):
        if a == b: continue
        for p in range(0, 20):
            if a + b + p <= 150:
                G = gen_asymmetric_barbell(a, b, p)
                test_graph(G, "AsymBarbell", f"a={a},b={b},path={p}")
families_tested["AsymBarbell"] = "a=3..14, b=3..14 (a!=b), path=0..19"

# 23. Theta graph
print("Testing Theta...")
for num_paths in range(2, 8):
    for max_len in range(1, 20):
        paths = [max_len] * num_paths
        if 2 + sum(paths) <= 150:
            G = gen_theta(paths)
            test_graph(G, "Theta", f"paths={num_paths}x{max_len}")
        if num_paths >= 2:
            paths2 = [1] + [max_len] * (num_paths - 1)
            if 2 + sum(paths2) <= 150:
                G = gen_theta(paths2)
                test_graph(G, "Theta-asym", f"1x1+{num_paths-1}x{max_len}")
families_tested["Theta"] = "2-7 paths, lengths 1..19 (uniform+asym)"

# 24. Lollipop
print("Testing Lollipop...")
for cs in range(3, 20):
    for tail in range(1, 30):
        if cs + tail <= 150:
            G = gen_lollipop(cs, tail)
            test_graph(G, "Lollipop", f"clique={cs},tail={tail}")
families_tested["Lollipop"] = "clique=3..19, tail=1..29 (n<=150)"

# 25. Tadpole
print("Testing Tadpole...")
for cs in range(3, 30):
    for tail in range(1, 30):
        if cs + tail <= 150:
            G = gen_tadpole(cs, tail)
            test_graph(G, "Tadpole", f"cycle={cs},tail={tail}")
families_tested["Tadpole"] = "cycle=3..29, tail=1..29 (n<=150)"

elapsed = time.time() - start_time
print(f"\n{'='*70}")
print(f"EXPLORATION COMPLETE: {total_graphs} graphs tested in {elapsed:.1f}s")
print(f"{'='*70}")

# ============================================================
# Report violations
# ============================================================
print(f"\n### ALL VIOLATIONS ({len(violations)} total):")
print(f"{'Bound':>5} | {'Family':<25} | {'Params':<35} | {'n':>4} | {'edges':>5} | {'mu':>10} | {'bound':>10} | {'gap':>10}")
print("-" * 115)
for bid, fam, params, n, e, mu, bv, gap in sorted(violations, key=lambda x: (x[0], x[7])):
    marker = "" if bid in known_violations else " ** NEW **"
    print(f"{bid:5d} | {fam:<25} | {params:<35} | {n:4d} | {e:5d} | {mu:10.6f} | {bv:10.6f} | {gap:10.6f}{marker}")

# Report new violations
new_violations = [v for v in violations if v[0] not in known_violations]
print(f"\n### NEW VIOLATIONS (not in bounds 11, 13, 45): {len(new_violations)}")
if new_violations:
    new_bound_ids = sorted(set(v[0] for v in new_violations))
    print(f"New violated bound IDs: {new_bound_ids}")
    print(f"{'Bound':>5} | {'Family':<25} | {'Params':<35} | {'n':>4} | {'edges':>5} | {'mu':>10} | {'bound':>10} | {'gap':>10}")
    print("-" * 115)
    for bid, fam, params, n, e, mu, bv, gap in sorted(new_violations, key=lambda x: (x[0], x[7])):
        print(f"{bid:5d} | {fam:<25} | {params:<35} | {n:4d} | {e:5d} | {mu:10.6f} | {bv:10.6f} | {gap:10.6f}")
else:
    print("None found.")

# Near-misses
print(f"\n### NEAR-MISSES (gap/bound < 5%, best per bound):")
bound_near = defaultdict(list)
for bid, fam, params, n, e, mu, bv, gap in near_misses:
    if bv > 0:
        ratio = gap / bv
        bound_near[bid].append((ratio, fam, params, n, e, mu, bv, gap))

print(f"{'Bound':>5} | {'gap%':>7} | {'Family':<25} | {'Params':<30} | {'n':>4} | {'mu':>10} | {'bound':>10} | {'gap':>10}")
print("-" * 125)
for bid in sorted(bound_near.keys()):
    items = sorted(bound_near[bid], key=lambda x: x[0])[:3]
    for ratio, fam, params, n, e, mu, bv, gap in items:
        is_violated_bid = bid in set(v[0] for v in violations)
        marker = " (VIOLATED elsewhere)" if is_violated_bid else ""
        print(f"{bid:5d} | {ratio*100:6.3f}% | {fam:<25} | {params:<30} | {n:4d} | {mu:10.6f} | {bv:10.6f} | {gap:10.6f}{marker}")

# Summary
all_violated_bounds = sorted(set(v[0] for v in violations))
new_violated_bounds = sorted(set(v[0] for v in violations if v[0] not in known_violations))
print(f"\n### SUMMARY")
print(f"Total families tested: {len(families_tested)}")
print(f"Total graphs tested: {total_graphs}")
print(f"Total violations: {len(violations)}")
print(f"All violated bounds: {all_violated_bounds}")
print(f"Known violations reproduced: {sorted(set(all_violated_bounds) & known_violations)}")
print(f"NEW bounds violated: {new_violated_bounds}")
print(f"Time: {elapsed:.1f}s")

# Best counterexample per violated bound
print(f"\n### BEST COUNTEREXAMPLE PER BOUND:")
best_per_bound = {}
for bid, fam, params, n, e, mu, bv, gap in violations:
    if bid not in best_per_bound or gap < best_per_bound[bid][7]:
        best_per_bound[bid] = (bid, fam, params, n, e, mu, bv, gap)
for bid in sorted(best_per_bound.keys()):
    _, fam, params, n, e, mu, bv, gap = best_per_bound[bid]
    marker = " ** NEW **" if bid not in known_violations else ""
    print(f"  Bound {bid:2d}: {fam} ({params}), n={n}, mu={mu:.6f}, bound={bv:.6f}, gap={gap:.6f}{marker}")
