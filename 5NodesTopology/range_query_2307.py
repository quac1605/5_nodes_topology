#!/usr/bin/env python3
import os
import re
import subprocess
import bisect
import random
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
P = 6   # 6-bit grid per dimension (64 cells)
N = 5   # dims: X, Y, Z, RAM, vCores

CONTAINER_NAME     = "clab-century-serf1"
CONTAINER_LOG_PATH = "/opt/serfapp/nodes_log.txt"
HOST_LOG_DIR       = "./dist"
HOST_LOG_PATH      = os.path.join(HOST_LOG_DIR, "nodes_log.txt")

# query thresholds
THR_RTT, THR_RAM, THR_CORES = 100.0, 8, 8  # ≤100ms, ≥8GB, ≥8 cores

# ─────────────────────────────────────────────────────────────────────────────
# Randomized node resources (RAM/vCores) with sensible ranges
# Ensures a mix of nodes above and below thresholds, with optional reproducible seed.
# ─────────────────────────────────────────────────────────────────────────────
SEED = int(os.getenv("NODE_RESOURCE_SEED", "42"))
random.seed(SEED)

RAM_CHOICES   = [4, 8, 12, 16, 24, 32]          # GB
CORE_CHOICES  = [4, 6, 8, 12, 16, 24, 32]       # vCores

# weights must match the lengths above:
RAM_WEIGHTS   = [1, 2, 3, 3, 2, 1]
CORE_WEIGHTS  = [1, 2, 3, 3, 2, 2, 1]
assert len(RAM_CHOICES)  == len(RAM_WEIGHTS)
assert len(CORE_CHOICES) == len(CORE_WEIGHTS)

def make_random_node_resources(num_nodes=26, thr_ram=THR_RAM, thr_cores=THR_CORES):
    nodes = [f"clab-century-serf{i}" for i in range(1, num_nodes + 1)]
    res = {}

    # Initial random draw (slight bias toward mid/high options)
    for n in nodes:
        ram   = random.choices(RAM_CHOICES,  weights=RAM_WEIGHTS,  k=1)[0]
        cores = random.choices(CORE_CHOICES, weights=CORE_WEIGHTS, k=1)[0]
        res[n] = (ram, cores)

    # Ensure at least ~25% nodes meet BOTH thresholds
    hi_nodes = [n for n,(r,c) in res.items() if r >= thr_ram and c >= thr_cores]
    need_hi  = max(1, num_nodes // 4)
    if len(hi_nodes) < need_hi:
        to_upgrade = [n for n in nodes if n not in hi_nodes][:need_hi - len(hi_nodes)]
        for n in to_upgrade:
            res[n] = (max(thr_ram, 16), max(thr_cores, 16))

    # Ensure at least ~15% nodes fall below (so filters have negatives)
    lo_nodes = [n for n,(r,c) in res.items() if r < thr_ram or c < thr_cores]
    need_lo  = max(1, num_nodes // 6)
    if len(lo_nodes) < need_lo:
        candidates = [n for n in nodes if n not in lo_nodes]
        for n in candidates[:need_lo - len(lo_nodes)]:
            res[n] = (min(thr_ram - 1, 4), min(thr_cores - 1, 4))

    return res

# Build randomized resources for the cluster nodes
node_resources = make_random_node_resources(num_nodes=26)

# ─────────────────────────────────────────────────────────────────────────────
# 1) Copy & Parse the Serf Log
# ─────────────────────────────────────────────────────────────────────────────
def copy_log():
    os.makedirs(HOST_LOG_DIR, exist_ok=True)
    subprocess.run(
        ["docker", "cp",
         f"{CONTAINER_NAME}:{CONTAINER_LOG_PATH}",
         HOST_LOG_PATH],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    print(f"✅ Copied log to {HOST_LOG_PATH}")

def parse_log():
    coords, rtts = {}, {}
    section = None
    if not os.path.isfile(HOST_LOG_PATH):
        raise FileNotFoundError(f"{HOST_LOG_PATH} not found")
    with open(HOST_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if "[COORDINATES]" in line:
                section = "coord"; continue
            if "[RTT]" in line:
                section = "rtt"; continue
            if "Node:" not in line:
                continue
            m = re.match(r".*Node:\s*(\S+)\s*=>\s*(.*)", line)
            if not m:
                continue
            node, rest = m.groups()
            if section == "coord":
                cm = re.search(r"X:\s*([-\d.]+)\s*Y:\s*([-\d.]+)\s*Z:\s*([-\d.]+)", rest)
                if cm:
                    x,y,z = map(float, cm.groups())
                    if node not in node_resources:
                        # If a node appears that's not in our randomized set, give it a default
                        node_resources[node] = (THR_RAM, THR_CORES)
                    ram, cores = node_resources[node]
                    coords[node] = (x,y,z,ram,cores)
            else:  # RTT section
                rm = re.search(r"RTT:\s*([\d.]+)\s*ms", rest)
                if rm:
                    rtts[node] = float(rm.group(1))
    # ensure current node has an RTT entry
    rtts[CONTAINER_NAME] = 0.0

    print(f"✅ Parsed {len(coords)} nodes with coordinates and RTTs")
    return coords, rtts

# ─────────────────────────────────────────────────────────────────────────────
# 2) Normalize & Compute 5-D Hilbert Index
# ─────────────────────────────────────────────────────────────────────────────
def normalize_and_index(coords, bits):
    arr  = np.array(list(coords.values()), dtype=float)  # shape = (n,5)
    minv = arr.min(axis=0)
    maxv = arr.max(axis=0)
    diff = maxv - minv

    scale = np.empty_like(diff)
    nz    = diff != 0
    scale[nz] = (2**bits - 1) / diff[nz]
    scale[~nz] = 1.0

    norm = (arr - minv) * scale
    norm[:, ~nz] = 0

    hc       = HilbertCurve(bits, N)
    hilb_map = {}
    for i, node in enumerate(coords):
        pt = [int(round(v)) for v in norm[i]]
        hilb_map[node] = hc.distance_from_point(pt)

    sorted_arr = sorted((h, n) for n, h in hilb_map.items())
    hil_list   = [h for h, _ in sorted_arr]

    print("✅ Computed Hilbert indices at P =", bits)
    return hilb_map, sorted_arr, hil_list, minv, maxv, scale

# ─────────────────────────────────────────────────────────────────────────────
# 3) Pure Hilbert Range Query (±1-cell padding)
# ─────────────────────────────────────────────────────────────────────────────
def decompose_hilbert_ranges(hc, min_pt, max_pt):
    N, size = hc.n, 2**hc.p
    total   = size**N
    intervals = []

    def recurse(origin, span, h_lo, h_hi):
        # If this cell is fully outside the query box, skip it
        for d in range(N):
            if origin[d] + span <= min_pt[d] or origin[d] >= max_pt[d]:
                return
        # If fully inside, add its entire Hilbert index range
        if all(origin[d] >= min_pt[d] and origin[d] + span <= max_pt[d] for d in range(N)):
            intervals.append((h_lo, h_hi))
            return
        # If we're at a single point, add that single index
        if span == 1:
            intervals.append((h_lo, h_lo + 1))
            return
        # Otherwise subdivide into 2^N children
        half  = span // 2
        block = half**N
        for i in range(1 << N):
            offs = [(i >> b) & 1 for b in range(N)]
            child_origin = [origin[d] + offs[d]*half for d in range(N)]
            c_lo = h_lo + i * block
            c_hi = c_lo + block
            recurse(child_origin, half, c_lo, c_hi)

    recurse([0]*N, size, 0, total)
    intervals.sort()
    # merge adjacent/overlapping intervals
    merged = []
    for a,b in intervals:
        if not merged or a > merged[-1][1]:
            merged.append([a,b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    return [(a,b) for a,b in merged]

def pure_hilbert_range_query(
    coords, rtts,
    hilb_map, sorted_arr, hil_list,
    minv, maxv, scale,
    thr_rtt, thr_ram, thr_cores
):
    # Build real-world box
    x0,y0,z0,_,_ = coords[CONTAINER_NAME]
    raw_min = (x0-thr_rtt, y0-thr_rtt, z0-thr_rtt, thr_ram,   thr_cores)
    raw_max = (x0+thr_rtt, y0+thr_rtt, z0+thr_rtt, maxv[3],   maxv[4])

    # Map to integer grid, pad ±1 cell
    pad = 1
    min_pt, max_pt = [], []
    for d in range(N):
        lo = int(np.floor((raw_min[d] - minv[d]) * scale[d])) - pad
        hi = int(np.ceil ((raw_max[d] - minv[d]) * scale[d])) + pad
        lo = max(0, min(lo, 2**P))
        hi = max(0, min(hi, 2**P))
        min_pt.append(lo)
        max_pt.append(hi)
    min_pt, max_pt = tuple(min_pt), tuple(max_pt)

    print(f"\n▶ Query grid box (cells):\n   min_pt={min_pt}\n   max_pt={max_pt}")

    # Decompose into intervals
    hc = HilbertCurve(P, N)
    intervals = decompose_hilbert_ranges(hc, min_pt, max_pt)
    print(f"▶ Decomposed into {len(intervals)} intervals")

    # Bisect each interval to collect candidates
    candidates = set()
    for lo,hi in intervals:
        i0 = bisect.bisect_left(hil_list, lo)
        i1 = bisect.bisect_right(hil_list, hi)
        for idx in range(i0, i1):
            candidates.add(sorted_arr[idx][1])
    print(f"▶ Found {len(candidates)} Hilbert‐interval candidates:\n   {sorted(candidates)}")

    # Final exact filter on raw RTT/RAM/cores
    hits = [
        n for n in candidates
        if rtts.get(n, float('inf')) <= thr_rtt
        and coords[n][3] >= thr_ram
        and coords[n][4] >= thr_cores
    ]
    print(f"▶ After exact filtering → {len(hits)} hits:\n   {hits}\n")
    return sorted(hits), sorted(candidates), intervals

# ─────────────────────────────────────────────────────────────────────────────
# 4) MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    copy_log()
    coords, rtts = parse_log()

    # Print raw data
    print("\nAll nodes data:")
    for n in sorted(coords):
        x,y,z,ram,cores = coords[n]
        rtt = rtts.get(n, float('nan'))
        print(f"  {n:25s} RTT={rtt:6.2f}ms  RAM={ram:2d}GB  Cores={cores:2d}")

    # Normalize & index
    hilb_map, sorted_arr, hil_list, minv, maxv, scale = normalize_and_index(coords, P)

    # Run the pure Hilbert range-query
    hits, cand, intervals = pure_hilbert_range_query(
        coords, rtts,
        hilb_map, sorted_arr, hil_list,
        minv, maxv, scale,
        THR_RTT, THR_RAM, THR_CORES
    )

    # Ground-truth & error counts
    true_set = sorted(
        n for n in coords
        if rtts[n] <= THR_RTT
        and coords[n][3] >= THR_RAM
        and coords[n][4] >= THR_CORES
    )
    print(f"▶ Ground truth ({len(true_set)} nodes): {true_set}")
    fp = sorted(set(cand) - set(true_set))
    fn = sorted(set(true_set) - set(cand))
    print(f"▶ False Positives ({len(fp)}): {fp}")
    print(f"▶ False Negatives ({len(fn)}): {fn}")

if __name__ == "__main__":
    main()
