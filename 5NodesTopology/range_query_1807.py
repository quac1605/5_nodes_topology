#!/usr/bin/env python3
import os
import re
import subprocess
import bisect
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve

# ─────────────────────────────────────────────────────────────────────────────
# 1) Configuration & Serf‑log parsing
# ─────────────────────────────────────────────────────────────────────────────
P = 8   # ↓ lowered from 16 to  8 bits per dim
N = 5   # dims: X, Y, Z, RAM, vCores

CONTAINER_NAME     = "clab-century-serf1"
CONTAINER_LOG_PATH = "/opt/serfapp/nodes_log.txt"
HOST_LOG_DIR       = "./dist"
HOST_LOG_PATH      = os.path.join(HOST_LOG_DIR, "nodes_log.txt")

node_resources = {f"clab-century-serf{i}": (8, 8) for i in range(1, 27)}

def copy_log_from_container():
    os.makedirs(HOST_LOG_DIR, exist_ok=True)
    try:
        subprocess.run([
            "docker", "cp",
            f"{CONTAINER_NAME}:{CONTAINER_LOG_PATH}",
            HOST_LOG_PATH
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"[Error] cannot copy log: {e}")
        return False
    return True

def parse_log():
    coords, rtts = {}, {}
    if not os.path.isfile(HOST_LOG_PATH):
        raise FileNotFoundError(HOST_LOG_PATH)
    section = None
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
                if cm and node in node_resources:
                    x, y, z = map(float, cm.groups())
                    ram, cores = node_resources[node]
                    coords[node] = (x, y, z, ram, cores)
            elif section == "rtt":
                rm = re.search(r"RTT:\s*([\d.]+)\s*ms", rest)
                if rm:
                    rtts[node] = float(rm.group(1))
    return coords, rtts

# ─────────────────────────────────────────────────────────────────────────────
# 2) Normalize coordinates → integer grid, compute Hilbert indices
# ─────────────────────────────────────────────────────────────────────────────
def normalize_coordinates(coords, bits):
    arr  = np.array(list(coords.values()), dtype=float)
    minv = arr.min(axis=0)
    maxv = arr.max(axis=0)
    diff = maxv - minv

    scale = np.empty_like(diff)
    nz = diff != 0
    scale[nz] = (2**bits - 1) / diff[nz]
    scale[~nz] = 1.0

    norm = (arr - minv) * scale
    norm[:, ~nz] = 0

    mapping = {
        node: tuple(norm[i].round().astype(int))
        for i, node in enumerate(coords)
    }
    return mapping, minv, maxv

def compute_hilbert_index(norm_map):
    hc = HilbertCurve(P, N)
    hilb_map   = {n: int(hc.distance_from_point(list(pt))) for n, pt in norm_map.items()}
    sorted_arr = sorted((h, n) for n, h in hilb_map.items())
    hil_list   = [h for h, _ in sorted_arr]
    return hilb_map, sorted_arr, hil_list

# ─────────────────────────────────────────────────────────────────────────────
# 3) Multi-interval decomposition with early-stop
# ─────────────────────────────────────────────────────────────────────────────
def decompose_hilbert_ranges(hc, min_pt, max_pt):
    N      = hc.n
    size   = 2**hc.p
    total  = size**N
    qwidth = [max_pt[d] - min_pt[d] for d in range(N)]
    intervals = []

    def recurse(origin, span, h_lo, h_hi):
        # Outside?
        for d in range(N):
            if origin[d] + span <= min_pt[d] or origin[d] >= max_pt[d]:
                return
        # Fully inside?
        if all(origin[d] >= min_pt[d] and origin[d] + span <= max_pt[d] for d in range(N)):
            intervals.append((h_lo, h_hi))
            return
        # If this cell is no larger than the query width in every dim, accept
        if all((span <= qwidth[d] or qwidth[d] == 0) for d in range(N)):
            intervals.append((h_lo, h_hi))
            return
        # Otherwise subdivide into 2^N children
        half  = span // 2
        block = half**N
        for i in range(2**N):
            offs = [(i >> b) & 1 for b in range(N)]
            child_origin = [origin[d] + offs[d]*half for d in range(N)]
            c_lo = h_lo + i*block
            c_hi = c_lo + block
            recurse(child_origin, half, c_lo, c_hi)

    recurse([0]*N, size, 0, total)
    intervals.sort()
    merged = []
    for a, b in intervals:
        if not merged or a > merged[-1][1]:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    return [(a, b) for a, b in merged]

# ─────────────────────────────────────────────────────────────────────────────
# 4) Range-query using multiple intervals + final filter
# ─────────────────────────────────────────────────────────────────────────────
def range_query(
    current_node,
    coords, rtts,
    norm_map, minv, maxv,
    sorted_arr, hil_list,
    rtt_thresh_ms,
    ram_min, cores_min,
    return_candidates_only=False
):
    # Build raw 5‑D box
    x0, y0, z0, _, _ = coords[current_node]
    raw_min = (x0 - rtt_thresh_ms, y0 - rtt_thresh_ms, z0 - rtt_thresh_ms, ram_min,   cores_min)
    raw_max = (x0 + rtt_thresh_ms, y0 + rtt_thresh_ms, z0 + rtt_thresh_ms, maxv[3], maxv[4])

    # Compute scale
    diff  = maxv - minv
    scale = np.empty_like(diff)
    nz    = diff != 0
    scale[nz] = (2**P - 1) / diff[nz]
    scale[~nz] = 1.0

    # Map raw box to integer grid (half-open)
    min_pt, max_pt = [], []
    for d in range(N):
        lo = int(np.floor ((raw_min[d] - minv[d]) * scale[d]))
        hi = int(np.ceil  ((raw_max[d] - minv[d]) * scale[d])) + 1
        min_pt.append(max(0, min(lo, 2**P)))
        max_pt.append(max(0, min(hi, 2**P)))
    min_pt, max_pt = tuple(min_pt), tuple(max_pt)

    # Decompose into exact intervals
    hc        = HilbertCurve(P, N)
    intervals = decompose_hilbert_ranges(hc, min_pt, max_pt)

    # Bisect each interval
    candidates = set()
    for lo, hi in intervals:
        i0 = bisect.bisect_left(hil_list, lo)
        i1 = bisect.bisect_right(hil_list, hi)
        for i in range(i0, i1):
            candidates.add(sorted_arr[i][1])

    if return_candidates_only:
        return sorted(candidates)

    # Final filter: RTT ≤, RAM ≥, cores ≥
    return sorted([
        n for n in candidates
        if rtts.get(n, float('inf')) <= rtt_thresh_ms
        and coords[n][3] >= ram_min
        and coords[n][4] >= cores_min
    ])

# ─────────────────────────────────────────────────────────────────────────────
# 5) Main – tie it all together, show results
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not copy_log_from_container():
        exit(1)

    coords, rtts                   = parse_log()
    norm_map, minv, maxv           = normalize_coordinates(coords, P)
    hilb_map, sorted_arr, hil_list = compute_hilbert_index(norm_map)

    # Print all nodes data
    hc     = HilbertCurve(P, N)
    h_curr = hc.distance_from_point(list(norm_map[CONTAINER_NAME]))
    print("\nAll nodes data:")
    for node in sorted(coords):
        x, y, z, ram, cores = coords[node]
        rtt = rtts.get(node, float('nan'))
        h   = hilb_map[node]
        hd  = abs(h - h_curr)
        print(f"  {node:25s} RTT={rtt:6.2f}ms  RAM={ram:2d}GB  "
              f"Cores={cores:2d}  Hilbert={h}  HDist={hd}")

    # Raw candidates
    candidates = range_query(
        current_node           = CONTAINER_NAME,
        coords                 = coords,
        rtts                   = rtts,
        norm_map               = norm_map,
        minv                   = minv,
        maxv                   = maxv,
        sorted_arr             = sorted_arr,
        hil_list               = hil_list,
        rtt_thresh_ms          = 20.0,
        ram_min                = 8,
        cores_min              = 8,
        return_candidates_only = True
    )
    print(f"\nHilbert-interval candidates: {len(candidates)} nodes")
    for n in candidates:
        print(f"  {n}")

    # Final hits
    final_hits = range_query(
        current_node  = CONTAINER_NAME,
        coords        = coords,
        rtts          = rtts,
        norm_map      = norm_map,
        minv          = minv,
        maxv          = maxv,
        sorted_arr    = sorted_arr,
        hil_list      = hil_list,
        rtt_thresh_ms = 20.0,
        ram_min       = 8,
        cores_min     = 8
    )
    print(f"\nFinal nodes matching all criteria: {len(final_hits)} nodes")
    for n in final_hits:
        print(f"  {n}")

    # False positives / negatives
    set_cand  = set(candidates)
    set_final = set(final_hits)
    false_pos = sorted(set_cand - set_final)
    false_neg = sorted(set_final - set_cand)

    print(f"\nFalse Positives ({len(false_pos)}):")
    for n in false_pos:
        print(f"  {n}")
    print(f"\nFalse Negatives ({len(false_neg)}):")
    for n in false_neg:
        print(f"  {n}")
