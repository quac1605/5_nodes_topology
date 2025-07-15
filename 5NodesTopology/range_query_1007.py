#!/usr/bin/env python3
import os, re, subprocess, bisect
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve

# ─────────────────────────────────────────────────────────────────────────────
# 1) Configuration & Serf‑log parsing (same as your analyze_cluster.py)
# ─────────────────────────────────────────────────────────────────────────────
P = 16  # bits per dimension
N = 5   # dims: X, Y, Z, RAM, vCores

CONTAINER_NAME     = "clab-century-serf1"
CONTAINER_LOG_PATH = "/opt/serfapp/nodes_log.txt"
HOST_LOG_DIR       = "./dist"
HOST_LOG_PATH      = os.path.join(HOST_LOG_DIR, "nodes_log.txt")

# All nodes have (8 GB, 8 vCores) in your setup
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
    arr  = np.array(list(coords.values()), dtype=float)  # shape=(n,5)
    minv = arr.min(axis=0)
    maxv = arr.max(axis=0)
    diff = maxv - minv

    # Build scale safely without divide‑by‑zero warnings:
    scale = np.empty_like(diff)
    nz = diff != 0
    scale[nz] = (2**bits - 1) / diff[nz]
    scale[~nz] = 1.0

    norm = (arr - minv) * scale
    norm[:, ~nz] = 0  # zero out any constant‐dims

    mapping = {
        node: tuple(norm[i].round().astype(int))
        for i, node in enumerate(coords)
    }
    return mapping, minv, maxv

def compute_hilbert_index(norm_map):
    hc        = HilbertCurve(P, N)
    hilb_map  = {n: int(hc.distance_from_point(list(pt))) for n, pt in norm_map.items()}
    sorted_arr= sorted((h, n) for n, h in hilb_map.items())
    hil_list  = [h for h, _ in sorted_arr]
    return hilb_map, sorted_arr, hil_list

# ─────────────────────────────────────────────────────────────────────────────
# 3) Fast range‐query via min/max of the 32 corners + one bisect
# ─────────────────────────────────────────────────────────────────────────────
def range_query(
    current_node,
    coords, rtts,
    norm_map, minv, maxv,
    sorted_arr, hil_list,
    rtt_thresh_ms, ram_max, cores_max
):
    # Raw center = Serf coords of current_node
    x0, y0, z0, _, _ = coords[current_node]

    # Define your raw query‐box in the 5 dims
    raw_min = (x0 - rtt_thresh_ms, y0 - rtt_thresh_ms, z0 - rtt_thresh_ms, 0, 0)
    raw_max = (x0 + rtt_thresh_ms, y0 + rtt_thresh_ms, z0 + rtt_thresh_ms, ram_max, cores_max)

    # Recompute scale exactly as above
    diff  = maxv - minv
    scale = np.empty_like(diff)
    nz = diff != 0
    scale[nz] = (2**P - 1) / diff[nz]
    scale[~nz] = 1.0

    # Generate all 2^5 = 32 corner grid‐points
    corners = []
    for mask in range(1 << N):
        pt = []
        for d in range(N):
            val = raw_max[d] if (mask >> d) & 1 else raw_min[d]
            # map to grid and choose floor for low‐corners, ceil for high‐corners
            scaled = (val - minv[d]) * scale[d]
            idx    = int(np.ceil(scaled)) if (mask >> d) & 1 else int(np.floor(scaled))
            # clamp into [0, 2^P−1]
            idx    = max(0, min(idx, 2**P - 1))
            pt.append(idx)
        corners.append(tuple(pt))

    # Include the current node’s own index too
    hc = HilbertCurve(P, N)
    h_curr = hc.distance_from_point(list(norm_map[current_node]))
    h_vals = [hc.distance_from_point(list(c)) for c in corners] + [h_curr]

    lo, hi = min(h_vals), max(h_vals)

    # Bisect once to get a small superset
    i0 = bisect.bisect_left(hil_list, lo)
    i1 = bisect.bisect_right(hil_list, hi)
    candidates = {sorted_arr[i][1] for i in range(i0, i1)}

    # Final exact filter on RTT, RAM, cores
    return sorted([
        n for n in candidates
        if rtts.get(n, float('inf')) <= rtt_thresh_ms
        and coords[n][3] <= ram_max
        and coords[n][4] <= cores_max
    ])

# ─────────────────────────────────────────────────────────────────────────────
# 4) Main – tie it all together
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not copy_log_from_container():
        exit(1)

    coords, rtts                   = parse_log()
    norm_map, minv, maxv           = normalize_coordinates(coords, P)
    hilb_map, sorted_arr, hil_list = compute_hilbert_index(norm_map)

    # Example: within 20 ms RTT, ≤16 GB RAM, ≤16 vCores of clab-century-serf1
    matches = range_query(
        current_node=CONTAINER_NAME,
        coords=coords,
        rtts=rtts,
        norm_map=norm_map,
        minv=minv,
        maxv=maxv,
        sorted_arr=sorted_arr,
        hil_list=hil_list,
        rtt_thresh_ms=40.0,
        ram_max=16,
        cores_max=16
    )

    print("Nodes matching query:", matches)
