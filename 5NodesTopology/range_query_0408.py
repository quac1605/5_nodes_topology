#!/usr/bin/env python3
import os
import sys
import re
import subprocess
import random

import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
P = 6  # bits per dimension for Hilbert transforms

CONTAINER_NAME     = "clab-century-serf1"
CONTAINER_LOG_PATH = "/opt/serfapp/nodes_log.txt"
HOST_LOG_DIR       = "./dist"
HOST_LOG_PATH      = os.path.join(HOST_LOG_DIR, "nodes_log.txt")

# Base resources: (RAM GB, vCores)
node_resources = {f"clab-century-serf{i}": (16, 16) for i in range(1, 27)}
for i in range(14, 27):
    node_resources[f"clab-century-serf{i}"] = (8, 8)

# Pseudo‐random storage: 100–1000 GB
node_storage = {n: random.randint(100, 1000) for n in node_resources}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Copy & Parse Serf Log for COORDINATES and RTT
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
    coords = {}
    rtts   = {}
    section = None
    with open(HOST_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if "[COORDINATES]" in line:
                section = "coord"; continue
            if "[RTT]" in line:
                section = "rtt"; continue

            if section == "coord" and "Node:" in line:
                m = re.match(
                    r".*Node:\s*(\S+)\s*=>.*X:\s*([\-0-9.]+)\s*Y:\s*([\-0-9.]+)\s*Z:\s*([\-0-9.]+)",
                    line
                )
                if m:
                    node, x, y, z = m.groups()
                    coords[node] = (float(x), float(y), float(z))

            elif section == "rtt" and "Node:" in line:
                # accept "RTT: 12.27 ms" or "RTT:12.27ms"
                m = re.match(r".*Node:\s*(\S+)\s*=>.*RTT:\s*([\d.]+)\s*ms", line)
                if m:
                    node, ms = m.groups()
                    rtts[node] = float(ms) / 1000.0  # convert to seconds

    print(f"✅ Parsed {len(coords)} coords  and {len(rtts)} RTTs")
    return coords, rtts

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Generic N-D → 1-D Hilbert Transform
# ─────────────────────────────────────────────────────────────────────────────
def hilbert_transform(data_map, bits, dims):
    nodes = list(data_map.keys())
    arr   = np.array([data_map[n] for n in nodes], dtype=float)

    minv  = arr.min(axis=0)
    maxv  = arr.max(axis=0)
    diff  = maxv - minv
    scale = np.where(diff != 0, (2**bits - 1) / diff, 1.0)
    grid  = np.rint((arr - minv) * scale).astype(int)

    hc    = HilbertCurve(bits, dims)
    dists = {
        n: hc.distance_from_point(grid[i].tolist())
        for i, n in enumerate(nodes)
    }
    return dists, minv, scale, hc

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Pure-Hilbert Range Query (coord-idx, param-idx)
# ─────────────────────────────────────────────────────────────────────────────
def pure_hilbert_query(
    coord_dists, coord_minv, coord_scale, coord_hc,
    res_dists,  res_minv,   res_scale,   res_hc,
    coords,
    time_window=0.1,      # seconds (100 ms)
    ram_thresh=8, core_thresh=8, stor_thresh=200
):
    # current node’s coord
    cur_xyz = np.array(coords[CONTAINER_NAME])
    lb_xyz  = cur_xyz - time_window
    ub_xyz  = cur_xyz + time_window

    def to_idx(pt, minv, scale, hc):
        grid = np.rint((pt - minv) * scale).astype(int)
        grid = np.clip(grid, 0, 2**P - 1)
        return hc.distance_from_point(grid.tolist())

    lb_idx = to_idx(lb_xyz, coord_minv, coord_scale, coord_hc)
    ub_idx = to_idx(ub_xyz, coord_minv, coord_scale, coord_hc)

    lb_param = np.array([ram_thresh, core_thresh, stor_thresh])
    ub_param = np.array([
        max(r[0] for r in node_resources.values()),
        max(r[1] for r in node_resources.values()),
        max(node_storage.values())
    ])
    plb_idx = to_idx(lb_param, res_minv, res_scale, res_hc)
    pub_idx = to_idx(ub_param, res_minv, res_scale, res_hc)

    return [
        n for n, cidx in coord_dists.items()
        if lb_idx <= cidx <= ub_idx
        and plb_idx <= res_dists[n] <= pub_idx
    ]

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"🚀 Running on node: {CONTAINER_NAME}")
    copy_log()
    coords, rtts = parse_log()

    # ensure current node has an RTT entry (0 ms)
    rtts[CONTAINER_NAME] = 0.0

    if not coords or not rtts:
        print("⚠️ Missing coords/RTT data; exiting.")
        sys.exit(0)

    # build Hilbert indexes
    coord_dists, coord_minv, coord_scale, coord_hc = hilbert_transform(coords, P, 3)
    resources = {n: (*node_resources[n], node_storage[n]) for n in node_resources}
    res_dists, res_minv, res_scale, res_hc       = hilbert_transform(resources, P, 3)

    # pure-Hilbert matches
    matches = pure_hilbert_query(
        coord_dists, coord_minv, coord_scale, coord_hc,
        res_dists,  res_minv,   res_scale,   res_hc,
        coords
    )

    # --- ground truth using actual RTT + resource thresholds ---
    time_window = 0.1  # sec
    gt = []
    for n, xyz in coords.items():
        if rtts.get(n, float('inf')) <= time_window:
            r, c = node_resources[n]
            s     = node_storage[n]
            if r > 8 and c > 8 and s > 200:
                gt.append(n)
    gt.sort()

    fp = sorted(n for n in matches if n not in gt)
    fn = sorted(n for n in gt      if n not in matches)

    # pretty‐print helper
    def show(n):
        x, y, z = coords[n]
        rt_ms   = rtts.get(n, 0.0) * 1000
        r, c    = node_resources[n]
        s       = node_storage[n]
        print(
            f"{n:20} coord=({x:.3f},{y:.3f},{z:.3f})  "
            f"RTT={rt_ms:6.1f}ms  RAM={r}  Cores={c}  Stor={s}  "
            f"CoordIdx={coord_dists[n]:6}  ParamIdx={res_dists[n]:6}"
        )

    print("\n=== Ground Truth (RTT≤100ms & RAM>8 & Cores>8 & Stor>200) ===")
    for n in gt:
        show(n)

    print("\n=== Pure-Hilbert Matches ===")
    for n in matches:
        show(n)

    print("\n=== False Positives (Hilbert ✓ but Truth ✗) ===")
    if fp:
        for n in fp:
            show(n)
    else:
        print("  (none)")

    print("\n=== False Negatives (Truth ✓ but Hilbert ✗) ===")
    if fn:
        for n in fn:
            show(n)
    else:
        print("  (none)")

if __name__ == "__main__":
    main()
