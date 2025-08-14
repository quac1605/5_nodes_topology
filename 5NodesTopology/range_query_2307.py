#!/usr/bin/env python3
import os
import sys
import re
import subprocess
import numpy as np
from math import cos, sin, pi
from hilbertcurve.hilbertcurve import HilbertCurve

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
P = 8  # bits per dimension for Hilbert transforms

CONTAINER_NAME     = "clab-century-serf1"
CONTAINER_LOG_PATH = "/opt/serfapp/nodes_log.txt"
HOST_LOG_DIR       = "./dist"
HOST_LOG_PATH      = os.path.join(HOST_LOG_DIR, "nodes_log.txt")

# Base resources: (RAM GB, vCores)
node_resources = {f"clab-century-serf{i}": (16, 16) for i in range(1, 27)}
for i in range(14, 27):
    node_resources[f"clab-century-serf{i}"] = (8, 8)

# Storage assignments (in GB)
node_storage = {}
for i in range(1, 7):    node_storage[f"clab-century-serf{i}"] = 100
for i in range(7, 14):   node_storage[f"clab-century-serf{i}"] = 300
for i in range(14, 21):  node_storage[f"clab-century-serf{i}"] = 300
for i in range(21, 27):  node_storage[f"clab-century-serf{i}"] = 300

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: COPY & PARSE LOG (coords + RTT)
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
    with open(HOST_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if "[COORDINATES]" in line:
                section = "coord"; continue
            if "[RTT]" in line:
                section = "rtt"; continue

            if section == "coord" and "Node:" in line:
                m = re.match(
                    r".*Node:\s*(\S+).*X:\s*([\-0-9.]+)\s*Y:\s*([\-0-9.]+)\s*Z:\s*([\-0-9.]+)",
                    line
                )
                if m:
                    n, x, y, z = m.groups()
                    coords[n] = (float(x), float(y), float(z))

            elif section == "rtt" and "Node:" in line:
                m = re.match(r".*Node:\s*(\S+).*RTT:\s*([\d.]+)\s*ms", line)
                if m:
                    n, ms = m.groups()
                    # store RTT in seconds
                    rtts[n] = float(ms) / 1000.0

    print(f"✅ Parsed {len(coords)} coords and {len(rtts)} RTTs")
    return coords, rtts

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: GENERIC N-D → 1-D HILBERT TRANSFORM
# ─────────────────────────────────────────────────────────────────────────────
def hilbert_transform(data_map, bits, dims):
    nodes = list(data_map.keys())
    arr   = np.array([data_map[n] for n in nodes], dtype=float)

    minv = arr.min(axis=0)
    maxv = arr.max(axis=0)
    diff = maxv - minv

    # build scale without divide-by-zero warnings
    scale = np.ones_like(diff)
    nz = diff != 0
    scale[nz] = (2**bits - 1) / diff[nz]

    grid = np.rint((arr - minv) * scale).astype(int)
    hc   = HilbertCurve(bits, dims)
    dists = {
        nodes[i]: hc.distance_from_point(grid[i].tolist())
        for i in range(len(nodes))
    }
    return dists, minv, scale, hc

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3A: 3-D SPHERE COVER → HILBERT INTERVALS (coords)
# ─────────────────────────────────────────────────────────────────────────────
def make_coord_sphere_intervals(hc, minv, scale, center, radius):
    lb_g = np.floor((center - radius - minv) * scale).astype(int)
    ub_g = np.ceil ((center + radius - minv) * scale).astype(int)
    lb_g = np.clip(lb_g, 0, 2**P - 1)
    ub_g = np.clip(ub_g, 0, 2**P - 1)

    dists = []
    for x in range(lb_g[0], ub_g[0]+1):
        for y in range(lb_g[1], ub_g[1]+1):
            for z in range(lb_g[2], ub_g[2]+1):
                real = np.array([x, y, z]) / scale + minv
                if np.linalg.norm(real - center) <= radius:
                    dists.append(hc.distance_from_point([x, y, z]))
    dists.sort()
    if not dists:
        return []
    intervals = []
    start, prev = dists[0], dists[0]
    for d in dists[1:]:
        if d == prev + 1:
            prev = d
        else:
            intervals.append((start, prev))
            start = prev = d
    intervals.append((start, prev))
    return intervals

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3B: 3-D BOX COVER → HILBERT INTERVALS (RAM, Cores, Storage)
# ─────────────────────────────────────────────────────────────────────────────
def make_param_intervals(hc, minv, scale, lb_pt, ub_pt):
    lb_g = np.floor((lb_pt - minv) * scale).astype(int)
    ub_g = np.ceil ((ub_pt - minv) * scale).astype(int)
    lb_g = np.clip(lb_g, 0, 2**P - 1)
    ub_g = np.clip(ub_g, 0, 2**P - 1)

    pts = [
        (x, y, z)
        for x in range(lb_g[0], ub_g[0]+1)
        for y in range(lb_g[1], ub_g[1]+1)
        for z in range(lb_g[2], ub_g[2]+1)
    ]
    dists = sorted(hc.distance_from_point(list(pt)) for pt in pts)
    if not dists:
        return []
    intervals = []
    start, prev = dists[0], dists[0]
    for d in dists[1:]:
        if d == prev + 1:
            prev = d
        else:
            intervals.append((start, prev))
            start = prev = d
    intervals.append((start, prev))
    return intervals

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: PURE-HILBERT QUERY (two coord curves + one param curve)
# ─────────────────────────────────────────────────────────────────────────────
def pure_hilbert_query(
    coord_dists, coord_minv, coord_scale, coord_hc,
    coord2_dists, coord2_minv, coord2_scale, coord2_hc,
    res_dists, res_minv, res_scale, res_hc,
    coords,
    window, ram_thresh, core_thresh, stor_thresh
):
    center = np.array(coords[CONTAINER_NAME])

    iv1 = make_coord_sphere_intervals(coord_hc, coord_minv, coord_scale, center, window)
    iv2 = make_coord_sphere_intervals(coord2_hc, coord2_minv, coord2_scale, center, window)

    lbp = np.array([ram_thresh, core_thresh, stor_thresh])
    ubp = np.array([
        max(r for r,_ in node_resources.values()),
        max(c for _,c in node_resources.values()),
        max(node_storage.values())
    ])
    ivp = make_param_intervals(res_hc, res_minv, res_scale, lbp, ubp)

    def in_any(idx, intervals):
        return any(lo <= idx <= hi for lo, hi in intervals)

    return [
        n for n in coord_dists
        if in_any(coord_dists[n],  iv1)
        and in_any(coord2_dists[n], iv2)
        and in_any(res_dists[n],    ivp)
    ]

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Query parameters (all in seconds or GB/cores units)
    window      = 20 / 1000.0    # 20 ms → 0.020 seconds
    ram_thresh  = 4
    core_thresh = 4
    stor_thresh = 200
    print(f"🔧 Params: window={window*1000:.0f}ms, RAM>{ram_thresh}, "
          f"Cores>{core_thresh}, Stor>{stor_thresh}, P={P}")

    print(f"🚀 Running on node: {CONTAINER_NAME}")
    copy_log()
    coords, rtts = parse_log()
    # ensure current node RTT == 0
    rtts[CONTAINER_NAME] = 0.0

    # Debug output to verify units
    print(f"debug: min RTT = {min(rtts.values())*1000:.2f} ms, "
          f"threshold = {window*1000:.2f} ms")

    # Hilbert #1: original coords
    coord_dists, coord_minv, coord_scale, coord_hc = hilbert_transform(coords, P, 3)
    # Hilbert #2: coords rotated by 45° about Z
    θ = pi/4
    rot2 = {
        n: (
            coords[n][0]*cos(θ) - coords[n][1]*sin(θ),
            coords[n][0]*sin(θ) + coords[n][1]*cos(θ),
            coords[n][2]
        )
        for n in coords
    }
    coord2_dists, coord2_minv, coord2_scale, coord2_hc = hilbert_transform(rot2, P, 3)
    # Hilbert #3: RAM, Cores, Storage
    res_map = {n: (*node_resources[n], node_storage[n]) for n in coords}
    res_dists, res_minv, res_scale, res_hc = hilbert_transform(res_map, P, 3)

    # Pure-Hilbert matches
    matches = pure_hilbert_query(
        coord_dists, coord_minv, coord_scale, coord_hc,
        coord2_dists, coord2_minv, coord2_scale, coord2_hc,
        res_dists, res_minv, res_scale, res_hc,
        coords,
        window, ram_thresh, core_thresh, stor_thresh
    )

    # Ground truth using RTT + resource thresholds
    gt = []
    for n, xyz in coords.items():
        if rtts[n] <= window:
            r, c = node_resources[n]
            s    = node_storage[n]
            if r > ram_thresh and c > core_thresh and s > stor_thresh:
                gt.append(n)
    gt.sort()

    fp = sorted(n for n in matches if n not in gt)
    fn = sorted(n for n in gt      if n not in matches)

    def show(n):
        x, y, z = coords[n]
        rt_ms   = rtts[n] * 1000
        r, c    = node_resources[n]
        s       = node_storage[n]
        print(f"{n:20} coord=({x:.3f},{y:.3f},{z:.3f})  "
              f"RTT={rt_ms:6.1f}ms  RAM={r}  Cores={c}  Stor={s}  "
              f"Idx1={coord_dists[n]:8}  Idx2={coord2_dists[n]:8}  IdxP={res_dists[n]:8}")

    print(f"\n=== Ground Truth (count={len(gt)}) ===")
    for n in gt:     show(n)

    print(f"\n=== Pure-Hilbert Matches (count={len(matches)}) ===")
    for n in matches: show(n)

    print(f"\n=== False Positives (count={len(fp)}) ===")
    if fp:
        for n in fp: show(n)
    else:
        print("  (none)")

    print(f"\n=== False Negatives (count={len(fn)}) ===")
    if fn:
        for n in fn: show(n)
    else:
        print("  (none)")

if __name__ == "__main__":
    main()
