#!/usr/bin/env python3
import os
import sys
import re
import subprocess
import random

import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve

P = 8  # bits per dimension

CONTAINER_NAME     = "clab-century-serf1"
CONTAINER_LOG_PATH = "/opt/serfapp/nodes_log.txt"
HOST_LOG_DIR       = "./dist"
HOST_LOG_PATH      = os.path.join(HOST_LOG_DIR, "nodes_log.txt")

node_resources = {f"clab-century-serf{i}": (16, 16) for i in range(1, 27)}
for i in range(14, 27):
    node_resources[f"clab-century-serf{i}"] = (8, 8)

node_storage = {}
for i in range(1, 7):    node_storage[f"clab-century-serf{i}"] = 300
for i in range(7, 14):   node_storage[f"clab-century-serf{i}"] = 300
for i in range(14, 21):  node_storage[f"clab-century-serf{i}"] = 300
for i in range(21, 27):  node_storage[f"clab-century-serf{i}"] = 300

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
                    r".*Node:\s*(\S+)\s*=>.*X:\s*([\-0-9.]+)\s*Y:\s*([\-0-9.]+)\s*Z:\s*([\-0-9.]+)",
                    line
                )
                if m:
                    node, x, y, z = m.groups()
                    coords[node] = (float(x), float(y), float(z))

            elif section == "rtt" and "Node:" in line:
                m = re.match(r".*Node:\s*(\S+)\s*=>.*RTT:\s*([\d.]+)\s*ms", line)
                if m:
                    node, ms = m.groups()
                    rtts[node] = float(ms) / 1000.0

    print(f"✅ Parsed {len(coords)} coords and {len(rtts)} RTTs")
    return coords, rtts

def hilbert_transform(data_map, bits, dims):
    nodes = list(data_map.keys())
    arr   = np.array([data_map[n] for n in nodes], dtype=float)

    minv  = arr.min(axis=0)
    maxv  = arr.max(axis=0)
    diff  = maxv - minv

    # avoid divide-by-zero when diff == 0
    scale = np.empty_like(diff)
    scale.fill(1.0)
    nonzero = diff != 0
    scale[nonzero] = (2**bits - 1) / diff[nonzero]

    grid  = np.rint((arr - minv) * scale).astype(int)

    hc    = HilbertCurve(bits, dims)
    dists = {
        n: hc.distance_from_point(grid[i].tolist())
        for i, n in enumerate(nodes)
    }
    return dists, minv, scale, hc

def make_hilbert_intervals(hc, minv, scale, lb_pt, ub_pt):
    lb_g = np.floor((lb_pt - minv) * scale).astype(int)
    ub_g = np.ceil ((ub_pt - minv) * scale).astype(int)
    lb_g = np.clip(lb_g, 0, 2**P - 1)
    ub_g = np.clip(ub_g, 0, 2**P - 1)

    ranges = [range(lb_g[d], ub_g[d] + 1) for d in range(len(lb_g))]
    pts = [(x, y, z)
           for x in ranges[0]
           for y in ranges[1]
           for z in ranges[2]]

    dists = sorted(hc.distance_from_point(list(pt)) for pt in pts)
    intervals, start, prev = [], dists[0], dists[0]
    for d in dists[1:]:
        if d == prev + 1:
            prev = d
        else:
            intervals.append((start, prev))
            start = prev = d
    intervals.append((start, prev))
    return intervals

def pure_hilbert_query(
    coord_dists, coord_minv, coord_scale, coord_hc,
    res_dists,   res_minv,    res_scale,   res_hc,
    coords,
    time_window, ram_thresh, core_thresh, stor_thresh
):
    cur = np.array(coords[CONTAINER_NAME])

    lb = cur - time_window
    ub = cur + time_window
    coord_iv = make_hilbert_intervals(coord_hc, coord_minv, coord_scale, lb, ub)

    lbp = np.array([ram_thresh, core_thresh, stor_thresh])
    ubp = np.array([
        max(r[0] for r in node_resources.values()),
        max(r[1] for r in node_resources.values()),
        max(node_storage.values())
    ])
    param_iv = make_hilbert_intervals(res_hc, res_minv, res_scale, lbp, ubp)

    def in_any(idx, intervals):
        return any(lo <= idx <= hi for lo, hi in intervals)

    return [
        n for n in coord_dists
        if in_any(coord_dists[n], coord_iv)
        and in_any(res_dists[n],   param_iv)
    ]

def main():
    time_window = 0.02  # seconds
    ram_thresh  = 4
    core_thresh = 4
    stor_thresh = 200

    print(f"🔧 Query parameters: window={time_window*1000:.0f}ms, "
          f"RAM>{ram_thresh}GB, Cores>{core_thresh}, Storage>{stor_thresh}GB, P={P}")
    print(f"🚀 Running on node: {CONTAINER_NAME}")

    copy_log()
    coords, rtts = parse_log()
    rtts[CONTAINER_NAME] = 0.0

    coord_dists, coord_minv, coord_scale, coord_hc = hilbert_transform(coords, P, 3)
    resources = {n: (*node_resources[n], node_storage[n]) for n in node_resources}
    res_dists, res_minv, res_scale, res_hc         = hilbert_transform(resources, P, 3)

    matches = pure_hilbert_query(
        coord_dists, coord_minv, coord_scale, coord_hc,
        res_dists,   res_minv,    res_scale,   res_hc,
        coords,
        time_window, ram_thresh, core_thresh, stor_thresh
    )

    gt = []
    for n, xyz in coords.items():
        if rtts.get(n, float('inf')) <= time_window:
            r, c = node_resources[n]
            s    = node_storage[n]
            if r > ram_thresh and c > core_thresh and s > stor_thresh:
                gt.append(n)
    gt.sort()

    fp = sorted(n for n in matches if n not in gt)
    fn = sorted(n for n in gt      if n not in matches)

    def show(n):
        x, y, z = coords[n]
        rt_ms   = rtts.get(n, 0.0) * 1000
        r, c    = node_resources[n]
        s       = node_storage[n]
        print(f"{n:20} coord=({x:.3f},{y:.3f},{z:.3f})  "
              f"RTT={rt_ms:6.1f}ms  RAM={r}  Cores={c}  Stor={s}  "
              f"CoordIdx={coord_dists[n]:7}  ParamIdx={res_dists[n]:7}")

    print(f"\n=== Ground Truth (count={len(gt)}) ===")
    for n in gt:     show(n)
    print(f"\n=== Pure-Hilbert Matches (count={len(matches)}) ===")
    for n in matches: show(n)
    print(f"\n=== False Positives (count={len(fp)}) ===")
    for n in fp:     show(n)
    if not fp:       print("  (none)")
    print(f"\n=== False Negatives (count={len(fn)}) ===")
    for n in fn:     show(n)
    if not fn:       print("  (none)")

if __name__ == "__main__":
    main()
