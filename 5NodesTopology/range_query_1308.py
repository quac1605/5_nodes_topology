#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
range_query_1308.py  (in-code per-node resources)
--------------------------------------------------
Two-stage *pure-Hilbert* range query with:
  1) Coordinate-space spherical cover (intervals in Hilbert index space)
     - radius is *calibrated* to measured RTTs using median(RTT_ms / coord_dist_sec)
  2) Resource-space filter (RAM, vCores, Storage)
     - intervals are formed from the *occupied* Hilbert indices of nodes that
       STRICTLY satisfy the thresholds (RAM>·, Cores>·, Storage>·).

Ground-truth (for evaluation only):
  - Prefer measured RTT (ms); fallback to Euclidean distance in Vivaldi coords (s→ms)
  - Apply RAM/cores/storage thresholds on original values (from the dicts below)

EDIT HERE: define your per-node RAM/cores/storage below in the section
'Per-node resources (editable)'. No CSV, no randomness.
"""
from __future__ import annotations

import os, re, sys, subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve

# -------------------- Config --------------------

P  = int(os.getenv("HILBERT_BITS", "6"))  # bits per dimension (8 good; 9–10 tighter)
ND = 3

CONTAINER_NAME     = os.getenv("CLAB_NODE", "clab-century-serf1")
CONTAINER_LOG_PATH = os.getenv("CONTAINER_LOG_PATH", "/opt/serfapp/nodes_log.txt")
HOST_LOG_DIR       = os.getenv("HOST_LOG_DIR", "./dist")
HOST_LOG_PATH      = os.path.join(HOST_LOG_DIR, "nodes_log.txt")

TIME_WINDOW_MS = float(os.getenv("TIME_WINDOW_MS", "120"))   # ms
RAM_THRESH     = float(os.getenv("RAM_THRESH", "7"))        # GB
CORE_THRESH    = float(os.getenv("CORE_THRESH", "7"))       # vCores
STOR_THRESH    = float(os.getenv("STOR_THRESH", "100"))     # GB

# -------------------- Per-node resources (editable) --------------------
# Define RAM/cores per node:
node_resources: Dict[str, tuple[int,int]] = {f"clab-century-serf{i}": (8, 8) for i in range(1, 27)}
for i in range(7, 27):  # example override for nodes 14..26
    node_resources[f"clab-century-serf{i}"] = (16, 16)

# Define storage per node (GB):
node_storage: Dict[str, int] = {}
for i in range(1, 7):    node_storage[f"clab-century-serf{i}"]  = 300
for i in range(7, 14):   node_storage[f"clab-century-serf{i}"]  = 300
for i in range(14, 21):  node_storage[f"clab-century-serf{i}"]  = 300
for i in range(21, 27):  node_storage[f"clab-century-serf{i}"]  = 300

# TIP: Adjust the dicts above as needed. All nodes present in the log must be covered.

# -------------------- Utilities --------------------

@dataclass
class Affine3:
    """Affine map between real space and integer grid [0..(2^P)-1]^3."""
    minv: np.ndarray     # shape (3,)
    scale: np.ndarray    # shape (3,)  real_to_grid: (x-min)*scale -> [0..N-1]

    @property
    def N(self) -> int:
        return (1 << P)

    def clamp_grid(self, g: np.ndarray) -> np.ndarray:
        return np.clip(g, 0, self.N - 1)

    def to_grid(self, x: np.ndarray) -> np.ndarray:
        """Real -> integer grid coordinates (np.int64)."""
        g = np.floor((x - self.minv) * self.scale + 1e-12)
        g = self.clamp_grid(g)
        return g.astype(np.int64)

    def to_real(self, g: np.ndarray) -> np.ndarray:
        """Integer grid -> real coordinate (lower cell corner)."""
        return (g.astype(float) / self.scale) + self.minv

def compress_intervals(sorted_indices: List[int]) -> List[Tuple[int, int]]:
    """Merge consecutive ints into [start, end] inclusive."""
    if not sorted_indices:
        return []
    out = []
    s = e = sorted_indices[0]
    for d in sorted_indices[1:]:
        if d == e + 1:
            e = d
        else:
            out.append((s, e)); s = e = d
    out.append((s, e))
    return out

def index_in_intervals(idx: int, intervals: List[Tuple[int, int]]) -> bool:
    """Binary search over non-overlapping, sorted intervals."""
    lo, hi = 0, len(intervals) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        a, b = intervals[mid]
        if idx < a: hi = mid - 1
        elif idx > b: lo = mid + 1
        else: return True
    return False

# -------------------- Parsing --------------------

COORD_RE = re.compile(r".*Node:\s*(\S+)\s*=>\s*X:\s*([\-0-9.]+)\s*Y:\s*([\-0-9.]+)\s*Z:\s*([\-0-9.]+).*")
RTT_RE   = re.compile(r".*Node:\s*(\S+)\s*=>\s*RTT:\s*([\d.]+)\s*ms")

def copy_log_from_container() -> bool:
    os.makedirs(HOST_LOG_DIR, exist_ok=True)
    try:
        subprocess.run(
            ["docker", "cp", f"{CONTAINER_NAME}:{CONTAINER_LOG_PATH}", HOST_LOG_PATH],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return True
    except Exception:
        return os.path.exists(HOST_LOG_PATH)

def parse_log(path: str):
    coords, rtts, current_node = {}, {}, None
    section = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line: continue
            if "[COORDINATES]" in line: section = "coord"; continue
            if "[RTT]" in line:         section = "rtt";   continue

            if section == "coord" and "Node:" in line:
                m = COORD_RE.match(line)
                if m:
                    node, xs, ys, zs = m.groups()
                    coords[node] = np.array([float(xs), float(ys), float(zs)], dtype=float)
                    if "[CURRENT NODE]" in line:
                        current_node = node
            elif section == "rtt" and "Node:" in line:
                m = RTT_RE.match(line)
                if m:
                    node, ms = m.groups()
                    rtts[node] = float(ms)
    return coords, rtts, current_node

# -------------------- Calibration --------------------

def coordinate_radius_with_calibration(center: np.ndarray,
                                       coords: Dict[str, np.ndarray],
                                       rtts_ms: Dict[str, float],
                                       time_window_ms: float,
                                       self_name: str) -> float:
    """
    radius_sec = time_window_ms / median_i( rtt_i_ms / coord_dist_i_sec )
    """
    ratios = []
    for n, x in coords.items():
        if n == self_name:
            continue
        d_sec = float(np.linalg.norm(x - center))  # seconds
        r_ms  = rtts_ms.get(n, None)
        if r_ms is not None and d_sec > 0:
            ratios.append(r_ms / d_sec)  # ms per second
    if ratios:
        k_ms_per_s = float(np.median(ratios))
        if k_ms_per_s > 0:
            return time_window_ms / k_ms_per_s  # seconds
    return time_window_ms / 1000.0  # fallback

# -------------------- Intervals builders --------------------

def build_coord_ball_intervals(hc: HilbertCurve,
                               aff: Affine3,
                               center_real: np.ndarray,
                               radius_real: float) -> tuple[list[tuple[int,int]], int]:
    """
    Cover only grid points whose REAL coords are inside the L2 ball.
    Return (intervals, covered_points_count).
    """
    N = aff.N
    cg = aff.to_grid(center_real)
    r_grid = np.ceil(radius_real * aff.scale + 1e-12).astype(int)
    mins = np.maximum(cg - r_grid, 0)
    maxs = np.minimum(cg + r_grid, N - 1)

    ds = []
    count = 0
    r2 = (radius_real + 1e-15) ** 2
    for ix in range(mins[0], maxs[0] + 1):
        for iy in range(mins[1], maxs[1] + 1):
            for iz in range(mins[2], maxs[2] + 1):
                g = np.array([ix, iy, iz], dtype=np.int64)
                xr = aff.to_real(g)
                if float(np.linalg.norm(xr - center_real)**2) <= r2:
                    ds.append(hc.distance_from_point(g.tolist()))
                    count += 1
    ds = sorted(set(ds))
    return compress_intervals(ds), count

def build_param_intervals_from_nodes(hc: HilbertCurve,
                                     aff: Affine3,
                                     ram: Dict[str, int],
                                     cores: Dict[str, int],
                                     stor: Dict[str, int],
                                     nodes: List[str],
                                     ram_thr: float, core_thr: float, stor_thr: float) -> tuple[list[tuple[int,int]], int]:
    """
    Build param intervals from *occupied* points that pass thresholds strictly.
    """
    ds = set()
    for n in nodes:
        if (ram[n] > ram_thr) and (cores[n] > core_thr) and (stor[n] > stor_thr):
            gp = aff.to_grid(np.array([ram[n], cores[n], stor[n]], dtype=float)).tolist()
            ds.add(hc.distance_from_point(gp))
    ds = sorted(ds)
    return compress_intervals(ds), len(ds)

# -------------------- Pretty printing --------------------

def show_node(name: str,
              coords: Dict[str, np.ndarray],
              rtts: Dict[str, float],
              ram: Dict[str, int],
              cores: Dict[str, int],
              stor: Dict[str, int],
              coord_idx: Dict[str, int],
              param_idx: Dict[str, int]):
    x, y, z = coords[name].tolist()
    rtt = rtts.get(name, 0.0)
    print(f"{name:<20} coord=({x:+.3f},{y:+.3f},{z:+.3f})  RTT={rtt:6.1f}ms  "
          f"RAM={ram[name]:<3}  Cores={cores[name]:<3}  Stor={stor[name]:<4}  "
          f"CoordIdx={coord_idx[name]:<8}  ParamIdx={param_idx[name]:<8}")

def print_intervals(title: str, intervals: List[Tuple[int,int]], total_points: int | None = None,
                    domain_total: int | None = None, max_lines: int = 32):
    print(f"\n=== {title} (count={len(intervals)}) ===")
    if total_points is not None and domain_total is not None:
        pct = (100.0 * total_points / domain_total) if domain_total > 0 else 0.0
        print(f"covering {total_points} / {domain_total} grid points (~{pct:.4f}%)")
    k = min(len(intervals), max_lines)
    for i in range(k):
        a, b = intervals[i]
        print(f"  [{a}, {b}]")
    if len(intervals) > k:
        print("  …")

# -------------------- Main --------------------

def main():
    # Header
    print(f"🔧 Query parameters: window={int(TIME_WINDOW_MS)}ms, RAM>{int(RAM_THRESH)}GB, "
          f"Cores>{int(CORE_THRESH)}, Storage>{int(STOR_THRESH)}GB, P={P}")
    print(f"🚀 Running on node: {CONTAINER_NAME}")

    if copy_log_from_container():
        print(f"✅ Copied log to {HOST_LOG_PATH}")
    else:
        print(f"⚠️ Could not docker cp; trying local file at {HOST_LOG_PATH}")

    coords, rtts, current = parse_log(HOST_LOG_PATH)
    if current is None:
        current = CONTAINER_NAME
    print(f"✅ Parsed {len(coords)} coords and {len(rtts)} RTTs")

    nodes = sorted(coords.keys())
    if current not in nodes:
        print(f"ERROR: current node '{current}' not found in coordinates.", file=sys.stderr)
        sys.exit(2)

    # Validate resources cover all nodes
    missing = [n for n in nodes if n not in node_resources or n not in node_storage]
    if missing:
        print(f"❌ Per-node resources missing entries for: {', '.join(missing)}", file=sys.stderr)
        sys.exit(3)

    # Split dicts
    ram   = {n: int(node_resources[n][0]) for n in nodes}
    cores = {n: int(node_resources[n][1]) for n in nodes}
    stor  = {n: int(node_storage[n])      for n in nodes}

    # Affine maps
    all_xyz = np.array([coords[n] for n in nodes], dtype=float)
    min_xyz = all_xyz.min(axis=0); max_xyz = all_xyz.max(axis=0)
    extent  = np.maximum(max_xyz - min_xyz, 1e-9)
    scale_xyz = ((1 << P) - 1) / extent
    coord_aff = Affine3(min_xyz, scale_xyz)

    all_res = np.array([[ram[n], cores[n], stor[n]] for n in nodes], dtype=float)
    min_res = all_res.min(axis=0); max_res = all_res.max(axis=0)
    extent_r = np.maximum(max_res - min_res, 1e-9)
    scale_r  = ((1 << P) - 1) / extent_r
    param_aff = Affine3(min_res, scale_r)

    # Hilbert
    hc_coord = HilbertCurve(P, ND)
    hc_param = HilbertCurve(P, ND)

    # Per-node indices (for printing only)
    coord_idx, param_idx = {}, {}
    for n in nodes:
        coord_idx[n] = hc_coord.distance_from_point(coord_aff.to_grid(coords[n]).tolist())
        param_idx[n] = hc_param.distance_from_point(param_aff.to_grid(np.array([ram[n], cores[n], stor[n]], float)).tolist())

    # --- Stage A: coord intervals with calibrated radius ---
    center = coords[current].astype(float)
    radius_real = coordinate_radius_with_calibration(center, coords, rtts, TIME_WINDOW_MS, current)  # seconds
    coord_intervals, coord_point_count = build_coord_ball_intervals(hc_coord, coord_aff, center, radius_real)

    # --- Stage B: param intervals from *occupied* points that pass thresholds (strict >) ---
    param_intervals, param_point_count = build_param_intervals_from_nodes(
        hc_param, param_aff, ram, cores, stor, nodes,
        RAM_THRESH, CORE_THRESH, STOR_THRESH
    )

    # Selection: indices-only
    matches = [n for n in nodes
               if index_in_intervals(coord_idx[n], coord_intervals)
               and index_in_intervals(param_idx[n], param_intervals)]
    matches.sort()

    # Ground Truth (evaluation only): measured RTT preferred
    gt = []
    for n in nodes:
        if n == current:
            metric_ms = 0.0
        else:
            metric_ms = rtts.get(n, float(np.linalg.norm(coords[n] - center) * 1000.0))
        if (metric_ms <= TIME_WINDOW_MS) and (ram[n] > RAM_THRESH) and (cores[n] > CORE_THRESH) and (stor[n] > STOR_THRESH):
            gt.append(n)
    gt.sort()

    fp = sorted(set(matches) - set(gt))
    fn = sorted(set(gt) - set(matches))

    # Prints
    print_intervals("Coord Intervals", coord_intervals, total_points=coord_point_count, domain_total=(1 << (3 * P)))
    print_intervals("Param Intervals", param_intervals, total_points=param_point_count)

    print("\n=== Ground Truth (count={}) ===".format(len(gt)))
    for n in gt: show_node(n, coords, rtts, ram, cores, stor, coord_idx, param_idx)

    print("\n=== Pure-Hilbert Matches (count={}) ===".format(len(matches)))
    for n in matches: show_node(n, coords, rtts, ram, cores, stor, coord_idx, param_idx)

    print("\n=== False Positives (count={}) ===".format(len(fp)))
    if not fp: print("  (none)")
    else:
        for n in fp: show_node(n, coords, rtts, ram, cores, stor, coord_idx, param_idx)

    print("\n=== False Negatives (count={}) ===".format(len(fn)))
    if not fn: print("  (none)")
    else:
        for n in fn: show_node(n, coords, rtts, ram, cores, stor, coord_idx, param_idx)

if __name__ == "__main__":
    main()
