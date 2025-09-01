#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import sys
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import random
from hilbertcurve.hilbertcurve import HilbertCurve

# ==================== Config ====================

P  = int(os.getenv("HILBERT_BITS", "8"))   # bits per dimension (try 6/8/10)
ND = 3

CONTAINER_NAME     = os.getenv("CLAB_NODE", "clab-century-serf1")
CONTAINER_LOG_PATH = os.getenv("CONTAINER_LOG_PATH", "/opt/serfapp/nodes_log.txt")
HOST_LOG_DIR       = os.getenv("HOST_LOG_DIR", "./dist")
HOST_LOG_PATH      = os.path.join(HOST_LOG_DIR, "nodes_log.txt")

TIME_WINDOW_MS = float(os.getenv("TIME_WINDOW_MS", "120"))   # ms
RAM_THRESH     = float(os.getenv("RAM_THRESH", "16"))       # GB
CORE_THRESH    = float(os.getenv("CORE_THRESH", "4"))      # vCores
STOR_THRESH    = float(os.getenv("STOR_THRESH", "300"))     # GB

RADIUS_STRATEGY = (os.getenv("RADIUS_STRATEGY", "maxd")).lower()  # cal|q90|maxd|hybrid|guaranteed
RADIUS_QUANTILE = float(os.getenv("RADIUS_QUANTILE", "0.90"))
RADIUS_PAD      = float(os.getenv("RADIUS_PAD", "1.02"))
RADIUS_CELL_PAD = float(os.getenv("RADIUS_CELL_PAD", "1.0"))  # add this many cells (in real units)
SHOW_INTERVALS  = int(os.getenv("SHOW_INTERVALS", "32"))

# Safety: ensure P >= 1 for HilbertCurve
if P < 1:
    print(f"❌ HILBERT_BITS (P) must be >= 1 (got {P}).", file=sys.stderr)
    sys.exit(4)

## ==================== Per-node resources (EDIT THESE) ====================
## RAM / vCores per node:
#node_resources: Dict[str, tuple[int,int]] = {f"clab-century-serf{i}": (16, 16) for i in range(1, 27)}
#for i in range(1, 3):
#   node_resources[f"clab-century-serf{i}"] = (8, 16)
#for i in range(3, 5):
#   node_resources[f"clab-century-serf{i}"] = (16, 8)
#for i in range(5, 14):
#   node_resources[f"clab-century-serf{i}"] = (16, 16)
#for i in range(14, 17):
#   node_resources[f"clab-century-serf{i}"] = (8, 16)
#for i in range(17, 18):
#   node_resources[f"clab-century-serf{i}"] = (8, 16)
#for i in range(18, 27):
#   node_resources[f"clab-century-serf{i}"] = (16, 16)  # example override; change as you like
#
## Storage (GB) per node:
#node_storage: Dict[str, int] = {}
#for i in range(1, 7):    node_storage[f"clab-century-serf{i}"]  = 100
#for i in range(7, 14):   node_storage[f"clab-century-serf{i}"]  = 300
#for i in range(14, 21):  node_storage[f"clab-century-serf{i}"]  = 100
#for i in range(21, 27):  node_storage[f"clab-century-serf{i}"]  = 300
## ========================================================================

# ==================== Per-node resources (RANDOMIZED) ====================
# Controls: set RESOURCE_RNG_SEED to get reproducible values across runs.
_seed = os.getenv("RESOURCE_RNG_SEED")
if _seed is not None:
    try:
        random.seed(int(_seed))
    except ValueError:
        random.seed(_seed)  # allow non-int seeds too

# Generate random resources for the 26 serf nodes:
node_resources: Dict[str, tuple[int, int]] = {}
node_storage: Dict[str, int] = {}

for i in range(1, 27):
    name = f"clab-century-serf{i}"
    # RAM: 4..128 GB (step 4)
    ram_gb = 4 * random.randint(1, 32)          # 4, 8, 12, ..., 128
    # vCores: 4..16 (integer)
    cores  = random.randint(4, 16)               # 4, 5, ..., 16
    # Storage: 100..1000 GB (step 10)
    stor_gb = 10 * random.randint(10, 100)       # 100, 110, ..., 1000

    node_resources[name] = (ram_gb, cores)
    node_storage[name]   = stor_gb
# ========================================================================

# ==================== Utilities ====================

@dataclass
class Affine3:
    """Affine map between real space and integer grid [0..(2^P)-1]^3."""
    minv: np.ndarray     # shape (3,)
    scale: np.ndarray    # shape (3,)  real_to_grid: (x - min) * scale -> [0..N-1]

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

def serf_index(name: str) -> int:
    # Extract the trailing number from names like "clab-century-serf17"
    m = re.search(r"serf(\d+)$", name)
    return int(m.group(1)) if m else 10**9  # unknowns go last

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

# ==================== Log parsing ====================

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
            if not line:
                continue
            if "[COORDINATES]" in line:
                section = "coord"; continue
            if "[RTT]" in line:
                section = "rtt"; continue

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

# ==================== Radius strategies ====================

def compute_coord_radius(center: np.ndarray,
                         coords: Dict[str, np.ndarray],
                         rtts_ms: Dict[str, float],
                         time_window_ms: float,
                         self_name: str,
                         quantile: float) -> tuple[float, dict]:
    """
    Return (radius_sec, debug) per strategy:
      r_cal  = window_ms / median(RTT_ms / d_sec)
      r_q    = quantile of d_sec among nodes with RTT <= window
      r_max  = max d_sec among nodes with RTT <= window
    """
    # calibrated radius (exclude self in ratios)
    ratios = []
    for n, x in coords.items():
        if n == self_name:
            continue
        d_sec = float(np.linalg.norm(x - center))
        r_ms  = rtts_ms.get(n, None)
        if r_ms is not None and d_sec > 0:
            ratios.append(r_ms / d_sec)  # ms per second
    r_cal = time_window_ms / np.median(ratios) if ratios and np.median(ratios) > 0 else time_window_ms / 1000.0

    # distances of nodes actually within RTT window (exclude self)
    d_in = []
    for n, x in coords.items():
        if n == self_name:
            continue
        r_ms = rtts_ms.get(n, None)
        if r_ms is not None and r_ms <= time_window_ms:
            d_in.append(float(np.linalg.norm(x - center)))
    if d_in:
        r_q   = float(np.quantile(d_in, quantile))
        r_max = float(np.max(d_in))
    else:
        r_q = r_max = time_window_ms / 1000.0

    strat = RADIUS_STRATEGY
    if strat == "cal":
        r = r_cal
    elif strat == "q90":
        r = r_q
    elif strat == "maxd":
        r = r_max
    elif strat == "hybrid":
        r = max(r_cal, r_q)
    else:
        # "guaranteed" handled outside (we won't use r)
        r = r_max

    # multiplicative pad
    r *= RADIUS_PAD
    dbg = {"r_cal": r_cal, "r_q": r_q, "r_max": r_max, "radius": r,
           "pad": RADIUS_PAD, "quantile": quantile, "strategy": strat}
    return r, dbg

# ==================== Interval builders ====================

def build_coord_ball_intervals(hc: HilbertCurve,
                               aff: Affine3,
                               center_real: np.ndarray,
                               radius_real: float) -> tuple[List[Tuple[int,int]], int]:
    """
    Cover ONLY grid points whose REAL coords are inside the L2 ball.
    Return (intervals, covered_points_count).
    """
    N = aff.N
    cg = aff.to_grid(center_real)
    inv_scale = 1.0 / aff.scale
    r2 = (radius_real + 1e-15) ** 2

    # integer grid radius per axis
    r_grid = np.ceil(radius_real * aff.scale + 1e-12).astype(int)
    lo = np.maximum(cg - r_grid, 0)
    hi = np.minimum(cg + r_grid, N - 1)

    ds = []
    count = 0
    for ix in range(lo[0], hi[0] + 1):
        dx2 = ((ix - cg[0]) * inv_scale[0]) ** 2
        if dx2 > r2:
            continue
        for iy in range(lo[1], hi[1] + 1):
            dxy2 = dx2 + ((iy - cg[1]) * inv_scale[1]) ** 2
            if dxy2 > r2:
                continue
            rem2 = r2 - dxy2
            for iz in range(lo[2], hi[2] + 1):
                dz2 = ((iz - cg[2]) * inv_scale[2]) ** 2
                if dz2 <= rem2:
                    ds.append(hc.distance_from_point([int(ix), int(iy), int(iz)]))
                    count += 1
    ds = sorted(set(ds))
    return compress_intervals(ds), count

def coord_intervals_from_inwindow_nodes(hc: HilbertCurve,
                                        aff: Affine3,
                                        coords: Dict[str, np.ndarray],
                                        rtts_ms: Dict[str, float],
                                        time_window_ms: float,
                                        self_name: str) -> tuple[List[Tuple[int,int]], int]:
    """Zero-FN (wrt measured RTT): cover exactly the coord indices of nodes with RTT <= window (excluding self)."""
    ds = []
    for n, x in coords.items():
        if n == self_name:
            continue
        r_ms = rtts_ms.get(n, None)
        if r_ms is not None and r_ms <= time_window_ms:
            g = aff.to_grid(x).tolist()
            ds.append(hc.distance_from_point(g))
    ds = sorted(set(ds))
    return compress_intervals(ds), len(ds)

def build_param_intervals_full_orthant(
    hc: HilbertCurve,
    aff: Affine3,
    ram_thr: float,
    core_thr: float,
    stor_thr: float,
) -> tuple[list[tuple[int, int]], int]:
    """
    EXACT Hilbert cover of the axis-aligned orthant:
        RAM > ram_thr, CORES > core_thr, STORAGE > stor_thr
    over the entire integer grid [0..N-1]^3 produced by `aff`.

    Returns:
      (intervals, covered_points_count)

    Notes:
      - Uses strict '>' thresholds. In grid terms this is:
            g >= floor((thr - min) * scale) + 1
      - Produces a union of Hilbert intervals via dyadic tiling aligned with the curve.
      - Efficient: descends only on boundary-touching subcubes.
    """
    ND_local = 3
    N = aff.N  # = 2^P

    # ---- Convert real thresholds to integer grid lower bounds (strict >)
    thr_real = np.array([ram_thr, core_thr, stor_thr], dtype=float)
    gthr = np.floor((thr_real - aff.minv) * aff.scale + 1e-12).astype(int)
    lower = np.clip(gthr + 1, 0, N)             # inclusive lower bounds per axis
    upper = np.array([N - 1, N - 1, N - 1])     # inclusive upper bounds per axis

    # Empty orthant check
    span = np.maximum(upper - lower + 1, 0)
    covered_points = int(np.prod(span))
    if covered_points <= 0:
        return [], 0

    # ---- Hierarchy parameters from 'hc'
    Pbits = hc.p
    if Pbits < 1:
        raise ValueError(f"Hilbert 'p' must be >= 1 for full-orthant cover (got p={Pbits}).")
    if hc.n != ND_local:
        raise ValueError(f"Hilbert dimension mismatch: expected {ND_local}, got {hc.n}")

    P_local = Pbits
    # Level L partitions space into (2^L)^ND blocks of side s = 2^(P-L).
    # Each block is a contiguous Hilbert run of length block_size = s^ND.
    side_by_level = [1 << (P_local - L) for L in range(P_local + 1)]               # s(L)
    block_size_by_level = [1 << (ND_local * (P_local - L)) for L in range(P_local + 1)]   # s(L)^ND
    CHILD_COUNT = 1 << ND_local  # 8 for 3D

    # Build curves only for L >= 1 (avoid p=0)
    hc_levels: list[HilbertCurve | None] = [None] + [HilbertCurve(L, ND_local) for L in range(1, P_local + 1)]

    intervals: list[tuple[int, int]] = []

    def block_box(L: int, dL: int) -> tuple[np.ndarray, np.ndarray]:
        """Grid AABB [origin .. origin+s-1] for level-L block with coarse-Hilbert id dL."""
        s = side_by_level[L]
        if L == 0:
            # single top-level block covering entire grid
            origin = np.array([0, 0, 0], dtype=int)
        else:
            # coarse block coordinates in [0..2^L-1]^3, in Hilbert order
            b = np.array(hc_levels[L].point_from_distance(dL), dtype=int)  # type: ignore[index]
            origin = b * s
        hi = origin + (s - 1)
        return origin, hi

    def overlap_state(origin: np.ndarray, hi: np.ndarray) -> str:
        """Classify block vs query box [lower..upper]: 'out' | 'in' | 'partial'."""
        if np.any(origin > upper) or np.any(hi < lower):
            return "out"
        if np.all(origin >= lower) and np.all(hi <= upper):
            return "in"
        return "partial"

    # Recurse: global start index of a level-L block is dL * block_size_by_level[L]
    def descend(L: int, dL: int):
        blk = block_size_by_level[L]
        origin, hi = block_box(L, dL)
        st = overlap_state(origin, hi)

        if st == "out":
            return
        if st == "in":
            start = dL * blk
            intervals.append((start, start + blk - 1))
            return
        # partial
        if L == P_local:
            # base voxel
            start = dL * blk  # blk == 1
            intervals.append((start, start))
            return
        base_child = dL * CHILD_COUNT  # children occupy a contiguous group
        for c in range(CHILD_COUNT):
            descend(L + 1, base_child + c)

    # Start recursion at top-level block
    descend(0, 0)

    # Merge adjacent intervals
    intervals.sort()
    merged: list[tuple[int, int]] = []
    for a, b in intervals:
        if not merged or a > merged[-1][1] + 1:
            merged.append((a, b))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))

    return merged, covered_points

# ==================== Printing ====================

def show_node_raw(name: str,
                  coords: Dict[str, np.ndarray],
                  rtts: Dict[str, float],
                  ram: Dict[str, int],
                  cores: Dict[str, int],
                  stor: Dict[str, int]):
    """Print original (raw) values only."""
    x, y, z = coords[name].tolist()
    rtt = rtts.get(name, float("nan"))
    print(f"{name:<20} X={x:+.6f} Y={y:+.6f} Z={z:+.6f}  RTT={rtt:6.1f}ms  "
          f"RAM={ram[name]:<4} Cores={cores[name]:<4} Storage={stor[name]:<4}")

def show_node(name: str,
              coords: Dict[str, np.ndarray],
              rtts: Dict[str, float],
              ram: Dict[str, int],
              cores: Dict[str, int],
              stor: Dict[str, int],
              coord_idx: Dict[str, int],
              param_idx: Dict[str, int]):
    """Print raw values plus Hilbert indices (used in results sections)."""
    x, y, z = coords[name].tolist()
    rtt = rtts.get(name, 0.0)
    print(f"{name:<20} coord=({x:+.3f},{y:+.3f},{z:+.3f})  RTT={rtt:6.1f}ms  "
          f"RAM={ram[name]:<3}  Cores={cores[name]:<3}  Stor={stor[name]:<4}  "
          f"CoordIdx={coord_idx[name]:<8}  ParamIdx={param_idx[name]:<8}")

def print_intervals(title: str, intervals: List[Tuple[int,int]], total_points: int | None = None,
                    domain_total: int | None = None, max_lines: int = SHOW_INTERVALS):
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

# ==================== Main ====================

def main():
    # Header
    print(f"🔧 Query parameters: window={int(TIME_WINDOW_MS)}ms, "
          f"RAM>{int(RAM_THRESH)}GB, Cores>{int(CORE_THRESH)}, Storage>{int(STOR_THRESH)}GB, P={P}")
    print(f"🚀 Running on node: {CONTAINER_NAME}")

    if copy_log_from_container():
        print(f"✅ Copied log to {HOST_LOG_PATH}")
    else:
        print(f"⚠️ Could not docker cp; trying local file at {HOST_LOG_PATH}")

    coords, rtts, current = parse_log(HOST_LOG_PATH)
    if current is None:
        current = CONTAINER_NAME
    print(f"✅ Parsed {len(coords)} coords and {len(rtts)} RTTs")

    # Ensure RTT 0 for self (helps display)
    rtts[current] = 0.0

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

    # ===== Original data dump (serf1..serf26 order) =====
    print("\n=== Original Data (all nodes) ===")
    nodes_numeric = sorted(nodes, key=serf_index)
    for n in nodes_numeric:
        show_node_raw(n, coords, rtts, ram, cores, stor)

    # Affine maps (coords + resources)
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

    # Hilbert curves
    hc_coord = HilbertCurve(P, ND)
    hc_param = HilbertCurve(P, ND)

    # Per-node indices (for printing)
    coord_idx, param_idx = {}, {}
    for n in nodes:
        coord_idx[n] = hc_coord.distance_from_point(coord_aff.to_grid(coords[n]).tolist())
        param_idx[n] = hc_param.distance_from_point(
            param_aff.to_grid(np.array([ram[n], cores[n], stor[n]], float)).tolist()
        )

    # ===== Stage A: coord intervals =====
    center = coords[current].astype(float)

    if RADIUS_STRATEGY == "guaranteed":
        coord_intervals, coord_point_count = coord_intervals_from_inwindow_nodes(
            hc_coord, coord_aff, coords, rtts, TIME_WINDOW_MS, current
        )
        print("ℹ️  coord radius strategy: guaranteed (zero-FN wrt measured RTT; excludes self)")
    else:
        radius_real, dbg = compute_coord_radius(center, coords, rtts, TIME_WINDOW_MS, current, RADIUS_QUANTILE)

        # Optional extra padding in REAL units (add n * cell_size)
        if RADIUS_CELL_PAD != 0.0:
            cell = 1.0 / np.maximum(coord_aff.scale, 1e-12)  # per-axis cell size in real units
            radius_real += RADIUS_CELL_PAD * float(np.max(cell))

        coord_intervals, coord_point_count = build_coord_ball_intervals(
            hc_coord, coord_aff, center_real=center, radius_real=radius_real
        )
        print(f"ℹ️  radii: cal={dbg['r_cal']:.6f}s  q{int(100*dbg['quantile'])}={dbg['r_q']:.6f}s  "
              f"max={dbg['r_max']:.6f}s  → chosen={radius_real:.6f}s "
              f"({dbg['strategy']} * pad {RADIUS_PAD} + cell_pad {RADIUS_CELL_PAD})")

    # ===== Stage B: param intervals (FULL ORTHANT, strict >) =====
    param_intervals, param_point_count = build_param_intervals_full_orthant(
        hc_param, param_aff,
        RAM_THRESH, CORE_THRESH, STOR_THRESH
    )

    # ===== Selection (pure index), excluding the current node =====
    matches = [n for n in nodes
               if n != current
               and index_in_intervals(coord_idx[n], coord_intervals)
               and index_in_intervals(param_idx[n], param_intervals)]
    matches.sort()

    # ===== Ground truth (evaluation only), excluding the current node =====
    gt = []
    for n in nodes:
        if n == current:
            continue
        metric_ms = rtts.get(n, float(np.linalg.norm(coords[n] - center) * 1000.0))
        if (metric_ms <= TIME_WINDOW_MS) and (ram[n] > RAM_THRESH) and (cores[n] > CORE_THRESH) and (stor[n] > STOR_THRESH):
            gt.append(n)
    gt.sort()

    fp = sorted(set(matches) - set(gt))
    fn = sorted(set(gt) - set(matches))

    # ===== Prints =====
    domain_total = (1 << (3 * P))
    print_intervals("Coord Intervals", coord_intervals, total_points=coord_point_count, domain_total=domain_total)
    # For param intervals we don't pass domain_total because it's the *coord* domain; the param domain is the same size if the same P, but coverage percentage of the orthant might be huge.
    print_intervals("Param Intervals", param_intervals, total_points=param_point_count)

    print(f"\n=== Ground Truth (count={len(gt)}) ===")
    for n in gt:
        show_node(n, coords, rtts, ram, cores, stor, coord_idx, param_idx)

    print(f"\n=== Pure-Hilbert Matches (count={len(matches)}) ===")
    for n in matches:
        show_node(n, coords, rtts, ram, cores, stor, coord_idx, param_idx)

    print(f"\n=== False Positives (count={len(fp)}) ===")
    if fp:
        for n in fp:
            show_node(n, coords, rtts, ram, cores, stor, coord_idx, param_idx)
    else:
        print("  (none)")

    print(f"\n=== False Negatives (count={len(fn)}) ===")
    if fn:
        for n in fn:
            show_node(n, coords, rtts, ram, cores, stor, coord_idx, param_idx)
    else:
        print("  (none)")

if __name__ == "__main__":
    main()
