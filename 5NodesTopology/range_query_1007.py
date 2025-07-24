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
P = 16  # bits per dim
N = 5    # dims: X, Y, Z, RAM, vCores

CONTAINER_NAME     = "clab-century-serf1"
CONTAINER_LOG_PATH = "/opt/serfapp/nodes_log.txt"
HOST_LOG_DIR       = "./dist"
HOST_LOG_PATH      = os.path.join(HOST_LOG_DIR, "nodes_log.txt")

# All nodes default to (8 GB, 8 vCores)
node_resources = {f"clab-century-serf{i}": (8, 8) for i in range(1, 27)}
# Override RAM for serf5–serf8 → 4 GB
for i in range(5, 9):
    node_resources[f"clab-century-serf{i}"] = (4, node_resources[f"clab-century-serf{i}"][1])
# Override vCores for serf9–serf13 → 4 cores
for i in range(9, 14):
    node_resources[f"clab-century-serf{i}"] = (node_resources[f"clab-century-serf{i}"][0], 4)
def copy_log_from_container():
    os.makedirs(HOST_LOG_DIR, exist_ok=True)
    try:
        subprocess.run([
            "docker", "cp",
            f"{CONTAINER_NAME}:{CONTAINER_LOG_PATH}",
            HOST_LOG_PATH
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"[Error] {e}")
        return False
    return True

def parse_log():
    coords, rtts = {}, {}
    section = None
    if not os.path.isfile(HOST_LOG_PATH):
        raise FileNotFoundError(HOST_LOG_PATH)
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
                    x, y, z = map(float, cm.groups())
                    ram, cores = node_resources[node]
                    coords[node] = (x, y, z, ram, cores)
            else:  # rtt
                rm = re.search(r"RTT:\s*([\d.]+)\s*ms", rest)
                if rm:
                    rtts[node] = float(rm.group(1))
    # ensure current node has RTT=0
    rtts[CONTAINER_NAME] = 0.0
    return coords, rtts

# ─────────────────────────────────────────────────────────────────────────────
# 2) Normalize & 5‑D Hilbert index
# ─────────────────────────────────────────────────────────────────────────────
def normalize_and_index(coords, bits):
    arr  = np.array(list(coords.values()), dtype=float)  # (n,5)
    minv = arr.min(axis=0)
    maxv = arr.max(axis=0)
    diff = maxv - minv

    scale = np.empty_like(diff)
    nz    = diff != 0
    scale[nz] = (2**bits - 1) / diff[nz]
    scale[~nz] = 1.0

    norm = (arr - minv) * scale
    norm[:, ~nz] = 0

    hc = HilbertCurve(bits, N)
    hilb_map = {}
    for i, node in enumerate(coords):
        pt = tuple(int(round(v)) for v in norm[i])
        hilb_map[node] = hc.distance_from_point(list(pt))

    sorted_arr = sorted((h, n) for n, h in hilb_map.items())
    hil_list   = [h for h, _ in sorted_arr]
    return hilb_map, sorted_arr, hil_list, minv, scale

# ─────────────────────────────────────────────────────────────────────────────
# 3) Sliding‐window range query on the Hilbert list
# ─────────────────────────────────────────────────────────────────────────────
def range_query_sliding(
    current_node, coords, rtts,
    hilb_map, sorted_arr, hil_list,
    rtt_thresh, ram_min, cores_min
):
    # helper: does node satisfy all 3 real criteria?
    def in_box(n):
        _, _, _, ram, cores = coords[n]
        return (
            rtts.get(n, float('inf')) <= rtt_thresh and
            ram   >= ram_min and
            cores >= cores_min
        )

    # locate current node in the sorted hil_list
    h0   = hilb_map[current_node]
    idx0 = bisect.bisect_left(hil_list, h0)

    candidates = set()
    # include the current node if it qualifies
    if in_box(current_node):
        candidates.add(current_node)

    left, right = idx0, idx0
    while True:
        moved = False
        # try right neighbor
        if right + 1 < len(hil_list):
            _, nr = sorted_arr[right + 1]
            if in_box(nr):
                candidates.add(nr)
                right += 1
                moved = True
        # try left neighbor
        if left - 1 >= 0:
            _, nl = sorted_arr[left - 1]
            if in_box(nl):
                candidates.add(nl)
                left -= 1
                moved = True
        if not moved:
            break

    return sorted(candidates)

# ─────────────────────────────────────────────────────────────────────────────
# 4) Main – tie it all together, compute FP & FN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not copy_log_from_container():
        exit(1)

    coords, rtts = parse_log()
    hilb_map, sorted_arr, hil_list, minv, scale = normalize_and_index(coords, P)

    # print all nodes for context
    print("\nAll nodes data:")
    for n in sorted(coords):
        x, y, z, ram, cores = coords[n]
        print(f"{n:25s} RTT={rtts[n]:6.2f}ms  RAM={ram:2d}GB  Cores={cores:2d}  H={hilb_map[n]}")

    # perform range query
    thresh, rmin, cmin = 30.0, 8, 8
    result = range_query_sliding(
        CONTAINER_NAME, coords, rtts,
        hilb_map, sorted_arr, hil_list,
        rtt_thresh=thresh, ram_min=rmin, cores_min=cmin
    )
    print(f"\nRange‐query result: {len(result)} nodes → {result}")

    # compute ground‑truth set
    true_set = sorted([
        n for n in coords
        if rtts.get(n, float('inf')) <= thresh
        and coords[n][3] >= rmin
        and coords[n][4] >= cmin
    ])
    print(f"\nGround truth       : {len(true_set)} nodes → {true_set}")

    # false positives and false negatives
    set_res  = set(result)
    set_true = set(true_set)
    fp = sorted(set_res - set_true)
    fn = sorted(set_true - set_res)

    print(f"\nFalse Positives ({len(fp)}): {fp}")
    print(f"False Negatives ({len(fn)}): {fn}")
