#!/usr/bin/env python3
import os
import re
import subprocess
import random
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
P = 6  # bits per dimension for Hilbert transform

CONTAINER_NAME     = "clab-century-serf1"
CONTAINER_LOG_PATH = "/opt/serfapp/nodes_log.txt"
HOST_LOG_DIR       = "./dist"
HOST_LOG_PATH      = os.path.join(HOST_LOG_DIR, "nodes_log.txt")

# All nodes default to (RAM GB, vCores)
node_resources = {f"clab-century-serf{i}": (16, 16) for i in range(1, 27)}
for i in range(1, 14):
    node_resources[f"clab-century-serf{i}"] = (16, 16)
for i in range(14, 27):
    node_resources[f"clab-century-serf{i}"] = (8, 8)

# Generate pseudo storage data (100GB to 1000GB)
node_storage = {node: random.randint(100, 1000) for node in node_resources}

# ─────────────────────────────────────────────────────────────────────────────
# 1) Copy & Parse the Serf Log for Coordinates
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


def parse_coords():
    coords = {}
    section = None
    if not os.path.isfile(HOST_LOG_PATH):
        raise FileNotFoundError(f"{HOST_LOG_PATH} not found")
    with open(HOST_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if "[COORDINATES]" in line:
                section = "coord"; continue
            if "[RTT]" in line:
                section = None; continue
            if section != "coord" or "Node:" not in line:
                continue
            m = re.match(r".*Node:\s*(\S+)\s*=>\s*(.*)", line)
            if not m:
                continue
            node, rest = m.groups()
            cm = re.search(r"X:\s*([\-\d.]+)\s*Y:\s*([\-\d.]+)\s*Z:\s*([\-\d.]+)", rest)
            if cm:
                x, y, z = map(float, cm.groups())
                coords[node] = (x, y, z)
    print(f"✅ Parsed {len(coords)} nodes with coordinates")
    return coords

# ─────────────────────────────────────────────────────────────────────────────
# 2) Hilbert Transform and Decode
# ─────────────────────────────────────────────────────────────────────────────
def hilbert_transform_and_decode(data_map, bits, dims):
    nodes = list(data_map.keys())
    arr = np.array([data_map[n] for n in nodes], dtype=float)

    # normalize to integer grid
    minv = arr.min(axis=0)
    maxv = arr.max(axis=0)
    diff = maxv - minv
    scale = np.empty_like(diff)
    nz = diff != 0
    scale[nz] = (2**bits - 1) / diff[nz]
    scale[~nz] = 1.0
    norm = (arr - minv) * scale
    norm[:, ~nz] = 0
    pts = np.rint(norm).astype(int)

    hc = HilbertCurve(bits, dims)
    distances = {}
    for i, node in enumerate(nodes):
        distances[node] = hc.distance_from_point(pts[i].tolist())

    decoded_map = {}
    for node, dist in distances.items():
        pt_dec = hc.point_from_distance(dist)
        real = [pt_dec[d] / scale[d] + minv[d] for d in range(dims)]
        decoded_map[node] = tuple(real)

    return distances, decoded_map

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"🚀 Running on node: {CONTAINER_NAME}")
    copy_log()
    coords = parse_coords()

    # Compute transforms
    coord_dists, _ = hilbert_transform_and_decode(coords, P, 3)
    resources = {node: (*node_resources[node], node_storage[node]) for node in node_resources}
    res_dists, _ = hilbert_transform_and_decode(resources, P, 3)

    # Write results to file
    output_file = 'nodes_data.txt'
    with open(output_file, 'w') as f:
        f.write('name,x,y,z,Coord Hilbert,RAM,Cores,Storage,Param Hilbert,Current Node\n')
        for node in sorted(coords):
            x, y, z = coords[node]
            coord_val = coord_dists.get(node, '')
            ram, cores = node_resources.get(node, ('', ''))
            storage = node_storage.get(node, '')
            param_val = res_dists.get(node, '')
            f.write(f"{node},{x},{y},{z},{coord_val},{ram},{cores},{storage},{param_val},{CONTAINER_NAME}\n")
    print(f"✅ Written Hilbert results to {output_file}")

if __name__ == "__main__":
    main()
