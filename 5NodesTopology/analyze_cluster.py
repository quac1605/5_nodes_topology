import os
import re
import subprocess
import numpy as np
import time
from hilbertcurve.hilbertcurve import HilbertCurve

# Hilbert Curve config
P = 16  # precision
N = 5   # dimensions

# Hardcoded pseudo resource data (RAM in GB, vCores)
node_resources = {
    "clab-century-serf1":  (8, 8),
    "clab-century-serf2":  (8, 8),
    "clab-century-serf3":  (8, 8),
    "clab-century-serf4":  (8, 8),
    "clab-century-serf5":  (8, 8),
    "clab-century-serf6":  (8, 8),
    "clab-century-serf7":  (8, 8),
    "clab-century-serf8":  (8, 8),
    "clab-century-serf9":  (8, 8),
    "clab-century-serf10": (8, 8),
    "clab-century-serf11": (8, 8),
    "clab-century-serf12": (8, 8),
    "clab-century-serf13": (8, 8),
    "clab-century-serf14": (8, 8),
    "clab-century-serf15": (8, 8),
    "clab-century-serf16": (8, 8),
    "clab-century-serf17": (8, 8),
    "clab-century-serf18": (8, 8),
    "clab-century-serf19": (8, 8),
    "clab-century-serf20": (8, 8),
    "clab-century-serf21": (8, 8),
    "clab-century-serf22": (8, 8),
    "clab-century-serf23": (8, 8),
    "clab-century-serf24": (8, 8),
    "clab-century-serf25": (8, 8),
    "clab-century-serf26": (8, 8),
}

# Paths and current node identifier
CONTAINER_NAME    = "clab-century-serf1"
CONTAINER_LOG_PATH= "/opt/serfapp/nodes_log.txt"
HOST_LOG_DIR      = "./dist"
HOST_LOG_PATH     = os.path.join(HOST_LOG_DIR, "nodes_log.txt")
SUMMARY_FILE      = "nodes_data.txt"

def copy_log_from_container():
    os.makedirs(HOST_LOG_DIR, exist_ok=True)
    try:
        subprocess.run([
            "docker", "cp",
            f"{CONTAINER_NAME}:{CONTAINER_LOG_PATH}", HOST_LOG_PATH
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"[Error] Could not copy log: {e}")
        return False
    return True

def normalize_coordinates(coords, bits):
    arr  = np.array(list(coords.values()))
    minv = arr.min(axis=0)
    maxv = arr.max(axis=0)
    diff = maxv - minv
    scale = np.where(diff == 0, 1.0, (2**bits - 1)/diff)
    norm = (arr - minv) * scale
    norm[:, diff==0] = 0
    return {n: tuple(norm[i].astype(int)) for i,n in enumerate(coords)}, minv, maxv

def denormalize(norm_pt, minv, maxv, bits):
    scale = (maxv - minv)/(2**bits-1)
    return norm_pt*scale + minv

def parse_log():
    coords, rtts = {}, {}
    section = None
    if not os.path.isfile(HOST_LOG_PATH):
        return coords, rtts
    with open(HOST_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if '[COORDINATES]' in line:
                section = 'coord'; continue
            if '[RTT]' in line:
                section = 'rtt'; continue
            if 'Node:' not in line:
                continue
            m = re.match(r'.*Node:\s*(\S+)\s*=>\s*(.*)', line)
            if not m:
                continue
            node, data = m.groups()
            if section == 'coord':
                cm = re.search(r'X:\s*([-\d.]+)\s*Y:\s*([-\d.]+)\s*Z:\s*([-\d.]+)', data)
                if cm and node in node_resources:
                    x,y,z = cm.groups()
                    ram, vcores = node_resources[node]
                    coords[node] = (float(x), float(y), float(z), ram, vcores)
            elif section == 'rtt':
                rm = re.search(r'RTT:\s*([\d.]+)\s*ms', data)
                if rm:
                    rtts[node] = float(rm.group(1))
    return coords, rtts

def compute_hilbert_distances(hilb, current_node):
    """
    Returns a dict mapping each node to the absolute difference
    between its Hilbert index and the current node’s index.
    """
    curr_h = hilb.get(current_node, 0)
    return {node: abs(h - curr_h) for node, h in hilb.items()}

def write_summary_to_file(hilb, distances, coords, rtts, current_node):
    """
    Writes node, RTT, RAM, vCores, Hilbert index, HilbertDist and
    marks the current node. Sorted by ascending RTT.
    """
    sorted_nodes = sorted(coords.keys(), key=lambda n: rtts.get(n, float('inf')))
    with open(SUMMARY_FILE, 'w') as f:
        f.write("# node RTT(ms) RAM(GB) vCores Hilbert HilbertDist Current\n")
        for node in sorted_nodes:
            rtt      = rtts.get(node, 0.0)
            _,_,_,ram,cores = coords[node]
            hval     = hilb.get(node, 0)
            hdist    = distances.get(node, 0)
            marker   = 'current' if node == current_node else 'not'
            f.write(f"{node} {rtt:.2f} {ram} {cores} {hval} {hdist} {marker}\n")

def print_console(coords, rtts, hilb, recon):
    print(f"\n--- Run at {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    print("RTT Results:")
    for n, r in sorted(rtts.items(), key=lambda x: x[1]):
        print(f"  {n:<25} => {r:.2f} ms")
    print("\nHilbert & Decoded 5D Points:")
    for n in sorted(hilb):
        h = hilb[n]
        ox,oy,oz,ram,cores = coords[n]
        dx,dy,dz,dram,dcores = recon[n]
        print(
            f"Node {n}: H={h}\n"
            f"  Orig => (X={ox:.6f}, Y={oy:.6f}, Z={oz:.6f}, RAM={ram}GB, Cores={cores})\n"
            f"  Dec  => (X={dx:.6f}, Y={dy:.6f}, Z={dz:.6f}, RAM={dram:.2f}GB, Cores={dcores:.2f})\n"
        )

def process():
    if not copy_log_from_container():
        print("[Error] Log copy failed.")
        return

    coords, rtts = parse_log()
    if not coords:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ No data.")
        return

    norm, minv, maxv = normalize_coordinates(coords, P)
    hc = HilbertCurve(P, N)
    hilb, recon = {}, {}
    for n in coords:
        h = int(hc.distance_from_point(list(norm[n])))
        hilb[n] = h
        dec_norm = hc.point_from_distance(h)
        recon[n] = denormalize(np.array(dec_norm), minv, maxv, P)

    # compute distances in Hilbert space from current node
    distances = compute_hilbert_distances(hilb, CONTAINER_NAME)

    # write full summary with RTT sort, current-node mark, and HilbertDist
    write_summary_to_file(hilb, distances, coords, rtts, CONTAINER_NAME)
    print(f"[Info] Written detailed summary to {SUMMARY_FILE}")

    # also print detailed info to console
    print_console(coords, rtts, hilb, recon)

if __name__ == '__main__':
    try:
        while True:
            process()
            time.sleep(3.5)
    except KeyboardInterrupt:
        print("Stopped by user.")
