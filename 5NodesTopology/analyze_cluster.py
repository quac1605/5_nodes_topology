import re
import subprocess
import numpy as np
import time
from hilbertcurve.hilbertcurve import HilbertCurve

# Hilbert Curve config
P = 16  # precision
N = 5   # dimensions

def copy_log_from_container(container_name, container_path, host_path):
    try:
        subprocess.run(
            ["docker", "cp", f"{container_name}:{container_path}", host_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError as e:
        print(f"Error copying file from Docker container: {e}")
        exit(1)

def normalize_coordinates(coords, bits):
    coords_array = np.array(list(coords.values()))
    min_vals = coords_array.min(axis=0)
    max_vals = coords_array.max(axis=0)
    scale = (2**bits - 1) / (max_vals - min_vals)
    normalized = ((coords_array - min_vals) * scale).astype(int)
    return (
        {node: tuple(normalized[i]) for i, node in enumerate(coords)},
        min_vals,
        max_vals,
    )

def denormalize(norm_point, min_vals, max_vals, bits):
    scale = (max_vals - min_vals) / (2**bits - 1)
    return norm_point * scale + min_vals

def parse_log(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    coordinates = {}
    rtts = {}
    current_section = None

    for line in lines:
        line = line.strip()

        if '[COORDINATES + RESOURCES]' in line:
            current_section = 'coordinates'
        elif '[RTT]' in line:
            current_section = 'rtt'
        elif 'Node:' in line:
            match = re.match(r'.*Node: (\S+)\s+=> (.+)', line)
            if not match:
                continue
            node, data = match.groups()

            if current_section == 'coordinates':
                match = re.search(
                    r'X:\s*([-\d.]+)\s*Y:\s*([-\d.]+)\s*Z:\s*([-\d.]+)\s*RAM:\s*(\d+)GB\s*vCores:\s*(\d+)',
                    data
                )
                if match:
                    x, y, z, ram, vcores = match.groups()
                    coordinates[node] = (
                        float(x), float(y), float(z), int(ram), int(vcores)
                    )

            elif current_section == 'rtt':
                rtt_match = re.search(r'RTT:\s*([\d.]+)\s*ms', data)
                if rtt_match:
                    rtts[node] = float(rtt_match.group(1))

    return coordinates, rtts

def process():
    container_name = "clab-century-serf1"
    container_path = "/opt/serfapp/nodes_log.txt"
    host_path = "./dist/nodes_log.txt"

    copy_log_from_container(container_name, container_path, host_path)

    coords, rtts = parse_log(host_path)

    if not coords:
        print(f"[{time.strftime('%H:%M:%S')}] ⚠️ No coordinate data found in the log file. Skipping this run.")
        return

    norm_coords, min_vals, max_vals = normalize_coordinates(coords, P)
    hilbert_curve = HilbertCurve(P, N)

    hilbert_values = {}
    recon_norm = {}
    recon_coords = {}

    for node, point in coords.items():
        norm_point = norm_coords[node]
        hilbert_val = int(hilbert_curve.distance_from_point(list(norm_point)))
        decoded_norm = hilbert_curve.point_from_distance(hilbert_val)
        decoded_coord = denormalize(np.array(decoded_norm), min_vals, max_vals, P)

        hilbert_values[node] = hilbert_val
        recon_norm[node] = decoded_norm
        recon_coords[node] = decoded_coord

    current_node = "clab-century-serf1"
    if current_node not in hilbert_values:
        print(f"[{time.strftime('%H:%M:%S')}] ⚠️ Current node data not found in parsed coordinates.")
        return

    current_hilbert = hilbert_values[current_node]

    print(f"\n--- Run at {time.strftime('%H:%M:%S')} ---")
    print(f"Current Node: {current_node}\n")

    # RTT Distance
    print("1. Distance through Round Trip Time (ms):")
    sorted_rtt = sorted(
        [(n, rtt) for n, rtt in rtts.items() if n != current_node],
        key=lambda x: x[1]
    )
    for node, rtt in sorted_rtt:
        print(f"   {node:<25} => {rtt:.2f} ms")
    print()

    # Hilbert Distance
    print("2. Distance with Hilbert 1D Transform:")
    sorted_hilbert = sorted(
        [(n, abs(hilbert_values[n] - current_hilbert)) for n in coords if n != current_node],
        key=lambda x: x[1]
    )
    for node, dist in sorted_hilbert:
        hv = hilbert_values[node]
        dn = recon_norm[node]
        dc = recon_coords[node]
        ox, oy, oz, ram, cores = coords[node]
        print(
            f"   {node:<25} => Hilbert1D: {hv:<10} HilbertDist: {dist:<10} "
            f"Decoded(X,Y,Z,RAM,vCores): ({dc[0]:.6f}, {dc[1]:.6f}, {dc[2]:.6f}, {dc[3]:.2f}, {dc[4]:.2f}) "
            f"Original: ({ox:.6f}, {oy:.6f}, {oz:.6f}, {ram}, {cores})"
        )


if __name__ == "__main__":
    try:
        while True:
            process()
            time.sleep(3.5)
    except KeyboardInterrupt:
        print("\nStopped by user.")
