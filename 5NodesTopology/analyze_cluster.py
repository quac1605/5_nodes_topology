import re
import subprocess
import numpy as np
import time
from hilbertcurve.hilbertcurve import HilbertCurve

# Hilbert Curve config
P = 14  # precision
N = 5   # dimensions

# Hardcoded pseudo resource data
node_resources = {
    "clab-century-serf1":  (16, 4),
    "clab-century-serf2":  (32, 8),
    "clab-century-serf3":  (8, 2),
    "clab-century-serf4":  (64, 12),
    "clab-century-serf5":  (16, 6),
    "clab-century-serf6":  (32, 10),
    "clab-century-serf7":  (8, 2),
    "clab-century-serf8":  (64, 16),
    "clab-century-serf9":  (16, 4),
    "clab-century-serf10": (32, 6),
    "clab-century-serf11": (8, 2),
    "clab-century-serf12": (64, 14),
    "clab-century-serf13": (16, 4),
    "clab-century-serf14": (32, 8),
    "clab-century-serf15": (8, 2),
    "clab-century-serf16": (64, 12),
    "clab-century-serf17": (16, 6),
    "clab-century-serf18": (32, 8),
    "clab-century-serf19": (8, 2),
    "clab-century-serf20": (64, 16),
    "clab-century-serf21": (16, 6),
    "clab-century-serf22": (32, 10),
    "clab-century-serf23": (8, 2),
    "clab-century-serf24": (64, 14),
    "clab-century-serf25": (16, 4),
    "clab-century-serf26": (32, 6),
}

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

        if '[COORDINATES]' in line:
            current_section = 'coordinates'
        elif '[RTT]' in line:
            current_section = 'rtt'
        elif 'Node:' in line:
            match = re.match(r'.*Node: (\S+)\s+=> (.+)', line)
            if not match:
                continue
            node, data = match.groups()

            if current_section == 'coordinates':
                c_match = re.search(r'X:\s*([\-\d.]+)\s*Y:\s*([\-\d.]+)\s*Z:\s*([\-\d.]+)', data)
                if c_match and node in node_resources:
                    x, y, z = c_match.groups()
                    ram, vcores = node_resources[node]
                    coordinates[node] = (float(x), float(y), float(z), ram, vcores)

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

    # Normalize and compute Hilbert indices
    norm_coords, min_vals, max_vals = normalize_coordinates(coords, P)
    hilbert_curve = HilbertCurve(P, N)

    hilbert_values = {}
    recon_norm = {}
    recon_coords = {}
    for node, point in coords.items():
        norm_point = norm_coords[node]
        h = int(hilbert_curve.distance_from_point(list(norm_point)))
        hilbert_values[node] = h
        decoded_norm = hilbert_curve.point_from_distance(h)
        recon_norm[node] = decoded_norm
        recon_coords[node] = denormalize(np.array(decoded_norm), min_vals, max_vals, P)

    # Print header
    print(f"\n--- Run at {time.strftime('%H:%M:%S')} ---")
    print(f"Current Node: clab-century-serf1\n")

    # RTT results
    print("1. Distance through Round Trip Time (ms):")
    sorted_rtt = sorted(
        [(n, rtt) for n, rtt in rtts.items() if n != 'clab-century-serf1'],
        key=lambda x: x[1]
    )
    for node, rtt in sorted_rtt:
        print(f"   {node:<25} => {rtt:.2f} ms")

    # Raw Hilbert index and decoded coordinates
    print("\n2. Raw Hilbert index & decoded point (X,Y,Z,RAM,vCores):")
    for node in sorted(hilbert_values):
        h = hilbert_values[node]
        orig = coords[node]
        dec = recon_coords[node]
        print(
            f"   {node:<25} => H={h:<10} Decoded=({dec[0]:.6f}, {dec[1]:.6f}, {dec[2]:.6f}, {dec[3]:.2f}, {dec[4]:.2f}) "
            f"Original=({orig[0]:.6f}, {orig[1]:.6f}, {orig[2]:.6f}, {orig[3]}, {orig[4]})"
        )

if __name__ == "__main__":
    try:
        while True:
            process()
            time.sleep(3.5)
    except KeyboardInterrupt:
        print("\nStopped by user.")
