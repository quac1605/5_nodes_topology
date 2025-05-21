import re
import json
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve

# Hilbert Curve config
P = 16  # precision
N = 2   # dimensions

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
    pings = {}
    current_section = None

    for line in lines:
        line = line.strip()

        if '[COORDINATES]' in line:
            current_section = 'coordinates'
        elif '[RTT]' in line:
            current_section = 'rtt'
        elif '[PING]' in line:
            current_section = 'ping'
        elif 'Node:' in line:
            match = re.match(r'.*Node: (\S+)\s+=> (.+)', line)
            if not match:
                continue
            node, data = match.groups()

            if current_section == 'coordinates':
                coord_match = re.findall(r'X:\s*([-\d.]+)\s*Y:\s*([-\d.]+)', data)
                if coord_match:
                    x, y = map(float, coord_match[0])
                    coordinates[node] = (x, y)

            elif current_section == 'rtt':
                rtt_match = re.search(r'RTT:\s*([\d.]+)\s*ms', data)
                if rtt_match:
                    rtts[node] = float(rtt_match.group(1))

            elif current_section == 'ping':
                ping_match = re.search(r'time=([\d.]+)\s*ms', data)
                if ping_match:
                    pings[node] = float(ping_match.group(1))

    return coordinates, rtts, pings

def main():
    filename = 'nodes_log.txt'
    coords, rtts, pings = parse_log(filename)
    norm_coords, min_vals, max_vals = normalize_coordinates(coords, P)
    hilbert_curve = HilbertCurve(P, N)

    results = {}
    for node, point in coords.items():
        norm_point = norm_coords[node]
        hilbert_val = int(hilbert_curve.distance_from_point(list(norm_point)))
        recon_norm = hilbert_curve.point_from_distance(hilbert_val)
        recon_orig = denormalize(np.array(recon_norm), min_vals, max_vals, P)

        results[node] = {
            'coordinates': {'x': float(point[0]), 'y': float(point[1])},
            'normalized_coords': {'x': int(norm_point[0]), 'y': int(norm_point[1])},
            'hilbert_index': int(hilbert_val),
            'reconstructed_norm': {'x': int(recon_norm[0]), 'y': int(recon_norm[1])},
            'reconstructed_coords': {'x': float(recon_orig[0]), 'y': float(recon_orig[1])},
            'rtt_ms': float(rtts[node]) if node in rtts else None,
            'ping_ms': float(pings[node]) if node in pings else None
        }

    # Print header
    print(f"{'Node':<30} {'Orig-X':>10} {'Orig-Y':>10} {'Recon-X':>12} {'Recon-Y':>12} {'Hilbert-1D':>15}")
    print("-" * 100)

    for node, data in results.items():
        x = data['coordinates']['x']
        y = data['coordinates']['y']
        rx = data['reconstructed_coords']['x']
        ry = data['reconstructed_coords']['y']
        hilbert = data['hilbert_index']
        print(f"{node:<30} {x:10.6f} {y:10.6f} {rx:12.6f} {ry:12.6f} {hilbert:>15}")

    with open('node_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
