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
                else:
                    pings[node] = None  # represents failed ping

    return coordinates, rtts, pings

def main():
    filename = 'nodes_log.txt'
    coords, rtts, pings = parse_log(filename)
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

    # Define current node
    current_node = "clab-century-serf1"
    current_hilbert = hilbert_values[current_node]

    print(f"\nCurrent Node: {current_node}\n")

    # 1. RTT
    print("1. Distance through Round Trip Time (ms):")
    sorted_rtt = sorted(
        [(n, rtt) for n, rtt in rtts.items() if n != current_node],
        key=lambda x: x[1]
    )
    for node, rtt in sorted_rtt:
        print(f"   {node:<25} => {rtt:.2f} ms")
    print()

    # 2. Ping
    print("2. Distance through Ping:")
    sorted_ping = sorted(
        [(n, pings.get(n)) for n in coords if n != current_node],
        key=lambda x: (float('inf') if x[1] is None else x[1])
    )
    for node, ping in sorted_ping:
        if ping is None:
            print(f"   {node:<25} => ping failed: exit status 1")
        else:
            print(f"   {node:<25} => {ping:.2f} ms")
    print()

    # 3. Hilbert Distance
    print("3. Distance with Hilbert 1D Transform:")
    sorted_hilbert = sorted(
        [(n, abs(hilbert_values[n] - current_hilbert)) for n in coords if n != current_node],
        key=lambda x: x[1]
    )
    for node, dist in sorted_hilbert:
        hv = hilbert_values[node]
        dn = recon_norm[node]
        dc = recon_coords[node]
        ox, oy = coords[node]
        print(f"   {node:<25} => Hilbert1D: {hv:<10} HilbertDist: {dist:<10} "
              f"Decoded(X,Y): ({dc[0]:.6f}, {dc[1]:.6f}) "
              f"Original(X,Y): ({ox:.6f}, {oy:.6f})")

if __name__ == "__main__":
    main()
