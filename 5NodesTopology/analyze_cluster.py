import re
import pandas as pd
from hilbertcurve.hilbertcurve import HilbertCurve

# Read the log file
with open("nodes_log.txt", "r") as f:
    log = f.read()

# Extract coordinates
coord_pattern = re.compile(r"Node: (.*?)\s+=> X:\s*(-?\d+\.\d+)\s+Y:\s*(-?\d+\.\d+)(.*)")
coords = {}
current_node = None
current_coord = None

for match in coord_pattern.finditer(log):
    node, x, y, extra = match.groups()
    x, y = float(x), float(y)
    coords[node] = (x, y)
    if "[CURRENT NODE]" in extra:
        current_node = node
        current_coord = (x, y)

# Extract RTT
rtt_pattern = re.compile(r"Node: (.*?)\s+=> RTT: ([\d.]+) ms")
rtt = {match[0]: float(match[1]) for match in rtt_pattern.findall(log)}

# Extract Ping
ping_pattern = re.compile(r"Node: (.*?)\s+=> Ping: time=([\d.]+) ms")
ping = {match[0]: float(match[1]) for match in ping_pattern.findall(log)}

# Normalize coordinates for Hilbert
def normalize(coords, scale=10000):
    all_coords = list(coords.values()) + [current_coord]
    min_x = min(x for x, _ in all_coords)
    min_y = min(y for _, y in all_coords)
    norm = {
        node: [int((x - min_x) * scale), int((y - min_y) * scale)]
        for node, (x, y) in coords.items()
    }
    current = [int((current_coord[0] - min_x) * scale), int((current_coord[1] - min_y) * scale)]
    return norm, current

norm_coords, norm_current = normalize(coords)

# Hilbert Transform
hilbert = HilbertCurve(p=16, n=2)
hilbert_indices = {node: hilbert.distance_from_coordinates(coord) for node, coord in norm_coords.items()}
hilbert_current_index = hilbert.distance_from_coordinates(norm_current)
hilbert_distances = {node: abs(idx - hilbert_current_index) for node, idx in hilbert_indices.items()}

# Build DataFrame
data = []
for node in coords:
    if node == current_node:
        continue
    data.append({
        "Node": node,
        "Ping (ms)": ping.get(node),
        "RTT (ms)": rtt.get(node),
        "Hilbert Distance": hilbert_distances.get(node)
    })

df = pd.DataFrame(data)
df_sorted = df.sort_values(by=["Ping (ms)", "RTT (ms)", "Hilbert Distance"])

# Output to terminal
print("\nSorted Node Proximity Analysis:\n")
print(df_sorted.to_string(index=False))

# Save to log file
output_file = "node_analysis_log.csv"
df_sorted.to_csv(output_file, index=False)
print(f"\n✅ Analysis saved to: {output_file}")
