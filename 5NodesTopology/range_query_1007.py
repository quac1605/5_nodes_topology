import pandas as pd
import argparse
from bisect import bisect_left, bisect_right
from hilbertcurve.hilbertcurve import HilbertCurve

# -- Argument parsing --------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Range query nodes via Hilbert index without decoding coordinates"
)
parser.add_argument('--input-file', '-i', type=str, default='nodes_data_with_Hilbert.txt',
                    help='Path to file with node and H_index columns')
# Hilbert curve parameters
parser.add_argument('--bits',    '-p', type=int, default=16,
                    help='Bits per dimension used in Hilbert encoding')
parser.add_argument('--dims',    '-n', type=int, default=5,
                    help='Number of dimensions encoded (e.g., 5 for X,Y,Z,RAM,vCores)')
# Global extents used during encoding
parser.add_argument('--global-mins',  nargs='+', required=True,
                    help='Minimum values for each dimension (space-separated)')
parser.add_argument('--global-maxs',  nargs='+', required=True,
                    help='Maximum values for each dimension (space-separated)')
# Query hyper-box in original space (one range per dimension)
parser.add_argument('--q-ranges', nargs='+', required=True,
                    help='Query ranges for dims, e.g. "xmin xmax ymin ymax ..."')
args = parser.parse_args()

# Convert numeric arguments
P = args.bits
N = args.dims
try:
    GLOBAL_MINS = [float(x) for x in args.global_mins]
    GLOBAL_MAXS = [float(x) for x in args.global_maxs]
    # Query ranges given as flat list
    q = [float(x) for x in args.q_ranges]
    if len(q) != 2*N:
        raise ValueError(f"Expected {2*N} query values, got {len(q)}")
    # Split into [(min, max), ...]
    q_ranges = [tuple(q[i*2:(i+1)*2]) for i in range(N)]
except Exception as e:
    raise RuntimeError(f"Invalid numeric arguments: {e}")

# Initialize Hilbert curve
hc = HilbertCurve(P, N)

# Load node + Hilbert index
df = pd.read_csv(
    args.input_file,
    sep=r'\s+', comment='#', header=None,
    names=['node','H_index'], dtype={'node': str, 'H_index': str}
)
df['H_index'] = df['H_index'].astype(int)
df_sorted = df.sort_values('H_index').reset_index(drop=True)
h_list = df_sorted['H_index'].tolist()

# Helper: quantize a real vector to Hilbert grid ints

def quantize_point(pt, mins, maxs, p=P):
    return [
        int((c - mn) / (mx - mn) * (2**p - 1))
        for c, mn, mx in zip(pt, mins, maxs)
    ]

# Build intervals via Hilbert query
lo = quantize_point([r[0] for r in q_ranges], GLOBAL_MINS, GLOBAL_MAXS)
hi = quantize_point([r[1] for r in q_ranges], GLOBAL_MINS, GLOBAL_MAXS)
h_intervals = hc.query_range(lo, hi)

# Range-scan on H_index
result_frames = []
for h0, h1 in h_intervals:
    i0 = bisect_left(h_list, h0)
    i1 = bisect_right(h_list, h1)
    result_frames.append(df_sorted.iloc[i0:i1])

# Combine and output
if result_frames:
    matches = pd.concat(result_frames, ignore_index=True)
else:
    matches = pd.DataFrame(columns=df_sorted.columns)

print(f"Found {len(matches)} matching nodes:")
print(matches.to_string(index=False))
