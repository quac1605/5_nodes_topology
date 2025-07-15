#!/usr/bin/env python3
import pandas as pd
import numpy as np
import statistics

def read_data(full_path: str, one_d_path: str):
    # Read full data (with RTT) and 1D-only Hilbert‑distance file
    full = pd.read_csv(
        full_path,
        sep=r'\s+',
        comment='#',
        names=['node', 'rtt', 'ram', 'vcores', 'hilbert', 'hilbertdist', 'current'],
        dtype={'hilbertdist': str}    # read as str to avoid overflow
    )
    one_d = pd.read_csv(
        one_d_path,
        sep=r'\s+',
        comment='#',
        names=['node', 'hilbertdist'],
        dtype={'hilbertdist': str}
    )

    # Convert to proper Python types
    full['rtt']          = full['rtt'].astype(float)
    full['hilbertdist']  = full['hilbertdist'].apply(int)   # Python int, no overflow
    one_d['hilbertdist'] = one_d['hilbertdist'].apply(int)

    return full, one_d

def make_bins(thresholds):
    bins = [(0, thresholds[0])]
    for i in range(len(thresholds)-1):
        bins.append((thresholds[i], thresholds[i+1]))
    bins.append((thresholds[-1], float('inf')))
    labels = [
        f"{int(low)}-{int(high)}ms" if high != float('inf') else f">{int(low)}ms"
        for low, high in bins
    ]
    return bins, labels

def assign_rtt_bucket(rtt, bins, labels):
    for (low, high), lab in zip(bins, labels):
        if low <= rtt < high:
            return lab
    return None

def compute_reps(full, labels):
    reps = {}
    for lab, grp in full.groupby('orig_bucket'):
        d = grp['hilbertdist'].tolist()
        if d:
            reps[lab] = int(statistics.median(d))
    # ensure every label has an entry (even if empty)
    for lab in labels:
        reps.setdefault(lab, None)
    return reps

def assign_hilbert_bucket(hd, reps):
    # pick the label whose representative hilbertdist is closest
    valid = {lab: rv for lab, rv in reps.items() if rv is not None}
    return min(valid.keys(), key=lambda lab: abs(hd - valid[lab]))

def main():
    full_path = 'nodes_data.txt'
    one_d_path = 'nodes_data_1DHilbert.txt'

    # 1) load
    full, one_d = read_data(full_path, one_d_path)

    # 2) bins
    thresholds = [15, 25, 35, 45, 115, 125, 135, 145]
    bins, labels = make_bins(thresholds)

    # 3) original RTT buckets
    full['orig_bucket'] = full['rtt'].apply(assign_rtt_bucket, args=(bins, labels,))

    # 4) representative HilbertDist
    reps = compute_reps(full, labels)
    print("Representative HilbertDist per RTT-bucket:")
    for lab in labels:
        print(f"  {lab:>10s}: {reps[lab]}")

    # 5) predicted buckets from 1D-only
    one_d['pred_bucket'] = one_d['hilbertdist'].apply(assign_hilbert_bucket, args=(reps,))

    # 6) compare
    compare = pd.merge(
        full[['node', 'orig_bucket']],
        one_d[['node', 'pred_bucket']],
        on='node',
        how='inner'
    )

    print("\nBucket Assignment Comparison:")
    print(compare.to_string(index=False))

    # 7) mismatches
    mismatches = compare[compare['orig_bucket'] != compare['pred_bucket']]
    print(f"\nTotal nodes: {len(compare)}, Mismatches: {len(mismatches)}")
    if not mismatches.empty:
        print("Mismatched nodes:")
        print(mismatches.to_string(index=False))

if __name__ == "__main__":
    main()
