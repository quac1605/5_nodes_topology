#!/usr/bin/env python3
import csv
import statistics
from typing import List, Dict, Any, Tuple


def read_nodes_data(path: str) -> List[Dict[str, Any]]:
    """
    Reads nodes_data.txt and returns a list of dicts with keys:
    node, rtt, ram, vcores, hilbert, hilbertdist, current
    """
    nodes = []
    with open(path, newline='') as f:
        for row in csv.reader(f, delimiter=' ', skipinitialspace=True):
            if not row or row[0].startswith('#'):
                continue
            # row: [node, RTT, RAM, vCores, Hilbert, HilbertDist, Current]
            node = row[0]
            rtt = float(row[1])
            ram = float(row[2])
            vcores = int(row[3])
            hilbert = int(row[4])
            hilbertdist = int(row[5])
            current = (row[6] == 'current')
            nodes.append({
                'node': node,
                'rtt': rtt,
                'ram': ram,
                'vcores': vcores,
                'hilbert': hilbert,
                'hilbertdist': hilbertdist,
                'current': current
            })
    return nodes


def make_bins(thresholds: List[float]) -> Tuple[List[Tuple[float, float]], List[str]]:
    """
    Given sorted thresholds [t1, t2, ...], returns:
      - bins: [(0,t1), (t1,t2), ..., (tN, inf)]
      - labels: ["0-t1ms", "t1-t2ms", ..., ">tNms"]
    """
    bins = []
    labels = []
    prev = 0.0
    for t in thresholds:
        bins.append((prev, t))
        labels.append(f"{int(prev)}-{int(t)}ms")
        prev = t
    bins.append((prev, float('inf')))
    labels.append(f">{int(prev)}ms")
    return bins, labels


def assign_to_buckets(
    nodes: List[Dict[str, Any]],
    bins: List[Tuple[float, float]],
    labels: List[str]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Assigns each node to the bucket whose (low, high) contains its RTT.
    Returns a dict label -> list of node dicts.
    """
    bucketed = {label: [] for label in labels}
    for node in nodes:
        for (low, high), label in zip(bins, labels):
            if low <= node['rtt'] < high:
                bucketed[label].append(node)
                break
    return bucketed


def representative_hilbertdist(
    bucketed: Dict[str, List[Dict[str, Any]]],
    method: str = 'median'
) -> Dict[str, int]:
    """
    For each bucket label, computes a representative HilbertDist:
    method can be 'min', 'max', 'mean', or 'median'.
    """
    rep = {}
    for label, items in bucketed.items():
        dists = [n['hilbertdist'] for n in items]
        if not dists:
            rep[label] = None
        else:
            if method == 'min':
                rep[label] = min(dists)
            elif method == 'max':
                rep[label] = max(dists)
            elif method == 'mean':
                rep[label] = int(statistics.mean(dists))
            else:  # median
                rep[label] = int(statistics.median(dists))
    return rep


def find_bucket_from_hilbertdist(
    hilbertdist: int,
    reps: Dict[str, int]
) -> str:
    """
    Given a hilbertdist and a dict of representative values reps[label] -> rep_val,
    returns the label whose rep_val is closest to hilbertdist.
    """
    best_label = None
    best_diff = None
    for label, rep_val in reps.items():
        if rep_val is None:
            continue
        diff = abs(hilbertdist - rep_val)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_label = label
    return best_label


def main():
    # 1) Read the data
    nodes = read_nodes_data("nodes_data.txt")

    # 2) Define the RTT thresholds and build bins
    thresholds = [15, 25, 35, 45, 115, 125, 135, 145]
    bins, labels = make_bins(thresholds)

    # 3) Assign nodes into RTT buckets
    bucketed = assign_to_buckets(nodes, bins, labels)

    # 4) Compute a representative HilbertDist per bucket (using median)
    reps = representative_hilbertdist(bucketed, method='median')

    # 5) Output results
    print("RTT Buckets and their nodes:")
    for label in labels:
        items = bucketed[label]
        print(f"\nBucket {label}:")
        if not items:
            print("  (empty)")
            continue
        for n in items:
            mark = "*" if n['current'] else " "
            print(f"  {mark} {n['node']:20s} RTT={n['rtt']:6.2f}ms  HilbertDist={n['hilbertdist']}")
        print(f"  → representative HilbertDist ({label}): {reps[label]}")

    # 6) Demonstration: how to recover bucket from a raw HilbertDist
    sample = nodes[0]['hilbertdist']
    found = find_bucket_from_hilbertdist(sample, reps)
    print(f"\nExample: HilbertDist {sample} belongs to bucket {found!r}")


if __name__ == "__main__":
    main()
