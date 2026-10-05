#!/usr/bin/env python3
"""Non-uniform per-layer slot allocation: optimal vs uniform (see RESULTS.md).

Reads the moe-expert-cache campaign routing profiles and, for a fixed total slot budget,
compares the uniform per-layer allocation with the greedy water-filling optimum on a separable
concave coverage objective.

Usage:
    python3 waterfill.py [path/to/moe-expert-cache]
"""
import csv
import collections
import heapq
import os
import sys

CAMPAIGN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "archive", "work", "moe-expert-cache")

FILES = {
    "prose":     "prof-prose-2000.csv",
    "mixed":     "prof-code-reasoning-mixed.csv",
    "python":    "prof-code-python.csv",
    "reasoning": "prof-reasoning.csv",
    "recall":    "prof-recall.csv",
}


def load(fn):
    per = collections.defaultdict(list)
    for row in csv.reader(open(fn)):
        if len(row) == 4 and row[0] == "expert":
            per[int(row[1])].append(int(row[3]))
    return {L: sorted(c, reverse=True) for L, c in per.items()}


def coverage(cs, S):
    t = sum(cs)
    return sum(cs[:S]) / t if t else 0.0


def optimal(book, layers, K):
    """Greedy water-filling: optimal for sum_l coverage_l(S_l) s.t. sum_l S_l <= K."""
    S = {L: 0 for L in layers}
    h = []
    for L in layers:
        if book[L]:
            heapq.heappush(h, (-book[L][0], L))
    for _ in range(K):
        if not h:
            break
        negm, L = heapq.heappop(h)
        S[L] += 1
        if S[L] < len(book[L]):
            heapq.heappush(h, (-book[L][S[L]], L))
    return S


def main():
    prof = {k: load(os.path.join(CAMPAIGN, v)) for k, v in FILES.items()}
    layers = sorted(prof["prose"].keys())

    print(f"{'slots/layer':>11}  {'in-sample gain (min/med/max)':>28}   held-out (prose alloc)")
    for per in (4, 8, 16, 32, 64, 96, 128):
        K = per * len(layers)
        gin, gh = [], []
        for name in FILES:
            b = prof[name]
            tot = sum(sum(b[L]) for L in layers)
            uni = sum(coverage(b[L], per) * sum(b[L]) for L in layers) / tot
            opt = sum(coverage(b[L], optimal(b, layers, K)[L]) * sum(b[L]) for L in layers) / tot
            gin.append(100 * (opt - uni) / uni)
        Sp = optimal(prof["prose"], layers, K)
        for name in ("mixed", "python", "reasoning", "recall"):
            b = prof[name]
            tot = sum(sum(b[L]) for L in layers)
            uni = sum(coverage(b[L], per) * sum(b[L]) for L in layers) / tot
            he = sum(coverage(b[L], Sp[L]) * sum(b[L]) for L in layers) / tot
            gh.append(100 * (he - uni) / uni)
        gin.sort()
        print(f"{per:>11}  {min(gin):+6.1f} / {gin[len(gin)//2]:+5.1f} / {max(gin):+5.1f} %"
              f"        {min(gh):+5.1f} .. {max(gh):+5.1f} %")


if __name__ == "__main__":
    main()
