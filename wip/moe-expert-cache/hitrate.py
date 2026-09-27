#!/usr/bin/env python3
"""wip/moe-expert-cache: compute the static-profile expert-cache hit rate h(S) from a
GGML_MOE_PROFILE dump (exp1-moe-routing-profiler.patch).

Usage:
  hitrate.py PROFILE.csv                 # in-sample h(S): rank by this profile, score this profile
  hitrate.py PROFILE.csv --book BOOK.csv # held-out h(S): rank by BOOK, score PROFILE
"""
import collections, csv, sys

def load(path):
    layers = collections.defaultdict(collections.Counter)
    for r in csv.reader(open(path)):
        if r and r[0] == 'expert':
            layers[int(r[1])][int(r[2])] = int(r[3])
    return layers

def main():
    args = sys.argv[1:]
    scored = load(args[0])
    book = scored
    book_path = None
    if len(args) >= 3 and args[1] == '--book':
        book_path = args[2]
        book = load(book_path)
    tot = sum(sum(v.values()) for v in scored.values())
    print(f"score={args[0]} book={book_path if book_path else 'same (in-sample)'}")
    print(f"{'S/layer':>7} {'resident':>8} {'h(S)':>7}  cache GiB(Q8_0)")
    for S in (4, 8, 12, 16, 24, 32, 48, 64, 85, 128, 192, 256):
        hit = 0
        for l, v in scored.items():
            ranked = [e for e, _ in book.get(l, collections.Counter()).most_common(S)]
            hit += sum(v.get(e, 0) for e in ranked)
        print(f"{S:>7} {100*S/256:>7.0f}% {hit/tot:>7.3f}  {S*len(scored)*3.0/1024:>5.1f}")

main()
