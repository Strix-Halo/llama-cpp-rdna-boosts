#!/usr/bin/env python3
"""wip/moe-expert-cache: dynamic admission-policy simulator (campaign §2).

Replays the token-indexed routing traces produced by exp2-moe-routing-profiler.patch
(`GGML_MOE_PROFILE=<path> GGML_MOE_PROFILE_MAXTOK=1`) and evaluates, for a per-layer VRAM
expert cache of S slots, the candidate admission/replacement policies the prior art uses plus
combined LFRU / segmented-LRU policies:

  static              top-S by frequency from tokens < SPLIT, no adaptation (R9V's
                      `materialize_hot_expert_cache`; the warm start, the held-out bound).
  warm_lru            preload the static top-S, then LRU.
  lru                 admit every miss, evict least-recently-used (moe-cache replacement).
  second_touch        admit a non-resident expert on its SECOND touch (per-layer ghost counter);
                      evict LRU.  R9V's `second_touch_rr` default.
  second_touch_fill   as above, but free slots are filled on the first touch (a fairer reading of
                      R9V when the cache starts empty).
  lfu_decay:PERIOD    per-expert hit counter halved every PERIOD tokens; admit the expert only if
                      its count beats the least-frequent resident, then evict that.
  lfru_decay:PERIOD   combined: admit every miss, but evict the resident with the smallest count,
                      breaking ties by oldest use; counts halved every PERIOD tokens.  This is the
                      LRU/LFU hybrid the campaign set out to tune.
  slru:PROT           segmented LRU: PROT fraction of the slots is the "protected" (frequent)
                      segment, the rest is a probationary LRU.  A second hit promotes.
  lru_window:TTL      LRU eviction, but a miss is admitted only if that expert was seen within TTL
                      tokens (a recency admission filter that rejects one-off scan experts).

Everything uses PER-LAYER slot ranges (Strata's load-bearing finding: a global free-slot pool fills
the first position it sees and measures ~3% hit at the same budget; per-layer ranges turn it into
21-70%).

Reports, per (prompt, policy, S):
  * h_all     -- hit rate over every routed reach;
  * h_steady  -- hit rate over the last STEADY tokens (the number to design against);
  * h_win_min -- worst ~200-token window hit rate (the "collapses mid-run" detector);
  * h_win_mean

Usage:
  policy_sim.py TRACE_DIR [--S 8,16,32,48,64] [--split 1000] [--window 200] [--steady 1000]
  policy_sim.py TRACE_DIR --policies 'lru,lfru_decay:64,lfru_decay:128,slru:0.5'
  policy_sim.py TRACE_DIR --windows out.csv      # also dump per-window hit rate for every cell
"""
import argparse, collections, csv, glob, os, sys

WINDOW = 200


class Lru:
    def __init__(self, S):
        self.S = S
        self.res = collections.OrderedDict()     # expert -> None, MRU at the end

    def access(self, tok, e):
        if e in self.res:
            self.res.move_to_end(e)
            return True
        if len(self.res) >= self.S:
            self.res.popitem(last=False)
        self.res[e] = None
        return False


class SecondTouch:
    def __init__(self, S, fill_empty=False):
        self.S = S
        self.fill_empty = fill_empty
        self.res = collections.OrderedDict()
        self.ghost = set()                        # touched once but not resident

    def access(self, tok, e):
        if e in self.res:
            self.res.move_to_end(e)
            return True
        if e in self.ghost or (self.fill_empty and len(self.res) < self.S):
            if len(self.res) >= self.S:
                self.res.popitem(last=False)
            self.res[e] = None
            self.ghost.discard(e)
        else:
            self.ghost.add(e)
        return False


class LfuDecay:
    def __init__(self, S, period):
        self.S = S
        self.period = period
        self.res = {}
        self.count = collections.defaultdict(int)
        self.last_decay = 0

    def _decay(self, tok):
        while tok - self.last_decay >= self.period:
            for k in list(self.count.keys()):
                c = self.count[k] >> 1
                if c:
                    self.count[k] = c
                else:
                    del self.count[k]
            self.last_decay += self.period

    def access(self, tok, e):
        self._decay(tok)
        self.count[e] += 1
        if e in self.res:
            return True
        if len(self.res) < self.S:
            self.res[e] = None
            return False
        victim = min(self.res, key=self.count.__getitem__)
        if self.count[e] > self.count[victim]:
            del self.res[victim]
            self.res[e] = None
        return False


class LfruDecay:
    """Combined LFRU: admit every miss, evict min count with an LRU tie-break, decay counts."""
    def __init__(self, S, period):
        self.S = S
        self.period = period
        self.res = {}                             # expert -> last_used tok
        self.count = collections.defaultdict(int)
        self.last_decay = 0

    def _decay(self, tok):
        while tok - self.last_decay >= self.period:
            for k in list(self.count.keys()):
                c = self.count[k] >> 1
                if c:
                    self.count[k] = c
                else:
                    del self.count[k]
            self.last_decay += self.period

    def access(self, tok, e):
        self._decay(tok)
        self.count[e] += 1
        if e in self.res:
            self.res[e] = tok
            return True
        if len(self.res) < self.S:
            self.res[e] = tok
            return False
        # evict: min count, tie -> oldest last_used
        victim = min(self.res, key=lambda x: (self.count[x], self.res[x]))
        del self.res[victim]
        self.res[e] = tok
        return False


class Slru:
    """Segmented LRU: `prot` fraction is the protected (frequent) segment."""
    def __init__(self, S, prot_frac):
        self.S = S
        self.cprot = max(0, min(S, int(round(S * prot_frac))))
        self.cprob = S - self.cprot
        self.prot = collections.OrderedDict()     # frequent, MRU at end
        self.prob = collections.OrderedDict()     # probationary, LRU at start

    def _evict_prob(self):
        if len(self.prob) >= self.cprob:
            self.prob.popitem(last=False)

    def access(self, tok, e):
        if e in self.prot:
            self.prot.move_to_end(e)
            return True
        if e in self.prob:
            del self.prob[e]
            self.prot[e] = None
            if len(self.prot) > self.cprot:
                _k, _ = self.prot.popitem(last=False)   # demote protected LRU
                if len(self.prob) >= self.cprob:
                    self.prob.popitem(last=False)
                self.prob[_k] = None
            return True
        # miss -> enter probation
        if self.cprob == 0:
            # no probationary room: behave as protected-only LRU
            if len(self.prot) >= self.S:
                self.prot.popitem(last=False)
            self.prot[e] = None
        else:
            if len(self.prob) >= self.cprob:
                self.prob.popitem(last=False)
            self.prob[e] = None
        return False


class LruWindow:
    """LRU eviction, but a miss is admitted only if the expert was seen within TTL tokens."""
    def __init__(self, S, ttl):
        self.S = S
        self.ttl = ttl
        self.res = collections.OrderedDict()
        self.last_seen = {}

    def access(self, tok, e):
        if e in self.res:
            self.res.move_to_end(e)
            self.last_seen[e] = tok
            return True
        if e in self.last_seen and tok - self.last_seen[e] <= self.ttl:
            if len(self.res) >= self.S:
                self.res.popitem(last=False)
            self.res[e] = None
        self.last_seen[e] = tok
        return False


def make_policy(spec, S, window):
    name, _, param = spec.partition(':')
    if name == 'lru':
        return Lru(S), None
    if name == 'second_touch':
        return SecondTouch(S), None
    if name == 'second_touch_fill':
        return SecondTouch(S, fill_empty=True), None
    if name == 'lfu_decay':
        return LfuDecay(S, period=int(param) if param else max(1, window // 2)), None
    if name == 'lfru_decay':
        return LfruDecay(S, period=int(param) if param else max(1, window // 2)), None
    if name == 'slru':
        return Slru(S, prot_frac=float(param) if param else 0.5), None
    if name == 'lru_window':
        return LruWindow(S, ttl=int(param) if param else window), None
    if name == 'static':
        return None, 'static'
    if name == 'warm_lru':
        return Lru(S), 'warm_lru'
    raise ValueError(spec)


def run_policy(layers, spec, S, split, window):
    # per-layer caches: replay each layer independently, aggregate the per-window counts.
    win_hit = collections.defaultdict(int)
    win_reach = collections.defaultdict(int)
    total_hit = 0
    total_reach = 0
    for _layer, evs in layers.items():
        p, special = make_policy(spec, S, window)
        hot = None
        if special in ('static', 'warm_lru'):
            freq = collections.Counter(e for (tok, e) in evs if tok < split)
            hot = [e for e, _ in freq.most_common(S)]
            if special == 'warm_lru':
                for e in hot:
                    p.res[e] = None
        for (tok, e) in evs:
            w = tok // window
            win_reach[w] += 1
            total_reach += 1
            hit = (e in hot) if special == 'static' else p.access(tok, e)
            if hit:
                win_hit[w] += 1
                total_hit += 1
    nwin = max(win_reach) + 1 if win_reach else 0
    return total_reach, total_hit, [win_hit[w] for w in range(nwin)], [win_reach[w] for w in range(nwin)]


def load_trace(path):
    layers = collections.defaultdict(list)
    with open(path) as f:
        for line in f:
            if not line.startswith('route,'):
                continue
            _, tok, layer, expert = line.rstrip('\n').split(',')
            layers[int(layer)].append((int(tok), int(expert)))
    return layers


def slots_gib(nlayers, S):
    # Q8_0 expert (gate+up+down) = 3.00 MiB; the campaign's Phase-1 vehicle.
    return nlayers * S * 3.0 / 1024.0


def h_steady(wh, wr, steady_tokens, window):
    nw = max(1, steady_tokens // window)
    hs, rs = sum(wh[-nw:]), sum(wr[-nw:])
    return (hs / rs) if rs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace_dir')
    ap.add_argument('--S', default='8,16,32,48,64')
    ap.add_argument('--split', type=int, default=1000)
    ap.add_argument('--window', type=int, default=200)
    ap.add_argument('--steady', type=int, default=1000)
    ap.add_argument('--windows', default='')
    ap.add_argument('--policies', default='static,lru,second_touch,lfu_decay')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    Ss = [int(x) for x in args.S.split(',')]
    policies = args.policies.split(',')
    traces = sorted(glob.glob(os.path.join(args.trace_dir, '*.csv.trace')))
    if not traces:
        sys.exit(f"no *.csv.trace under {args.trace_dir}")

    loaded = {}
    for tr in traces:
        name = os.path.basename(tr)[:-len('.csv.trace')]
        loaded[name] = load_trace(tr)

    rows, win_rows = [], []
    for name, layers in loaded.items():
        nlayers = len(layers)
        for S in Ss:
            for spec in policies:
                tr_, th, wh, wr = run_policy(layers, spec, S, args.split, args.window)
                h_all = th / tr_ if tr_ else 0.0
                hs = h_steady(wh, wr, args.steady, args.window)
                rates = [wh[i] / wr[i] for i in range(len(wr)) if wr[i] > 0]
                hmin = min(rates) if rates else 0.0
                hmean = sum(rates) / len(rates) if rates else 0.0
                rows.append(dict(prompt=name, policy=spec, S=S, reaches=tr_, hits=th,
                                 h_all=round(h_all, 4), h_steady=round(hs, 4),
                                 h_win_min=round(hmin, 4), h_win_mean=round(hmean, 4),
                                 cache_gib=round(slots_gib(nlayers, S), 2)))
                if args.windows:
                    for i in range(len(wr)):
                        if wr[i]:
                            win_rows.append(dict(prompt=name, policy=spec, S=S, window=i,
                                                 reaches=wr[i], h=round(wh[i] / wr[i], 4)))

    out = os.path.join(args.trace_dir, 'policy-sim.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if args.windows:
        with open(args.windows, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(win_rows[0].keys()))
            w.writeheader()
            w.writerows(win_rows)

    if args.quiet:
        return

    def print_table(title, key, fmt):
        print(f"\n{title}")
        print(f"{'S':>3} {'policy':>18} {'mean':>7} {'worst':>7} {'best':>7} {'GiB':>6}")
        for S in Ss:
            for spec in policies:
                rs = [r for r in rows if r['S'] == S and r['policy'] == spec]
                if not rs:
                    continue
                vals = [r[key] for r in rs]
                print(f"{S:>3} {spec:>18} {sum(vals)/len(vals):>7.3f} {min(vals):>7.3f} "
                      f"{max(vals):>7.3f} {rs[0]['cache_gib']:>6.2f}")
            print()

    print_table(f"h_steady (last {args.steady} tokens), over {len(traces)} prompts",
                'h_steady', '%.3f')
    print_table("h_win_min (worst 200-token window), over 5 prompts x 10 windows",
                'h_win_min', '%.3f')


if __name__ == '__main__':
    main()
