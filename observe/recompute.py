"""Tell a RELOAD apart from a RECOMPUTE — the one KV-cost split the raw transfer stream
cannot make on its own.

When a request reuses a prefix whose KV was evicted from the GPU, the engine makes that
content resident again one of two ways:
  * RELOAD   — copy the KV back from a lower tier (host/remote). This is an H2D transfer the
               eBPF copy face captures.
  * RECOMPUTE— re-derive the KV by running the prefill compute again. There is NO copy at
               all; to a transfer-only observer the content simply reappears from nowhere.

So you cannot classify by watching copies alone: a recompute is invisible there. What ties
"this content just became resident" to "the same content that was here before" is its
CONTENT HASH (token-derived, prefix-chained — see content_hashes_from_tokens). Given a
timeline of which content-hashes are resident over time plus the captured H2D loads, a
re-materialization of a previously-evicted hash is a RELOAD iff an H2D load landed in its
block just before it reappeared, else a RECOMPUTE. This is exactly eKV's value-add: the
block-level content identity makes the otherwise-invisible recompute observable, and
separates wasted recompute (a caching/placement bug) from honest tier movement.
"""
from __future__ import annotations

import bisect


def classify_rematerializations(timeline, loads, window_ns: int = 500_000_000,
                                compute=None, min_compute: int = 2) -> dict:
    """Classify each re-materialization of a previously-evicted content hash into RELOAD,
    RECOMPUTE, or (free) CACHE-HIT.

    timeline: ordered ``[(t_ns, {content_hash: block_id}), ...]`` resident snapshots (e.g. one
              per scheduler step), content_hash from content_hashes_from_tokens.
    loads:    iterable of ``(block_id, finished_at_ns)`` — the captured H2D KV loads.
    compute:  OPTIONAL parallel list (same length as timeline) of ``{content_hash: query_len}``
              — the per-slot query length the attention kernel computed that step (the Triton
              pass's ``cur_batch_query_len`` / cu_seqlens_q diff). When given, RECOMPUTE vs
              CACHE-HIT is decided EXACTLY by whether compute actually ran for the block
              (query_len >= ``min_compute`` -> the prefix was re-derived) instead of the
              same-vs-fresh-block heuristic; this is the pass-based exact path.

    A hash resident for the first time ever is first_compute. A hash that was resident, went
    absent, and reappears at step t is:
      * RELOAD    — an H2D load into its block finished within ``window_ns`` before t (copy);
      * RECOMPUTE — no copy, and (compute given) the kernel computed it [query_len>=min_compute],
                    else (heuristic) it reappears in a DIFFERENT/freshly-allocated block;
      * CACHE-HIT — no copy, and (compute given) the kernel did NOT compute it [query_len small],
                    else (heuristic) it reappears in the SAME physical block (free pool reuse).
    Returns {'reload', 'recompute', 'cache_hit': [(hash, block, t)], 'first_compute': int}."""
    load_times: dict = {}
    for blk, t in loads:
        load_times.setdefault(int(blk), []).append(int(t))
    for v in load_times.values():
        v.sort()

    seen: set = set()        # every content hash that has ever been resident
    present: set = set()     # resident in the previous snapshot
    last_block: dict = {}    # content hash -> the block it last occupied while resident
    reload_ev: list = []
    recompute_ev: list = []
    cache_hit_ev: list = []
    first = 0
    for idx, (t, snap) in enumerate(timeline):
        cur = set(snap)
        comp = compute[idx] if compute is not None else None
        for h in cur - present:                       # appeared since the last snapshot
            if h not in seen:
                first += 1                            # first-ever materialization (fresh)
                continue
            blk = int(snap[h])
            ts = load_times.get(blk, [])
            i = bisect.bisect_right(ts, t)
            if i > 0 and (t - ts[i - 1]) <= window_ns:
                reload_ev.append((h, blk, t))         # a copy brought it back
            elif comp is not None:                    # EXACT: did the kernel compute it?
                if comp.get(h, 0) >= min_compute:
                    recompute_ev.append((h, blk, t))  # query_len shows compute ran -> re-derived
                else:
                    cache_hit_ev.append((h, blk, t))  # not computed, no copy -> free reuse
            elif last_block.get(h) == blk:
                cache_hit_ev.append((h, blk, t))      # heuristic: same block, no copy
            else:
                recompute_ev.append((h, blk, t))      # heuristic: fresh block, no copy
        for h in cur:
            last_block[h] = int(snap[h])
        seen |= cur
        present = cur
    return {"reload": reload_ev, "recompute": recompute_ev,
            "cache_hit": cache_hit_ev, "first_compute": first}
