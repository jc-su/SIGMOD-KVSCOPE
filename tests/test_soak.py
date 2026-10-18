"""Soak/stress: drive ~200k synthetic copy events (8 concurrent streams) through the
full offline pipeline — read_copies -> fuse -> dedup -> flame -> health — and assert
it stays correct, lossless, and bounded. The production "does it hold under sustained
volume" check, no GPU/eBPF needed."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observe import (BlockAddressMap, build_flame, collect_health,  # noqa: E402
                     dedup_by_hash, fuse_copies)
from observe.ebpf_reader import read_copies_report  # noqa: E402

N_STREAMS, N_PER = 8, 25000          # 200k enqueues
BASE, STRIDE, NBLK = 0x700000000, 16384, 4096
BATCH = 64


def gen_lines():
    for i in range(N_PER):
        for s in range(N_STREAMS):
            addr = BASE + ((i + s) % NBLK) * STRIDE
            yield (f'{{"type":"enqueue","t_ns":{i * 1000 + s},"kind":"D2H","stream":{s},'
                   f'"src":{addr},"dst":1,"bytes":{STRIDE},"pid":100}}')
        if i % BATCH == BATCH - 1:                      # sync each stream every 64
            for s in range(N_STREAMS):
                yield (f'{{"type":"completion","via":"stream-sync","t_ns":{i * 1000 + 999},'
                       f'"n":{BATCH},"handle":{s}}}')
    rem = N_PER - (N_PER // BATCH) * BATCH              # final partial batch
    if rem:
        for s in range(N_STREAMS):
            yield (f'{{"type":"completion","via":"stream-sync","t_ns":{N_PER * 1000},'
                   f'"n":{rem},"handle":{s}}}')


m = BlockAddressMap()
for bid in range(NBLK):
    m.add(bid, BASE + bid * STRIDE, STRIDE, content_hash=1000 + bid)

t0 = time.monotonic()
rep = read_copies_report(gen_lines())
edges = fuse_copies(rep.copies, m)
deduped = dedup_by_hash(edges)
flame = build_flame(deduped)
health = collect_health(capture=rep, edges=edges)
dt = time.monotonic() - t0

total = N_STREAMS * N_PER
costed = sum(1 for c in rep.copies if c.is_complete)
print(f"soak: {len(rep.copies)} copies, {len(edges)} edges, {len(deduped)} deduped, "
      f"flame={flame['value']:.0f}ns, {dt:.1f}s ({total / dt:,.0f} ev/s)")
print("health:", health.as_dict())

assert len(rep.copies) == total, (len(rep.copies), total)
assert rep.complete and health.ok, (rep, health.as_dict())   # lossless: no gaps/overflow
assert costed == total, (costed, total)                       # every copy correlated
assert len(edges) == total                                    # every copy resolved to a block
print(f"PASS: soak — {total} events, lossless, fully correlated, bounded ({dt:.1f}s)")
