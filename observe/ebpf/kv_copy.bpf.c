// eKV OS-data-plane probe: the HBM<->DRAM KV offload/swap "edge".
//
// Enqueue (uprobe cudaMemcpyAsync): bytes + direction + CUDA stream, accumulated
// per-stream.
//
// Completion latency without CUPTI. A uprobe fires at ENQUEUE, not completion. We
// recover completion by correlating with the host's wait. Real KV-offload paths
// (vLLM connector / LMCache) do NOT block on cudaStreamSynchronize -- they use
// CUDA EVENTS (cudaEventRecord then cudaEventSynchronize / cudaEventQuery). So we
// support BOTH:
//   * stream sync:  cudaStreamSynchronize(stream) return -> stream's copies done.
//   * event record: cudaEventRecord(ev, stream) moves the stream's pending copies
//                   onto the event; cudaEventSynchronize(ev) return, or the first
//                   cudaEventQuery(ev) that returns cudaSuccess(0), is completion.
// completion.kind: 0 stream-sync, 1 event-sync, 2 event-query.
//
// Transport: one BPF ring buffer (Linux 5.8+) carries every record. A failed reservation, batch
// truncation, and a batch descriptor read fault are each COUNTED in the stats map, and the loader
// reports the counts at exit, so coverage is auditable. cudaStreamDestroy clears per-stream state
// (cudaStream_t handles are pointers CUDA reuses across lifetimes).
//
// Kernel requirement: Linux 5.8+ with BTF (/sys/kernel/btf/vmlinux). The program uses only
// helpers that Linux 5.8 has: no bpf_loop (5.17), no BPF arena (6.9).
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "kv_copy.h"

struct stream_acc {
  __u64 n;
  __u64 bytes;
  __u64 first_ns;
  __u64 last_poll_ns; /* event-query path only: ts of the latest cudaEventQuery that returned
                         NOT-READY before the first success. The copy's GPU-completion lies in
                         (last_poll_ns, completion_ts] -- a tight bracket whose width is the
                         engine's poll gap, recovered with NO in-process timing. 0 for blocking
                         stream/event sync (already tight: the wait wakes ON the GPU signal) or
                         when the very first poll already succeeded (no lower bound observed). */
};
struct sync_ctx {
  __u64 handle;
  __u64 entry_ns;
  __u32 kind; /* 0 stream-sync, 1 event-sync, 2 event-query */
};

struct {
  __uint(type, BPF_MAP_TYPE_RINGBUF);
  __uint(max_entries, EKV_RINGBUF_BYTES);  /* see kv_copy.h for the sizing */
} rb SEC(".maps");

struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, 8192);
  __type(key, __u64); /* stream handle */
  __type(value, struct stream_acc);
} stream_state SEC(".maps");

struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, 8192);
  __type(key, __u64); /* event handle */
  __type(value, struct stream_acc);
} event_state SEC(".maps");

struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, 8192);
  __type(key, __u32); /* tid */
  __type(value, struct sync_ctx);
} sync_tid SEC(".maps");

/* Loss counters, indexed by the ST_* constants in kv_copy.h. They stay DISTINCT because they are
 * different failure modes:
 *   ST_COPY_DROP      = copy records lost on a failed ring buffer reservation (transport loss).
 *   ST_BATCH_TRUNC    = blocks in a cuMemcpyBatchAsync beyond KV_BATCH_MAX. This is an upstream
 *                       cap, not a transport drop.
 *   ST_BATCH_READFAIL = blocks in a chunk whose dst/src/sz read from the engine's host arrays
 *                       faulted (e.g. paged out under memory pressure). The chunk is discarded
 *                       rather than emitted with zero/garbage fields (which would violate the
 *                       reader's nbytes>0 contract).
 *   ST_META_DROP      = metadata blobs lost on a failed ring buffer reservation. */
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, ST_COUNT);
  __type(key, __u32);
  __type(value, __u64);
} stats SEC(".maps");

static __always_inline void count_loss(__u32 k, __u64 n) {
  __u64 *v = bpf_map_lookup_elem(&stats, &k);
  if (v)
    __sync_fetch_and_add(v, n);
}

/* Per-tid last copy, to drop a wrapper+_ptsz double-fire: a driver entry point
 * (cuMemcpyHtoDAsync_v2, cuMemcpyBatchAsync) can reach its _ptsz variant, so both uprobes
 * fire for ONE logical copy with identical (dst,src,count) on the same tid ~ns apart. We keep
 * the first and skip a byte-identical repeat within DEDUP_NS. (The runtime's own nesting,
 * cudaMemcpyAsync -> _ptsz -> driver, is handled exactly by rt_depth below.) */
struct last_copy {
  __u64 dst, src, count, ts;
};
struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, 8192);
  __type(key, __u32); /* tid */
  __type(value, struct last_copy);
} last_copy SEC(".maps");
#define DEDUP_NS 5000

/* Per-tid nesting depth of runtime cudaMemcpyAsync[_ptsz] calls (entry +1, return -1).
 * The runtime implements an async H2D/D2H copy by calling the driver's exported
 * cuMemcpyHtoDAsync_v2 / cuMemcpyDtoHAsync_v2, 5-25 us after its own entry on this host, which
 * is too late for the DEDUP_NS window. Without this map, one runtime copy produced two enqueue
 * records (measured on a vLLM 0.21 engine: 57 of 57 driver single copies duplicated a runtime
 * copy). A driver copy made while the tid is inside a runtime copy is that same copy, so the
 * driver probes skip it; a nested runtime entry (cudaMemcpyAsync -> cudaMemcpyAsync_ptsz) is
 * skipped too. */
struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, 8192);
  __type(key, __u32); /* tid */
  __type(value, __u32); /* depth */
} rt_depth SEC(".maps");

/* Enter a runtime copy. Returns 1 if this tid was already inside one (a nested entry). */
static __always_inline int runtime_enter(void) {
  __u32 tid = (__u32)bpf_get_current_pid_tgid();
  __u32 *d = bpf_map_lookup_elem(&rt_depth, &tid);
  if (d) {
    *d += 1;
    return 1;
  }
  __u32 one = 1;
  bpf_map_update_elem(&rt_depth, &tid, &one, BPF_ANY);
  return 0;
}

static __always_inline int in_runtime_copy(void) {
  __u32 tid = (__u32)bpf_get_current_pid_tgid();
  return bpf_map_lookup_elem(&rt_depth, &tid) != NULL;
}

/* EKV_SAMPLE: capture 1-in-N offload batches. The offload uprobe emits one record
 * per block (hundreds/batch) synchronously on the offload thread, which costs ~5%
 * on an offload-bottlenecked workload. Sampling 1-in-N cuts that ~N×, trading
 * per-block coverage (extrapolate volume ×N) for overhead. sample_n=1 = full. */
const volatile __u32 sample_n = 1;
struct {
  __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
  __uint(max_entries, 1);
  __type(key, __u32);
  __type(value, __u64);  /* per-cpu batch counter for sampling */
} batch_seq SEC(".maps");

/* EKV_META: capture the HOST CONTENT of small H2D copies (the attention metadata —
 * block_table/page_table/cu_seqlens — that carries request<->block). Off by default
 * (=0): no host-content read, no overhead. The loader identifies + parses the metadata
 * copies among these by their stable pinned destination. */
const volatile __u32 meta_on = 0;
/* TEST-ONLY (EKV_FORCE_READFAIL): treat every batch descriptor read as a fault, to exercise the
 * do_chunk read-fault path (skip + audit stats[ST_BATCH_READFAIL]) on demand -- a real
 * bpf_probe_read_user fault cannot be injected safely (the driver faults on the same bogus array).
 * Default 0 = off (a single dead-on-zero compare on the hot path). */
const volatile __u32 force_readfail = 0;
/* No per-cpu scratch map: a PERCPU_ARRAY value is capped at PCPU_MIN_UNIT_SIZE (32768),
 * and the block_table (request<->block map) can be ~24KB+ in one copy. on_meta reserves
 * the record DIRECTLY in the ring buffer instead -- per-reservation (no cross-CPU race),
 * so KV_META_MAX can be a full 32KB. */

/* Append ONE fixed-size copy record (enqueue / completion / bind) to the ring buffer, with
 * pid/tid from the current context. A failed reservation is counted in ST_COPY_DROP. */
static __always_inline void put_evt(__u32 type, __u32 kind, __u32 api, __u64 ts, __u64 bytes,
                                    __u64 handle, __u64 dst, __u64 src, __u64 first_ns,
                                    __u64 wait_ns, __u32 n) {
  struct kv_copy_evt *e = bpf_ringbuf_reserve(&rb, sizeof(*e), 0);
  if (!e) {
    count_loss(ST_COPY_DROP, 1);
    return;
  }
  __u64 id = bpf_get_current_pid_tgid();
  e->type = type;
  e->kind = kind;
  e->api = api;
  e->pid = (__u32)(id >> 32);
  e->tid = (__u32)id;
  e->n = n;
  e->ts_ns = ts;
  e->bytes = bytes;
  e->first_ns = first_ns;
  e->wait_ns = wait_ns;
  e->handle = handle;
  e->dst = dst;
  e->src = src;
  bpf_ringbuf_submit(e, 0);
}

static __always_inline void on_copy(void *dst, void *src, __u64 count,
                                    __u32 kind, __u32 api, __u64 stream) {
  // Host<->host (cudaMemcpyKind 0) is never a KV tier movement. Drop it AT THE
  // SOURCE — not just in the reader — so it is neither emitted nor counted in the
  // per-stream accumulator. Otherwise a completion's `n` would include H2H copies
  // the reader discards, falsely tripping the count-overflow check.
  if (kind == 0)
    return;

  __u64 now = bpf_ktime_get_ns();
  __u64 id = bpf_get_current_pid_tgid();

  // drop the wrapper+_ptsz double-fire (byte-identical copy on this tid within window)
  __u32 tid = (__u32)id;
  struct last_copy *lc = bpf_map_lookup_elem(&last_copy, &tid);
  if (lc && lc->dst == (__u64)dst && lc->src == (__u64)src && lc->count == count &&
      now - lc->ts < DEDUP_NS) {
    lc->ts = now;
    return;
  }
  struct last_copy ncp = {.dst = (__u64)dst, .src = (__u64)src, .count = count, .ts = now};
  bpf_map_update_elem(&last_copy, &tid, &ncp, BPF_ANY);

  struct stream_acc *a = bpf_map_lookup_elem(&stream_state, &stream);
  if (a) {
    a->n += 1;
    a->bytes += count;
    if (a->n == 1)
      a->first_ns = now;
  } else {
    struct stream_acc na = {.n = 1, .bytes = count, .first_ns = now};
    bpf_map_update_elem(&stream_state, &stream, &na, BPF_ANY);
  }

  put_evt(KV_ENQUEUE, kind, api, now, count, stream, (__u64)dst, (__u64)src,
          /*first_ns=*/now, /*wait_ns=*/0, /*n=*/1);
}

// EKV_META lives in its OWN lean probe (up_meta), NOT inlined into on_copy: the
// variable-size host read + ~1KB ringbuf_output is just enough extra verifier state
// to push on_copy (already near the 5.15 ceiling) over the limit. Keeping it separate
// lets BOTH verify — on_copy stays as-is, up_meta carries only the metadata capture.
static __always_inline void on_meta(__u64 dst, __u64 src, __u64 count) {
  if (!meta_on || count < 16 || count > KV_META_MAX)
    return;
  // Reserve the whole record in the ring buffer (constant size -> verifier-friendly),
  // then read the host source straight into it. No percpu scratch (PCPU_MIN_UNIT_SIZE
  // would cap us at 32K-headroom and reject the block_table), no cross-CPU race.
  struct kv_meta_blob *m = bpf_ringbuf_reserve(&rb, sizeof(*m), 0);
  if (!m) {
    count_loss(ST_META_DROP, 1);
    return;
  }
  // Bound the read size with a power-of-2 BITMASK, not only a compare. A u64 range
  // check (count <= KV_META_MAX) leaves the low 32 bits with a messy var_off the
  // 5.15 verifier won't accept for a variable-size write; the non-max path uses
  // `& (KV_META_MAX-1)` to prove [0, KV_META_MAX). Keep the exact-max case
  // separate: masking KV_META_MAX itself would turn a valid full-size copy into
  // size 0 and silently erase the metadata payload.
  __u32 sz = (count == KV_META_MAX) ? KV_META_MAX : ((__u32)count & (KV_META_MAX - 1));
  m->type = KV_META_BLOB;
  m->size = sz;
  m->ts_ns = bpf_ktime_get_ns();
  m->dst = dst;
  m->pid_tid = bpf_get_current_pid_tgid();
  // Read the host source. For an H2D copy src is a host pointer -> succeeds. For the
  // other direction (device src, if this symbol is ever bidirectional) the user read
  // faults and returns <0; discard the reservation. A successful host read IS the H2D
  // discriminator, so no explicit kind check is needed.
  if (bpf_probe_read_user(m->data, sz, (void *)src) < 0) {
    bpf_ringbuf_discard(m, 0);
    return;
  }
  bpf_ringbuf_submit(m, 0);
}

SEC("uprobe")
int BPF_KPROBE(up_memcpy_async, void *dst, void *src, __u64 count, __u32 kind,
               __u64 stream) {
  if (!runtime_enter())
    on_copy(dst, src, count, kind, 0, stream);
  return 0;
}
SEC("uprobe")
int BPF_KPROBE(up_memcpy_async_ptsz, void *dst, void *src, __u64 count,
               __u32 kind, __u64 stream) {
  if (!runtime_enter())
    on_copy(dst, src, count, kind, 1, stream);
  return 0;
}
// Return of cudaMemcpyAsync[_ptsz]: leave the runtime copy (see rt_depth).
SEC("uretprobe")
int BPF_KRETPROBE(uret_memcpy_async) {
  __u32 tid = (__u32)bpf_get_current_pid_tgid();
  __u32 *d = bpf_map_lookup_elem(&rt_depth, &tid);
  if (!d)
    return 0; // the collector attached while this call was in flight
  if (*d <= 1)
    bpf_map_delete_elem(&rt_depth, &tid);
  else
    *d -= 1;
  return 0;
}

// Dedicated metadata-content probe. (dst, src, count) are the first three args of BOTH
// cudaMemcpyAsync(dst,src,count,kind,stream) and cuMemcpyHtoDAsync_v2(dst,src,count,stream),
// so one signature attaches to either entry point. Separate from on_copy so the heavy
// runtime copy handler stays under the verifier ceiling (see on_meta comment).
SEC("uprobe")
int BPF_KPROBE(up_meta, __u64 dst, __u64 src, __u64 count) {
  on_meta(dst, src, count);
  return 0;
}
// The same capture on the driver's cuMemcpyHtoDAsync_v2, except for a copy the runtime is making
// (up_meta on cudaMemcpyAsync already captured it; see rt_depth).
SEC("uprobe")
int BPF_KPROBE(up_meta_drv, __u64 dst, __u64 src, __u64 count) {
  if (!in_runtime_copy())
    on_meta(dst, src, count);
  return 0;
}

// emit a completion record for a finished batch.
static __always_inline void emit_completion(__u64 handle, struct stream_acc *a,
                                            __u64 now, __u64 entry_ns,
                                            __u32 src) {
  // dst (unused on a completion) carries the event-query ready LOWER bound (last not-ready
  // poll); with t_ns as the upper bound the reader reconstructs the GPU-done window. 0 for
  // blocking sync (tight) or first-poll success (no bracket observed).
  put_evt(KV_COMPLETION, /*kind=*/src, /*api=*/0, now, /*bytes=*/a->bytes,
          handle, /*dst=*/a->last_poll_ns, /*src=*/0, /*first_ns=*/a->first_ns,
          /*wait_ns=*/now - entry_ns, /*n=*/(unsigned int)a->n);
}

// cudaEventRecord(event, stream): move the stream's pending copies onto the event.
static __always_inline void on_event_record(__u64 event, __u64 stream) {
  struct stream_acc *a = bpf_map_lookup_elem(&stream_state, &stream);
  if (a && a->n > 0) {
    bpf_map_update_elem(&event_state, &event, a, BPF_ANY);
    a->n = 0;
    a->bytes = 0;
    a->first_ns = 0;
    // Tell the reader these copies now belong to `event`, so an event completion
    // pairs to EXACTLY them (not the global-oldest, which crosses concurrent
    // offload streams). handle=event, dst=stream (reused field).
    put_evt(KV_BIND, /*kind=*/0, /*api=*/0, bpf_ktime_get_ns(), /*bytes=*/0,
            /*handle=*/event, /*dst=*/stream, /*src=*/0, /*first_ns=*/0,
            /*wait_ns=*/0, /*n=*/0);
  }
}
SEC("uprobe")
int BPF_KPROBE(up_event_record, __u64 event, __u64 stream) {
  on_event_record(event, stream);
  return 0;
}
SEC("uprobe")
int BPF_KPROBE(up_event_record_ptsz, __u64 event, __u64 stream) {
  on_event_record(event, stream);
  return 0;
}

static __always_inline void stash(__u64 handle, __u32 kind) {
  __u32 tid = (__u32)bpf_get_current_pid_tgid();
  struct sync_ctx c;
  __builtin_memset(&c, 0, sizeof(c)); /* zero padding: verifier rejects uninit bytes */
  c.handle = handle;
  c.entry_ns = bpf_ktime_get_ns();
  c.kind = kind;
  bpf_map_update_elem(&sync_tid, &tid, &c, BPF_ANY);
}

SEC("uprobe")
int BPF_KPROBE(up_stream_sync, __u64 stream) { stash(stream, 0); return 0; }
SEC("uprobe")
int BPF_KPROBE(up_event_sync, __u64 event) { stash(event, 1); return 0; }
SEC("uprobe")
int BPF_KPROBE(up_event_query, __u64 event) { stash(event, 2); return 0; }

SEC("uretprobe")
int BPF_KRETPROBE(uret_stream_sync, int ret) {
  __u32 tid = (__u32)bpf_get_current_pid_tgid();
  __u64 now = bpf_ktime_get_ns();
  struct sync_ctx *c = bpf_map_lookup_elem(&sync_tid, &tid);
  if (!c || c->kind != 0)
    return 0;
  struct stream_acc *a = bpf_map_lookup_elem(&stream_state, &c->handle);
  if (a && a->n > 0) {
    if (ret == 0)              // only a SUCCESSFUL sync means the copies are done
      emit_completion(c->handle, a, now, c->entry_ns, 0);
    a->n = 0; a->bytes = 0; a->first_ns = 0;  // clear either way: don't leak a failed
  }                                           // batch into the next sync's latency
  bpf_map_delete_elem(&sync_tid, &tid);
  return 0;
}

SEC("uretprobe")
int BPF_KRETPROBE(uret_event_sync, int ret) {
  __u32 tid = (__u32)bpf_get_current_pid_tgid();
  __u64 now = bpf_ktime_get_ns();
  struct sync_ctx *c = bpf_map_lookup_elem(&sync_tid, &tid);
  if (!c || c->kind != 1)
    return 0;
  struct stream_acc *a = bpf_map_lookup_elem(&event_state, &c->handle);
  if (a && a->n > 0) {
    if (ret == 0)              // only a SUCCESSFUL event sync means the copies are done
      emit_completion(c->handle, a, now, c->entry_ns, 1);
    bpf_map_delete_elem(&event_state, &c->handle);  // remove either way
  }
  bpf_map_delete_elem(&sync_tid, &tid);
  return 0;
}

SEC("uretprobe")
int BPF_KRETPROBE(uret_event_query, int ret) {
  __u32 tid = (__u32)bpf_get_current_pid_tgid();
  __u64 now = bpf_ktime_get_ns();
  struct sync_ctx *c = bpf_map_lookup_elem(&sync_tid, &tid);
  if (!c || c->kind != 2)
    return 0;
  struct stream_acc *a = bpf_map_lookup_elem(&event_state, &c->handle);
  if (a && a->n > 0) {
    if (ret == 0)            /* cudaSuccess: the event (and its copies) is complete */
      emit_completion(c->handle, a, now, c->entry_ns, 2);
    else                     /* cudaErrorNotReady: GPU not done as of `now` -> lower bound on
                                GPU-done; keep the latest so the next success closes (last_poll, ts] */
      a->last_poll_ns = now;
  }
  if (ret == 0) {            /* success consumes the event: clear state on EVERY success (incl.
                                a->n==0), matching the original cleanup so no sync_tid lingers */
    bpf_map_delete_elem(&event_state, &c->handle);
    bpf_map_delete_elem(&sync_tid, &tid);
  }
  return 0;
}

// cudaStreamDestroy(stream): drop per-stream state (handles get reused).
SEC("uprobe")
int BPF_KPROBE(up_stream_destroy, __u64 stream) {
  bpf_map_delete_elem(&stream_state, &stream);
  return 0;
}

// ---- DRIVER API (libcuda) copies ----
// vLLM V1 KV offload moves blocks with cuMemcpyBatchAsync (driver, via
// cuGetProcAddress); the runtime cudaMemcpyAsync probes above never see it.

// cuMemcpyBatchAsync(dsts[], srcs[], sizes[], count, attrs, attrsIdx, numAttrs,
// failIdx, stream). EFFICIENT capture, CHUNKED so a LARGE batch is captured whole:
// real vLLM offload batches run ~900 blocks (layers x blocks x K/V), so the old single
// 512-block read truncated ~33% of batches (~376 blocks each, the "90336" once mislabeled
// as ringbuf drops). The batch is processed in KV_MAXB(512)-block chunks. Each chunk is ONE
// ring buffer record (struct kv_batch_chunk, ~12 KB): 3 BULK reads (dst/src/sz arrays) go
// straight into the reservation, and the loader expands the chunk into one KIND_BATCH enqueue
// per block. There is no per-block loop in BPF (bpf_loop needs Linux 5.17). Up to
// KV_BATCH_MAX blocks/batch are captured; anything beyond is AUDITED in stats[ST_BATCH_TRUNC]
// (never silent). stream is arg 9 (stack at rsp+24).
#define KV_BATCH_CHUNKS 8                        /* <= 8 chunks ... */
#define KV_BATCH_MAX (KV_BATCH_CHUNKS * KV_MAXB) /* ... = 4096 blocks/batch captured (~4.6x real max) */
struct batch_in {           /* per-batch invariants (packed: BPF subprogs allow <=5 args) */
  __u64 dsts, srcs, sizes, stream, ts;
};
// Process ONE chunk (<= KV_MAXB blocks at element offset `done`): reserve a chunk record, read
// the 3 descriptor arrays into it, and submit it. It is a GLOBAL (non-static) function, so the
// verifier checks it once, independent of the caller's state. A static subprog is re-verified
// at each of the 8 chunk iterations, and on Linux 5.15 that exceeds the 1M-instruction limit
// (-E2BIG). As a global function it cannot assume anything about its arguments, so it re-bounds
// m itself. Returns the number of blocks the engine issued in this chunk as the per-stream
// accumulator must count them:
//   * submitted, or lost on a failed reservation (ST_COPY_DROP): m. A lost chunk still counts,
//     so the completion's n exceeds the captured enqueues and the reader flags the window
//     (overflowed completion), the same as a lost single-copy enqueue.
//   * discarded on a descriptor read fault (ST_BATCH_READFAIL): 0. The addresses are unknown,
//     so the blocks are not counted as copies of this stream.
__noinline __u32 do_chunk(struct batch_in *in, __u64 done, __u32 m) {
  if (!in)              // unreachable (the caller passes a stack address); the verifier types a
    return 0;           // global function's pointer argument as possibly NULL
  if (m > KV_MAXB)      // re-bound: a global function gets an unknown m
    m = KV_MAXB;
  struct kv_batch_chunk *c = bpf_ringbuf_reserve(&rb, sizeof(*c), 0);
  if (!c) {
    count_loss(ST_COPY_DROP, m);
    return m;
  }
  __u32 bytes = m * 8;
  __u64 off = done * 8;                      // byte offset into the engine's arrays
  // 3 BULK reads / chunk. CHECK the returns: on a fault (host array paged out) the record holds
  // garbage -- emitting it would produce bytes=0/dst=0 records that violate the reader's
  // nbytes>0 contract. Discard the whole chunk and AUDIT the skipped blocks.
  long e1 = bpf_probe_read_user(c->dst, bytes, (void *)(in->dsts + off));
  long e2 = bpf_probe_read_user(c->src, bytes, (void *)(in->srcs + off));
  long e3 = bpf_probe_read_user(c->sz, bytes, (void *)(in->sizes + off));
  if (e1 || e2 || e3 || force_readfail) {       /* force_readfail: test-only fault injection */
    bpf_ringbuf_discard(c, 0);
    count_loss(ST_BATCH_READFAIL, m);
    return 0;
  }
  __u64 id = bpf_get_current_pid_tgid();
  c->type = KV_BATCH_CHUNK;
  c->m = m;
  c->pid = (__u32)(id >> 32);
  c->tid = (__u32)id;
  c->ts_ns = in->ts;
  c->stream = in->stream;
  bpf_ringbuf_submit(c, 0);
  return m;
}
SEC("uprobe")
int up_memcpy_batch(struct pt_regs *ctx) {
  __u64 dsts = PT_REGS_PARM1(ctx);
  __u64 srcs = PT_REGS_PARM2(ctx);
  __u64 sizes = PT_REGS_PARM3(ctx);
  __u64 count = PT_REGS_PARM4(ctx);
  __u64 sp = PT_REGS_SP(ctx);
  __u64 stream = 0;
  bpf_probe_read_user(&stream, sizeof(stream), (void *)(sp + 24));

  // Drop the wrapper+_ptsz (and _v2) DOUBLE-FIRE: the loader attaches up_memcpy_batch to all four
  // cuMemcpyBatchAsync entry points, and a single logical call can reach us via both the non-_ptsz and
  // _ptsz variant. Mirror on_copy's per-tid duplicate check (shared last_copy map; the exact
  // (dsts,srcs,count) match disambiguates it from a single-copy entry) so one batch is counted ONCE.
  // BEFORE sampling, so a double-fire skews neither the per-block count nor the 1-in-N sampling counter.
  __u32 btid = (__u32)bpf_get_current_pid_tgid();
  __u64 bnow = bpf_ktime_get_ns();
  struct last_copy *blc = bpf_map_lookup_elem(&last_copy, &btid);
  if (blc && blc->dst == dsts && blc->src == srcs && blc->count == count && bnow - blc->ts < DEDUP_NS) {
    blc->ts = bnow;
    return 0;
  }
  struct last_copy bncp = {.dst = dsts, .src = srcs, .count = count, .ts = bnow};
  bpf_map_update_elem(&last_copy, &btid, &bncp, BPF_ANY);

  // sampling: skip whole non-sampled batches (overhead drops ~sample_n×)
  if (sample_n > 1) {
    __u32 z = 0;
    __u64 *seq = bpf_map_lookup_elem(&batch_seq, &z);
    if (seq) {
      __u64 s = (*seq)++;
      if (s % sample_n)
        return 0;
    }
  }

  __u64 ts = bpf_ktime_get_ns();
  // Capture up to KV_BATCH_MAX blocks, KV_MAXB per chunk. `done` = blocks PROCESSED (offset
  // advance); `captured` = blocks counted for the stream (see do_chunk: a chunk that faulted on
  // read is skipped + audited, so captured < done there).
  struct batch_in in = {.dsts = dsts, .srcs = srcs, .sizes = sizes, .stream = stream, .ts = ts};
  __u64 total = count < KV_BATCH_MAX ? count : KV_BATCH_MAX;
  __u64 done = 0, captured = 0;
  for (int ch = 0; ch < KV_BATCH_CHUNKS; ch++) {
    if (done >= total)
      break;
    __u64 rem = total - done;
    __u32 m = rem < KV_MAXB ? (__u32)rem : KV_MAXB;   // blocks in this chunk (<= KV_MAXB)
    captured += do_chunk(&in, done, m);               // returns m, or 0 if read-faulted
    done += m;
  }

  // feed the per-stream accumulator with the CAPTURED count (the loader sums bytes from the blocks)
  if (stream && captured) {
    struct stream_acc *a = bpf_map_lookup_elem(&stream_state, &stream);
    if (a) {
      if (a->n == 0)
        a->first_ns = ts;
      a->n += captured;
    } else {
      struct stream_acc na = {.n = captured, .bytes = 0, .first_ns = ts};
      bpf_map_update_elem(&stream_state, &stream, &na, BPF_ANY);
    }
  }
  // audit any tail beyond KV_BATCH_MAX as batch truncation (distinct from ringbuf drops + read fails)
  if (count > done)
    count_loss(ST_BATCH_TRUNC, count - done);
  return 0;
}

// cuMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, stream): driver D2H.
SEC("uprobe")
int BPF_KPROBE(up_cu_dtoh, __u64 dst, __u64 src, __u64 count, __u64 stream) {
  if (!in_runtime_copy()) // a runtime copy's own driver call (see rt_depth)
    on_copy((void *)dst, (void *)src, count, 2 /*D2H*/, 3 /*driver*/, stream);
  return 0;
}

// cuMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, stream): driver H2D.
SEC("uprobe")
int BPF_KPROBE(up_cu_htod, __u64 dst, __u64 src, __u64 count, __u64 stream) {
  if (!in_runtime_copy()) // a runtime copy's own driver call (see rt_depth)
    on_copy((void *)dst, (void *)src, count, 1 /*H2D*/, 3 /*driver*/, stream);
  return 0;
}

char LICENSE[] SEC("license") = "GPL";
