/* Record layouts shared by the BPF program and the userspace loader.
 * Plain C scalar types so the header compiles unchanged on both sides (LP64).
 *
 * Transport: one BPF ring buffer (BPF_MAP_TYPE_RINGBUF, Linux 5.8+). Every record starts with a
 * u32 type field, so the loader dispatches on the first word. A failed reservation is counted in
 * the stats map (ST_COPY_DROP or ST_META_DROP) and the loader reports the counts at exit. */
#ifndef EKV_KV_COPY_H
#define EKV_KV_COPY_H

/* Record type (first u32 of every ring buffer record). */
#define KV_ENQUEUE     0  /* cudaMemcpyAsync issued (enqueue) */
#define KV_COMPLETION  1  /* a stream/event sync (or first successful event-query) returned ->
                             the batch on it is done. ts_ns = when the host learned it; for the
                             event-query path `dst` carries the ready LOWER bound (last not-ready
                             poll) so the reader brackets GPU-done to (dst, ts_ns]. */
#define KV_BIND        2  /* cudaEventRecord(ev,stream): the stream's pending copies
                             now belong to event `ev` (handle=ev, dst=stream). Lets
                             the reader pair an event completion to EXACTLY those
                             copies instead of the global-oldest, which mis-assigns
                             latency across concurrent offload streams. */
#define KV_BATCH_CHUNK 3  /* one chunk (<= KV_MAXB blocks) of a cuMemcpyBatchAsync: the loader
                             expands it into one KV_ENQUEUE record per block with kind=KIND_BATCH. */
#define KV_META_BLOB   4  /* host content of a small H2D copy (EKV_META only) */

/* evt.kind sentinel for driver-API batch copies (cuMemcpyBatchAsync): direction
 * is per-element and unknown to the probe, so the fuser resolves whichever of
 * dst/src lands in a KV block. */
#define KIND_BATCH    9

/* Indexes of the BPF stats array. Each counts one distinct loss mode; none is silent. */
#define ST_COPY_DROP      0  /* copy records lost on a failed ring buffer reservation. A lost
                                batch chunk counts its blocks (one per enqueue it would emit). */
#define ST_BATCH_TRUNC    1  /* blocks beyond KV_BATCH_MAX in one cuMemcpyBatchAsync */
#define ST_BATCH_READFAIL 2  /* blocks in chunks skipped because a descriptor read faulted */
#define ST_META_DROP      3  /* metadata blobs lost on a failed ring buffer reservation */
#define ST_COUNT          4

/* Ring buffer size in bytes (power of two, multiple of the page size). A batch chunk record is
 * about 12 KB and a metadata blob about 32 KB, so 64 MB holds about 5000 full chunks
 * (2.6M blocks) before the loader must drain. */
#define EKV_RINGBUF_BYTES (1u << 26)

/* One chunk of a cuMemcpyBatchAsync: the three descriptor arrays, read with one
 * bpf_probe_read_user per array straight into the ring buffer reservation. Only the first
 * `m` entries of each array are valid. KV_MAXB caps one chunk; a batch beyond KV_BATCH_MAX
 * (kv_copy.bpf.c) is counted as truncated. */
#define KV_MAXB 512
struct kv_batch_chunk {
  unsigned int type;          /* KV_BATCH_CHUNK */
  unsigned int m;             /* valid entries in dst/src/sz (1..KV_MAXB) */
  unsigned int pid;
  unsigned int tid;
  unsigned long long ts_ns;   /* batch entry ts (same for every chunk of one batch) */
  unsigned long long stream;  /* CUDA stream handle (join key) */
  unsigned long long dst[KV_MAXB];
  unsigned long long src[KV_MAXB];
  unsigned long long sz[KV_MAXB];
};

/* Metadata blob (KV_META_BLOB): the CONTENT of a small H2D copy, the engine-agnostic
 * way to recover request<->block WITHOUT engine-specific uprobes. Paged/contiguous
 * attention kernels can't run without a KV-location map (block_table/page_table or
 * cu_seqlens); the engine CPU-builds it and copies it H2D every step. We bulk-read the
 * HOST SOURCE of those small H2D copies and let the loader identify (stable pinned dst)
 * and parse the layout. Gated by the EKV_META rodata flag so it never fires on the hot
 * path unless requested. */
#define KV_META_MAX   32768       /* bytes of host content captured; MUST be power-of-2
                                     (on_meta bounds the read with & (KV_META_MAX-1)).
                                     32K covers the block_table (request<->block map):
                                     num_reqs * max_blocks_per_req * 4B, e.g. 48*128*4=24576.
                                     This is the capture overhead/coverage knob -- larger
                                     batches/longer seqs need more; small copies still read
                                     only `count` bytes (the read size is count-bounded). */
struct kv_meta_blob {
  unsigned int type;              /* KV_META_BLOB */
  unsigned int size;              /* bytes actually read (<= KV_META_MAX) */
  unsigned long long ts_ns;
  unsigned long long dst;         /* device destination (the stable pinned buffer id) */
  unsigned long long pid_tid;
  unsigned char data[KV_META_MAX];/* the host-source bytes (e.g. an int32 metadata array) */
};

/* Fixed-size copy record (KV_ENQUEUE, KV_COMPLETION, KV_BIND). The loader also builds one per
 * block of a KV_BATCH_CHUNK, so the output stream has a single per-copy record layout. */
struct kv_copy_evt {
  unsigned int type;          /* KV_ENQUEUE | KV_COMPLETION | KV_BIND */
  unsigned int kind;          /* enqueue: cudaMemcpyKind 0 H2H,1 H2D,2 D2H,3 D2D */
  unsigned int api;           /* enqueue: 0 cudaMemcpyAsync, 1 _ptsz */
  unsigned int pid;
  unsigned int tid;
  unsigned int n;             /* completion: #copies in the synced batch (enqueue: 1) */
  unsigned long long ts_ns;   /* enqueue ts, or completion (sync-return) ts */
  unsigned long long bytes;   /* enqueue: this copy; completion: total bytes in batch */
  unsigned long long first_ns;/* completion: ts of the first enqueue in the batch */
  unsigned long long wait_ns; /* completion: ts - sync_entry  (host-paid GPU wait) */
  unsigned long long handle;  /* CUDA stream handle (join key) */
  unsigned long long dst;     /* enqueue: device dst. completion(event-query): ready lower-bound
                                 (ts of the last not-ready poll); 0 for blocking sync. */
  unsigned long long src;     /* enqueue only */
};

#endif /* EKV_KV_COPY_H */
