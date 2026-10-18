// eKV eBPF KV-copy collector (loader). Enqueue (cudaMemcpyAsync[_ptsz]) +
// completion via stream sync AND CUDA events (cudaEventRecord[_ptsz] +
// cudaEventSynchronize / cudaEventQuery) + cudaStreamDestroy cleanup. Reads one BPF ring
// buffer and reports the loss counters at exit. Linux 5.8+ with BTF.
//   sudo ./kv_copy <libcudart.so>[:<libcudart.so>...] [pid]
// Give every libcudart the target maps: a uprobe sees only calls into its own file.
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "kv_copy.h"
#include "kv_copy.skel.h"

static volatile int exiting;
static void on_sig(int s) { exiting = 1; }

static int libbpf_quiet(enum libbpf_print_level lvl, const char *f, va_list a) {
  if (lvl == LIBBPF_DEBUG)
    return 0;
  return vfprintf(stderr, f, a);
}

static const char *kind_str(unsigned int k) {
  switch (k) {
  case 0: return "H2H";
  case 1: return "H2D";
  case 2: return "D2H";
  case 3: return "D2D";
  case 4: return "DEFAULT";  /* cudaMemcpyDefault: direction inferred from pointers */
  case 9: return "BATCH";    /* driver cuMemcpyBatchAsync: direction per-element */
  default: return "UNKNOWN"; /* schema drift -> the reader raises (not silently DEFAULT) */
  }
}

static const char *api_str(unsigned int a) {
  switch (a) {
  case 1: return "cudaMemcpyAsync_ptsz";
  case 2: return "cuMemcpyBatchAsync";
  case 3: return "cuMemcpy(driver)";
  default: return "cudaMemcpyAsync";
  }
}
static const char *comp_src(unsigned int k) {
  switch (k) {
  case 1: return "event-sync";
  case 2: return "event-query";
  case 0: return "stream-sync";
  default: return "UNKNOWN";
  }
}

// Binary transport (EKV_BINARY=1): write the raw event struct to stdout instead of
// formatting JSON in the hot path. A 12-byte self-describing header (magic, version,
// record size) lets the Python reader validate the layout; then fixed-size records.
// This removes per-event sprintf + the Python json.loads, the cost at high rates.
static int g_binary = 0;
static void emit_binary(const struct kv_copy_evt *e) {
  static int hdr = 0;
  if (!hdr) {
    unsigned int ver = 1, sz = (unsigned int)sizeof(struct kv_copy_evt);
    fwrite("eKV1", 1, 4, stdout);
    fwrite(&ver, sizeof(ver), 1, stdout);
    fwrite(&sz, sizeof(sz), 1, stdout);
    hdr = 1;
  }
  fwrite(e, sizeof(*e), 1, stdout);  // flushed per poll-batch in main, not per event
}

// Format one fixed-size record (enqueue / bind / completion). Batch blocks arrive here as
// KV_ENQUEUE records with kind=BATCH, built by expand_chunk. No per-record fflush here -- the
// main loop flushes once per poll pass (a batch emits hundreds of records per pass).
static void handle_evt(const struct kv_copy_evt *e) {
  if (g_binary) {
    emit_binary(e);
    return;
  }
  if (e->type == KV_ENQUEUE) {
    printf("{\"type\":\"enqueue\",\"t_ns\":%llu,\"pid\":%u,\"tid\":%u,"
           "\"bytes\":%llu,\"kind\":\"%s\",\"api\":\"%s\",\"stream\":%llu,"
           "\"dst\":%llu,\"src\":%llu}\n",
           e->ts_ns, e->pid, e->tid, e->bytes, kind_str(e->kind),
           api_str(e->api), e->handle, e->dst, e->src);
  } else if (e->type == KV_BIND) {
    printf("{\"type\":\"bind\",\"t_ns\":%llu,\"pid\":%u,\"tid\":%u,"
           "\"event\":%llu,\"stream\":%llu}\n",
           e->ts_ns, e->pid, e->tid, e->handle, e->dst);
  } else {
    unsigned long long lat = e->ts_ns > e->first_ns ? e->ts_ns - e->first_ns : 0;
    // GPU-done lies in (ready_lo_ns, t_ns]. On the polling (event-query) path ready_lo_ns is the
    // last not-ready poll, so ready_window_ns (the poll gap) is the residual uncertainty on the
    // true GPU-ready instant. 0 = a blocking sync (t_ns already == the GPU-completion signal) or a
    // first-poll success (no lower bound observed; the floor is then first_ns / latency_ns).
    unsigned long long rlo = e->dst;
    unsigned long long rwin = (rlo && e->ts_ns > rlo) ? e->ts_ns - rlo : 0;
    printf("{\"type\":\"completion\",\"via\":\"%s\",\"t_ns\":%llu,\"pid\":%u,"
           "\"tid\":%u,\"n\":%u,\"bytes\":%llu,\"latency_ns\":%llu,"
           "\"wait_ns\":%llu,\"ready_lo_ns\":%llu,\"ready_window_ns\":%llu,\"handle\":%llu}\n",
           comp_src(e->kind), e->ts_ns, e->pid, e->tid, e->n, e->bytes, lat,
           e->wait_ns, rlo, rwin, e->handle);
  }
}

// Expand one batch chunk into one KIND_BATCH enqueue per block, the same per-copy record the
// single-copy probes emit, so readers see one record layout.
static void expand_chunk(const struct kv_batch_chunk *c) {
  for (unsigned int i = 0; i < c->m; i++) {
    struct kv_copy_evt e = {
        .type = KV_ENQUEUE, .kind = KIND_BATCH, .api = 2 /*cuMemcpyBatchAsync*/,
        .pid = c->pid, .tid = c->tid, .n = 1, .ts_ns = c->ts_ns, .bytes = c->sz[i],
        .first_ns = c->ts_ns, .wait_ns = 0, .handle = c->stream,
        .dst = c->dst[i], .src = c->src[i],
    };
    handle_evt(&e);
  }
}

// Check that a ring buffer record has the size of its type. A mismatch means the BPF object and
// this loader were built from different kv_copy.h versions.
static int check_size(const char *name, size_t got, size_t want) {
  if (got == want)
    return 0;
  fprintf(stderr, "kv_copy: %s record has %zu bytes, expected %zu (kv_copy.h drift)\n", name,
          got, want);
  return -EINVAL;
}

// Ring buffer callback: dispatch on the record type. An unknown type or a size mismatch returns
// an error, which stops ring_buffer__poll and makes the loader exit non-zero.
static int on_ringbuf(void *ctx, void *data, size_t sz) {
  unsigned int type = *(const unsigned int *)data;
  switch (type) {
  case KV_ENQUEUE:
  case KV_COMPLETION:
  case KV_BIND:
    if (check_size("copy", sz, sizeof(struct kv_copy_evt)))
      return -EINVAL;
    handle_evt(data);
    return 0;
  case KV_BATCH_CHUNK: {
    if (check_size("batch chunk", sz, sizeof(struct kv_batch_chunk)))
      return -EINVAL;
    const struct kv_batch_chunk *c = data;
    if (c->m == 0 || c->m > KV_MAXB) {
      fprintf(stderr, "kv_copy: batch chunk with m=%u (expected 1..%d)\n", c->m, KV_MAXB);
      return -EINVAL;
    }
    expand_chunk(c);
    return 0;
  }
  case KV_META_BLOB:
    if (check_size("metadata", sz, sizeof(struct kv_meta_blob)))
      return -EINVAL;
    break;
  default:
    fprintf(stderr, "kv_copy: unknown ring buffer record type %u\n", type);
    return -EINVAL;
  }
  // host content of a small H2D copy (attention metadata). Emit dst + the int32
  // values so the analysis can identify (stable dst) + parse (block_table/cu_seqlens).
  const struct kv_meta_blob *m = data;
  unsigned int n = m->size / 4;
  if (n > KV_META_MAX / 4)
    n = KV_META_MAX / 4;
  const int *vals = (const int *)m->data;
  printf("{\"type\":\"meta\",\"t_ns\":%llu,\"dst\":%llu,\"size\":%u,\"vals\":[",
         m->ts_ns, m->dst, m->size);
  for (unsigned int i = 0; i < n; i++)
    printf("%s%d", i ? "," : "", vals[i]);
  printf("]}\n");
  return 0;
}

static int attach(struct bpf_program *p, const char *path, int pid,
                  const char *fn, bool ret, bool required) {
  LIBBPF_OPTS(bpf_uprobe_opts, o, .func_name = fn, .retprobe = ret);
  struct bpf_link *l = bpf_program__attach_uprobe_opts(p, pid, path, 0, &o);
  if (!l) {
    fprintf(stderr, "%s %s%s\n", required ? "MISS(req)" : "skip", fn,
            ret ? " [ret]" : "");
    return 0;
  }
  fprintf(stderr, "attached %s%s\n", fn, ret ? " [ret]" : "");
  return 1;
}

// Attach the runtime-API probes to one libcudart. cudaMemcpyAsync is required: a runtime
// without it is not a CUDA runtime, so that is an error (returns non-zero).
static int attach_runtime(struct kv_copy_bpf *skel, const char *path, int pid) {
  fprintf(stderr, "runtime library %s\n", path);
  // The entry and return probes come in pairs: a missing return probe would leave the tid's
  // rt_depth set, and the driver probes would then skip every later copy on that thread.
  if (!attach(skel->progs.up_memcpy_async, path, pid, "cudaMemcpyAsync", false, true) ||
      !attach(skel->progs.uret_memcpy_async, path, pid, "cudaMemcpyAsync", true, true)) {
    fprintf(stderr, "kv_copy: cannot attach cudaMemcpyAsync in %s\n", path);
    return 1;
  }
  if (attach(skel->progs.up_memcpy_async_ptsz, path, pid, "cudaMemcpyAsync_ptsz", false, false) &&
      !attach(skel->progs.uret_memcpy_async, path, pid, "cudaMemcpyAsync_ptsz", true, true)) {
    fprintf(stderr, "kv_copy: cannot attach the cudaMemcpyAsync_ptsz return probe in %s\n", path);
    return 1;
  }
  // metadata-content capture (EKV_META) on the runtime H2D entry point
  attach(skel->progs.up_meta, path, pid, "cudaMemcpyAsync", false, false);
  attach(skel->progs.up_meta, path, pid, "cudaMemcpyAsync_ptsz", false, false);
  // stream-sync completion
  attach(skel->progs.up_stream_sync, path, pid, "cudaStreamSynchronize", false, false);
  attach(skel->progs.uret_stream_sync, path, pid, "cudaStreamSynchronize", true, false);
  // event-based completion (the real vLLM/LMCache path)
  attach(skel->progs.up_event_record, path, pid, "cudaEventRecord", false, false);
  attach(skel->progs.up_event_record_ptsz, path, pid, "cudaEventRecord_ptsz", false, false);
  // PyTorch records via cudaEventRecordWithFlags (same first 2 args: event, stream)
  attach(skel->progs.up_event_record, path, pid, "cudaEventRecordWithFlags", false, false);
  attach(skel->progs.up_event_record_ptsz, path, pid, "cudaEventRecordWithFlags_ptsz", false, false);
  attach(skel->progs.up_event_sync, path, pid, "cudaEventSynchronize", false, false);
  attach(skel->progs.uret_event_sync, path, pid, "cudaEventSynchronize", true, false);
  attach(skel->progs.up_event_query, path, pid, "cudaEventQuery", false, false);
  attach(skel->progs.uret_event_query, path, pid, "cudaEventQuery", true, false);
  // handle reuse cleanup
  attach(skel->progs.up_stream_destroy, path, pid, "cudaStreamDestroy", false, false);
  return 0;
}

int main(int argc, char **argv) {
  if (getenv("EKV_BINARY") && getenv("EKV_META")) {
    fprintf(stderr, "kv_copy: EKV_BINARY and EKV_META are mutually exclusive "
                    "(binary copy records and JSON metadata would interleave on stdout)\n");
    return 2;
  }

  if (argc < 2) {
    fprintf(stderr, "usage: %s <libcudart.so>[:<libcudart.so>...] [pid]\n", argv[0]);
    return 2;
  }
  int pid = (argc > 2) ? atoi(argv[2]) : -1;

  libbpf_set_print(libbpf_quiet);
  signal(SIGINT, on_sig);
  signal(SIGTERM, on_sig);

  struct kv_copy_bpf *skel = kv_copy_bpf__open();
  if (!skel) {
    fprintf(stderr, "open failed\n");
    return 1;
  }
  const char *sn = getenv("EKV_SAMPLE");   // capture 1-in-N offload batches (overhead knob)
  if (sn)
    skel->rodata->sample_n = (unsigned int)atoi(sn);
  if (getenv("EKV_META"))                  // capture host content of small H2D metadata copies
    skel->rodata->meta_on = 1;
  if (getenv("EKV_FORCE_READFAIL"))        // test-only: force the batch read-fault path
    skel->rodata->force_readfail = 1;
  if (kv_copy_bpf__load(skel)) {
    fprintf(stderr, "load failed\n");
    kv_copy_bpf__destroy(skel);
    return 1;
  }

  // EKV_DRIVER_ONLY: attach ONLY the driver batch offload path (cuMemcpyBatchAsync),
  // skipping the high-rate runtime cudaMemcpyAsync flood. vLLM's KV offload IS the
  // driver-batch path, so this captures the KV transfers with far less overhead.
  // (Blanket runtime attach is for the per-copy worst case.)
  int driver_only = getenv("EKV_DRIVER_ONLY") != NULL;
  if (!driver_only) {
    // One process can map several CUDA runtimes (torch cu130 calls libcudart.so.13 while a
    // dependency maps libcudart.so.12), and a uprobe sees only calls into its own file, so
    // every runtime the caller names gets the full probe set.
    char *paths = strdup(argv[1]);
    int runtimes = 0;
    for (char *path = strtok(paths, ":"); path; path = strtok(NULL, ":"), runtimes++) {
      if (attach_runtime(skel, path, pid)) {
        free(paths);
        kv_copy_bpf__destroy(skel);
        return 1;
      }
    }
    free(paths);
    if (!runtimes) {
      fprintf(stderr, "kv_copy: no libcudart path given\n");
      kv_copy_bpf__destroy(skel);
      return 2;
    }
  }

  // DRIVER API (libcuda): vLLM KV offload uses cuMemcpyBatchAsync; the runtime
  // probes above never see it. Also cover the single driver async copies and
  // driver-side completion (cuStreamSynchronize / cuEventSynchronize).
  const char *libcuda = getenv("EKV_LIBCUDA");
  if (!libcuda)
    libcuda = "/usr/lib/x86_64-linux-gnu/libcuda.so";
  int drv = 0;
  drv += attach(skel->progs.up_memcpy_batch, libcuda, pid, "cuMemcpyBatchAsync", false, false);
  attach(skel->progs.up_memcpy_batch, libcuda, pid, "cuMemcpyBatchAsync_ptsz", false, false);
  attach(skel->progs.up_memcpy_batch, libcuda, pid, "cuMemcpyBatchAsync_v2", false, false);
  attach(skel->progs.up_memcpy_batch, libcuda, pid, "cuMemcpyBatchAsync_v2_ptsz", false, false);
  attach(skel->progs.up_cu_dtoh, libcuda, pid, "cuMemcpyDtoHAsync_v2", false, false);
  attach(skel->progs.up_cu_dtoh, libcuda, pid, "cuMemcpyDtoHAsync_v2_ptsz", false, false);
  attach(skel->progs.up_cu_htod, libcuda, pid, "cuMemcpyHtoDAsync_v2", false, false);
  attach(skel->progs.up_cu_htod, libcuda, pid, "cuMemcpyHtoDAsync_v2_ptsz", false, false);
  // metadata-content capture (EKV_META) on the driver H2D entry point — this is the
  // universal bottom: runtime cudaMemcpyAsync H2D funnels through cuMemcpyHtoDAsync_v2,
  // so attaching here catches the metadata copy in DRIVER_ONLY mode as well.
  attach(skel->progs.up_meta_drv, libcuda, pid, "cuMemcpyHtoDAsync_v2", false, false);
  attach(skel->progs.up_meta_drv, libcuda, pid, "cuMemcpyHtoDAsync_v2_ptsz", false, false);
  attach(skel->progs.up_stream_sync, libcuda, pid, "cuStreamSynchronize", false, false);
  attach(skel->progs.uret_stream_sync, libcuda, pid, "cuStreamSynchronize", true, false);
  attach(skel->progs.up_event_sync, libcuda, pid, "cuEventSynchronize", false, false);
  attach(skel->progs.uret_event_sync, libcuda, pid, "cuEventSynchronize", true, false);
  attach(skel->progs.up_event_record, libcuda, pid, "cuEventRecord", false, false);
  // driver busy-poll completion + WithFlags record + per-thread-default-stream
  // (_ptsz) variants — the runtime side covers these; the driver side must too.
  attach(skel->progs.up_event_query, libcuda, pid, "cuEventQuery", false, false);
  attach(skel->progs.uret_event_query, libcuda, pid, "cuEventQuery", true, false);
  attach(skel->progs.up_event_record, libcuda, pid, "cuEventRecordWithFlags", false, false);
  attach(skel->progs.up_stream_sync, libcuda, pid, "cuStreamSynchronize_ptsz", false, false);
  attach(skel->progs.uret_stream_sync, libcuda, pid, "cuStreamSynchronize_ptsz", true, false);
  attach(skel->progs.up_event_record, libcuda, pid, "cuEventRecord_ptsz", false, false);
  attach(skel->progs.up_event_record, libcuda, pid, "cuEventRecordWithFlags_ptsz", false, false);
  fprintf(stderr, "driver-API (libcuda) copy probes: %s\n",
          drv ? "attached" : "MISS (cuMemcpyBatchAsync)");

  if (driver_only && !drv) {
    fprintf(stderr, "no enqueue uprobe attached\n");
    kv_copy_bpf__destroy(skel);
    return 1;
  }

  struct ring_buffer *rb =
      ring_buffer__new(bpf_map__fd(skel->maps.rb), on_ringbuf, NULL, NULL);
  if (!rb) {
    fprintf(stderr, "ringbuf new failed\n");
    kv_copy_bpf__destroy(skel);
    return 1;
  }
  g_binary = getenv("EKV_BINARY") != NULL;
  fprintf(stderr, "eKV kv_copy collector running (%s, ring buffer %u MB, Ctrl-C to stop)\n",
          g_binary ? "binary" : "jsonl", EKV_RINGBUF_BYTES >> 20);
  // One transport: the BPF ring buffer. Block on its epoll fd and flush stdout once per pass.
  int rc = 0;
  while (!exiting) {
    int n = ring_buffer__poll(rb, 100);
    if (n == -EINTR)
      continue;
    if (n < 0) {
      fprintf(stderr, "kv_copy: ring buffer poll failed: %d\n", n);
      rc = 1;
      break;
    }
    if (n > 0)
      fflush(stdout);
  }
  // Final pass: emit the records submitted before the stop signal.
  if (rc == 0) {
    int n = ring_buffer__consume(rb);
    if (n < 0) {
      fprintf(stderr, "kv_copy: final ring buffer drain failed: %d\n", n);
      rc = 1;
    }
  }
  fflush(stdout);
  // Capture-completeness audit, four DISTINCT loss modes (each counted, none silent):
  //   ringbuf drops (copy records) = copy records lost on a failed reservation (transport loss)
  //   batch truncation             = blocks in a cuMemcpyBatchAsync beyond KV_BATCH_MAX
  //   batch read faults            = blocks in chunks whose descriptor read faulted
  //   ringbuf drops (metadata)     = metadata blobs lost on a failed reservation (EKV_META)
  int sfd = bpf_map__fd(skel->maps.stats);
  __u64 loss[ST_COUNT];
  for (__u32 k = 0; k < ST_COUNT; k++) {
    if (bpf_map_lookup_elem(sfd, &k, &loss[k])) {
      fprintf(stderr, "kv_copy: cannot read loss counter %u: errno %d\n", k, errno);
      ring_buffer__free(rb);
      kv_copy_bpf__destroy(skel);
      return 1;
    }
  }
  fprintf(stderr, "ringbuf drops (copy records): %llu\n", loss[ST_COPY_DROP]);
  fprintf(stderr, "batch truncation (blocks beyond cap): %llu\n", loss[ST_BATCH_TRUNC]);
  fprintf(stderr, "batch read faults (chunk skipped): %llu\n", loss[ST_BATCH_READFAIL]);
  fprintf(stderr, "ringbuf drops (metadata): %llu\n", loss[ST_META_DROP]);

  ring_buffer__free(rb);
  kv_copy_bpf__destroy(skel);
  return rc;
}
