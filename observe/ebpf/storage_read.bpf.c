// eKV storage-tier read probe. Times read()/pread64() syscalls so a tiered KV
// backend's lower-tier fetch (local disk / network file) is no longer invisible:
// the analysis joins each large read to the following H2D on the SAME thread and
// reports the full miss cost = tier_read + H2D. Syscall tracepoints (not a libc
// uprobe) so it catches the read regardless of buffering / libc path; PID- and
// size-filtered (large reads are KV chunks, small ones are metadata/noise).
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "storage_read.h"

char LICENSE[] SEC("license") = "GPL";

const volatile __u32 target_pid = 0;   // 0 = all PIDs
const volatile __u64 min_size = 0;     // emit only reads returning >= this many bytes

struct {
  __uint(type, BPF_MAP_TYPE_RINGBUF);
  __uint(max_entries, 1 << 22);
} rb SEC(".maps");

/* Ring buffer reservation failures. The loader prints the count at exit ("ringbuf drops: N"),
 * so a lost record is never silent. */
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, 1);
  __type(key, __u32);
  __type(value, __u64);
} drops SEC(".maps");

static __always_inline void count_drop(void) {
  __u32 k = 0;
  __u64 *d = bpf_map_lookup_elem(&drops, &k);
  if (d)
    __sync_fetch_and_add(d, 1);
}

struct rd_args {
  __u64 fd;
  __u64 count;
  __u64 entry_ns;
};

// keyed by pid_tgid: one read in flight per thread at a time (syscalls are serial per thread)
struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, 8192);
  __type(key, __u64);
  __type(value, struct rd_args);
} inflight SEC(".maps");

static __always_inline int on_enter(__u64 fd, __u64 count) {
  __u64 id = bpf_get_current_pid_tgid();
  if (target_pid && (id >> 32) != target_pid)
    return 0;
  struct rd_args a = {};
  a.fd = fd;
  a.count = count;
  a.entry_ns = bpf_ktime_get_ns();
  bpf_map_update_elem(&inflight, &id, &a, BPF_ANY);
  return 0;
}

static __always_inline int on_exit(long ret) {
  __u64 id = bpf_get_current_pid_tgid();
  struct rd_args *a = bpf_map_lookup_elem(&inflight, &id);
  if (!a)
    return 0;
  __u64 fd = a->fd, entry = a->entry_ns;
  bpf_map_delete_elem(&inflight, &id);
  if (ret <= 0 || (__u64)ret < min_size)   // short/failed read, or below the KV-chunk threshold
    return 0;
  struct read_rec *r = bpf_ringbuf_reserve(&rb, sizeof(*r), 0);
  if (!r) {
    count_drop();
    return 0;
  }
  __u64 now = bpf_ktime_get_ns();
  r->ts_ns = now;
  r->pid = id >> 32;
  r->tid = (__u32)id;
  r->fd = fd;
  r->bytes = (__u64)ret;
  r->dur_ns = now - entry;
  bpf_ringbuf_submit(r, 0);
  return 0;
}

// syscall tracepoints: sys_enter args[0]=fd, args[2]=count; sys_exit ret=bytes read
SEC("tracepoint/syscalls/sys_enter_read")
int te_read(struct trace_event_raw_sys_enter *ctx) { return on_enter(ctx->args[0], ctx->args[2]); }

SEC("tracepoint/syscalls/sys_exit_read")
int tx_read(struct trace_event_raw_sys_exit *ctx) { return on_exit(ctx->ret); }

SEC("tracepoint/syscalls/sys_enter_pread64")
int te_pread(struct trace_event_raw_sys_enter *ctx) { return on_enter(ctx->args[0], ctx->args[2]); }

SEC("tracepoint/syscalls/sys_exit_pread64")
int tx_pread(struct trace_event_raw_sys_exit *ctx) { return on_exit(ctx->ret); }
