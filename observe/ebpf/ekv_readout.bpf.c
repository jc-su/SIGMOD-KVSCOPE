// eKV zero-touch READOUT: lift the GPU pass's published capture/timer buffer out of
// the engine process with bpf_probe_read_user — the eBPF realization of the readout
// (the /proc/<pid>/mem variant is observe/zero_touch.py). The pass (EKV_PASS_AUTO)
// writes the block_table rows (and per-slot clock64 cycles) into a host-pinned,
// device-mapped buffer and publishes its host VA. A uprobe on a function the engine
// calls each step (default cuLaunchKernel — every Triton attention launch hits it)
// fires in the ENGINE's address space, where bpf_probe_read_user can read that VA.
// PID-filtered; throttled to one snapshot per interval_ns so a hot launch path costs
// at most one read per window. Same 5.15 verifier idiom as http_label/kv_copy: the
// variable read is bounded by a power-of-2 bitmask.
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "ekv_readout.h"

char LICENSE[] SEC("license") = "GPL";

const volatile __u32 target_pid = 0;          /* 0 = no filter (set to the engine pid) */
const volatile __u64 cap_va = 0;              /* published host_va of the capture buffer */
const volatile __u32 cap_bytes = 0;
const volatile __u64 timer_va = 0;            /* published timer_host_va (0 if timer off) */
const volatile __u32 timer_bytes = 0;
const volatile __u64 interval_ns = 50000000;  /* 50 ms throttle between snapshots */

struct {
  __uint(type, BPF_MAP_TYPE_RINGBUF);
  __uint(max_entries, 1 << 24);               /* 16 MB */
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

struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, 1);
  __type(key, __u32);
  __type(value, __u64);                       /* last-emit ts, for throttling */
} gate SEC(".maps");

static __always_inline void emit(__u32 kind, __u64 va, __u32 nbytes) {
  if (!va || !nbytes)
    return;
  /* power-of-2 bound so the variable-size probe_read verifies on 5.15; EKV_BUF_CAP is
   * larger than any real buffer, so the mask preserves nbytes exactly. */
  __u32 sz = nbytes & (EKV_BUF_CAP - 1);
  if (sz == 0)
    return;
  struct ekv_buf_rec *r = bpf_ringbuf_reserve(&rb, sizeof(*r), 0);
  if (!r) {
    count_drop();
    return;
  }
  r->kind = kind;
  r->len = sz;
  r->ts_ns = bpf_ktime_get_ns();
  if (bpf_probe_read_user(r->data, sz, (void *)va) < 0) {
    bpf_ringbuf_discard(r, 0);
    return;
  }
  bpf_ringbuf_submit(r, 0);
}

SEC("uprobe")
int ekv_tick(struct pt_regs *ctx) {
  (void)ctx;
  __u64 id = bpf_get_current_pid_tgid();
  if (target_pid && (id >> 32) != target_pid)
    return 0;
  __u32 k = 0;
  __u64 now = bpf_ktime_get_ns();
  __u64 *last = bpf_map_lookup_elem(&gate, &k);
  if (last) {
    if (*last && now - *last < interval_ns)
      return 0;
    *last = now;
  }
  emit(0, cap_va, cap_bytes);
  emit(1, timer_va, timer_bytes);
  return 0;
}
