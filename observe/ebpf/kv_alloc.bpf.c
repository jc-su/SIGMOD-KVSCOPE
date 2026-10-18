// eKV KV-pool allocation probe: capture device allocations so the KV pool's base
// address is recovered from the data plane, feeding BlockAddressMap.from_alloc so the
// transfer face's address->block resolution needs NO in-process KV-tensor read.
//
// cudaMalloc(void** devPtr, size_t size) (and cuMemAlloc(CUdeviceptr*, size_t), same
// shape): the device pointer is written to *devPtr by the time the call returns, so we
// stash (devPtr, size) at entry and bpf_probe_read_user the *devPtr at the uretprobe.
// PID-filtered; min_size drops the small allocations so only pool-scale ones surface.
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "kv_alloc.h"

char LICENSE[] SEC("license") = "GPL";

const volatile __u32 target_pid = 0;          /* 0 = no filter */
const volatile __u64 min_size = 0;            /* only emit allocs >= this many bytes */

struct {
  __uint(type, BPF_MAP_TYPE_RINGBUF);
  __uint(max_entries, 1 << 20);               /* 1 MB (alloc records are tiny + rare) */
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

struct alloc_args {
  __u64 ptrptr;                               /* the void** the callee writes the addr to */
  __u64 size;
};

struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, 4096);
  __type(key, __u64);                         /* pid_tgid */
  __type(value, struct alloc_args);
} inflight SEC(".maps");

SEC("uprobe")
int alloc_enter(struct pt_regs *ctx) {
  __u64 id = bpf_get_current_pid_tgid();
  if (target_pid && (id >> 32) != target_pid)
    return 0;
  struct alloc_args a = {};
  a.ptrptr = (__u64)PT_REGS_PARM1(ctx);       /* arg0: void** devPtr */
  a.size = (__u64)PT_REGS_PARM2(ctx);         /* arg1: size_t size */
  bpf_map_update_elem(&inflight, &id, &a, BPF_ANY);
  return 0;
}

SEC("uretprobe")
int alloc_exit(struct pt_regs *ctx) {
  (void)ctx;
  __u64 id = bpf_get_current_pid_tgid();
  struct alloc_args *a = bpf_map_lookup_elem(&inflight, &id);
  if (!a)
    return 0;
  __u64 ptrptr = a->ptrptr, size = a->size;
  bpf_map_delete_elem(&inflight, &id);
  if (size < min_size)
    return 0;
  __u64 addr = 0;
  if (bpf_probe_read_user(&addr, sizeof(addr), (void *)ptrptr) < 0 || !addr)
    return 0;
  struct alloc_rec *r = bpf_ringbuf_reserve(&rb, sizeof(*r), 0);
  if (!r) {
    count_drop();
    return 0;
  }
  r->ts_ns = bpf_ktime_get_ns();
  r->pid = id >> 32;
  r->tid = (__u32)id;
  r->addr = addr;
  r->size = size;
  bpf_ringbuf_submit(r, 0);
  return 0;
}
