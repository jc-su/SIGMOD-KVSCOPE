// eKV network-LABEL probe (optional): capture the inbound HTTP request a served
// engine recv()s on its API socket, transparently. Request identity is NOT taken
// here -- it is intrinsic to the location map (a request is keyed by its own
// physical blocks; see observe/zero_touch.py). This probe only adds the
// human-facing label (prompt, external request id) for billing or log correlation.
//
// recv()/recv_into() on a CPython asyncio socket lower to the recvfrom syscall
// (glibc recv == recvfrom with NULL src). We stash the user buffer ptr at
// sys_enter_recvfrom (keyed by tid) and, at sys_exit_recvfrom,
// bpf_probe_read_user the bytes the kernel just delivered. The userspace loader
// reassembles per fd and parses the HTTP request for that label; tokenizing the
// prompt to match the engine's input_ids H2D copy is an optional content
// cross-check, not the identity path.
//
// PID-filtered to the API-server process (target_pid). Variable read bounded by
// a power-of-2 bitmask (5.15 verifier; see http_label.h).
#include <vmlinux.h>          // MUST precede bpf_helpers.h (defines __sk_buff/__u32 the helper protos use)
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "http_label.h"

char LICENSE[] SEC("license") = "GPL";

const volatile __u32 target_pid =
    0; /* 0 = no filter (set to the API server pid) */

struct {
  __uint(type, BPF_MAP_TYPE_RINGBUF);
  __uint(max_entries, 1 << 24); /* 16 MB */
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

struct recv_args {
  __u64 buf;
  __s64 fd;
  __u64 cnt_ptr;   /* SSL_read_ex: user ptr to *readbytes (the count); 0 for syscall / SSL_read */
};

struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, 8192);
  __type(key, __u64); /* pid_tgid */
  __type(value, struct recv_args);
} inflight SEC(".maps");

// SSL_read nests an inner socket read() on the same thread; if it shared `inflight` the inner
// read would clobber the outer SSL_read's stash. Separate map keeps the plaintext capture clean.
struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, 8192);
  __type(key, __u64);
  __type(value, struct recv_args);
} ssl_inflight SEC(".maps");

// Reserve a ring record and copy `ret` user bytes from `buf`, tagged with `fd` (a real fd for the
// syscall path, or the SSL* pointer as a connection id for the TLS path).
static __always_inline int emit_capture(__u64 buf, long ret, __s64 fd) {
  if (ret <= 0)
    return 0;
  __u32 sz = (__u32)ret;
  if (sz > HTTP_CAP - 1)
    sz = HTTP_CAP - 1;        // clamp an oversized read to a clean prefix, not ret % HTTP_CAP
  sz &= (HTTP_CAP - 1);       // power-of-2 bound the 5.15 verifier needs (a no-op after the clamp)
  if (sz == 0)
    return 0;
  struct http_rec *r = bpf_ringbuf_reserve(&rb, sizeof(*r), 0);
  if (!r) {
    count_drop();
    return 0;
  }
  __u64 id = bpf_get_current_pid_tgid();
  r->ts_ns = bpf_ktime_get_ns();
  r->pid = id >> 32;
  r->tid = (__u32)id;
  r->fd = fd;                // fd is __s64 and http_rec.fd is now 64-bit: no connection-id truncation
  r->len = sz;
  if (bpf_probe_read_user(r->data, sz, (void *)buf) < 0) {
    bpf_ringbuf_discard(r, 0);
    return 0;
  }
  bpf_ringbuf_submit(r, 0);
  return 0;
}

// recvfrom(fd, buf, ...) and read(fd, buf, ...) share arg positions [0]=fd
// [1]=buf, so one enter/exit pair serves both. We probe BOTH because async
// servers differ: CPython asyncio recv() -> recvfrom; uvicorn/uvloop (libuv)
// reads stream sockets via read().
static __always_inline int on_enter(__u64 fd, __u64 buf) {
  __u64 id = bpf_get_current_pid_tgid();
  if (target_pid && (id >> 32) != target_pid)
    return 0;
  struct recv_args a = {};
  a.fd = (__s64)fd;
  a.buf = buf;
  bpf_map_update_elem(&inflight, &id, &a, BPF_ANY);
  return 0;
}

static __always_inline int on_exit(long ret) {
  __u64 id = bpf_get_current_pid_tgid();
  struct recv_args *a = bpf_map_lookup_elem(&inflight, &id);
  if (!a)
    return 0;
  __u64 buf = a->buf;
  __s64 fd = a->fd;
  bpf_map_delete_elem(&inflight, &id);
  return emit_capture(buf, ret, fd);
}

SEC("tracepoint/syscalls/sys_enter_recvfrom")
int e_recvfrom(struct trace_event_raw_sys_enter *ctx) {
  return on_enter(ctx->args[0], ctx->args[1]);
}
SEC("tracepoint/syscalls/sys_exit_recvfrom")
int x_recvfrom(struct trace_event_raw_sys_exit *ctx) {
  return on_exit(ctx->ret);
}
SEC("tracepoint/syscalls/sys_enter_read")
int e_read(struct trace_event_raw_sys_enter *ctx) {
  return on_enter(ctx->args[0], ctx->args[1]);
}
SEC("tracepoint/syscalls/sys_exit_read")
int x_read(struct trace_event_raw_sys_exit *ctx) { return on_exit(ctx->ret); }

// TLS path. SSL_read(SSL *ssl, void *buf, int num) returns the DECRYPTED plaintext in buf, so this
// reads the request one layer above the encrypted socket, making the capture TLS-immune. The SSL*
// pointer is used as a per-connection id in the fd field; the reader reassembles on it like an fd.
SEC("uprobe")
int BPF_KPROBE(up_ssl_read, void *ssl, void *buf, int num) {
  __u64 id = bpf_get_current_pid_tgid();
  if (target_pid && (id >> 32) != target_pid)
    return 0;
  struct recv_args a = {.buf = (__u64)buf, .fd = (__s64)(long)ssl};
  bpf_map_update_elem(&ssl_inflight, &id, &a, BPF_ANY);
  return 0;
}
SEC("uretprobe")
int BPF_KRETPROBE(uret_ssl_read, int ret) {   // ret = decrypted bytes; buf holds the plaintext
  __u64 id = bpf_get_current_pid_tgid();
  struct recv_args *a = bpf_map_lookup_elem(&ssl_inflight, &id);
  if (!a)
    return 0;
  __u64 buf = a->buf;
  __s64 fd = a->fd;
  bpf_map_delete_elem(&ssl_inflight, &id);
  return emit_capture(buf, ret, fd);
}

// SSL_read_ex(SSL *ssl, void *buf, size_t num, size_t *readbytes): the path OpenSSL 3 / Python 3.12
// uses. Plaintext is in buf; the COUNT is written to *readbytes; the return is 1 on success.
SEC("uprobe")
int BPF_KPROBE(up_ssl_read_ex, void *ssl, void *buf, __u64 num, void *readbytes) {
  __u64 id = bpf_get_current_pid_tgid();
  if (target_pid && (id >> 32) != target_pid)
    return 0;
  struct recv_args a = {.buf = (__u64)buf, .fd = (__s64)(long)ssl, .cnt_ptr = (__u64)readbytes};
  bpf_map_update_elem(&ssl_inflight, &id, &a, BPF_ANY);
  return 0;
}
SEC("uretprobe")
int BPF_KRETPROBE(uret_ssl_read_ex, int ret) {
  __u64 id = bpf_get_current_pid_tgid();
  struct recv_args *a = bpf_map_lookup_elem(&ssl_inflight, &id);
  if (!a)
    return 0;
  __u64 buf = a->buf, cp = a->cnt_ptr;
  __s64 fd = a->fd;
  bpf_map_delete_elem(&ssl_inflight, &id);
  if (ret != 1 || !cp)
    return 0;
  __u64 n = 0;
  if (bpf_probe_read_user(&n, sizeof(n), (void *)cp) < 0)
    return 0;
  return emit_capture(buf, (long)n, fd);
}
