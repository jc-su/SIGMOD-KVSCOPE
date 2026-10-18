// Userspace loader for the eKV network-LABEL probe (optional). Attaches the
// recvfrom tracepoints (PID-filtered to the API server), polls the ring buffer,
// and prints one JSONL record per captured recv: {t_ns, pid, tid, fd, len,
// data(hex)}. The Python side reassembles per (pid,fd) and parses the HTTP
// request to recover the human-facing label (prompt, external id). Request
// identity itself is intrinsic to the block table (observe/zero_touch.py),
// not the network.
//
//   sudo ./http_label <api_server_pid>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "http_label.h"
#include "ringbuf_report.h"
#include "http_label.skel.h"
#include <bpf/libbpf.h>

static volatile int exiting;
static void on_sig(int s) {
  (void)s;
  exiting = 1;
}
static int libbpf_quiet(enum libbpf_print_level l, const char *f, va_list a) {
  if (l == LIBBPF_DEBUG)
    return 0;
  return vfprintf(stderr, f, a);
}

static int on_evt(void *ctx, void *data, size_t sz) {
  (void)ctx;
  (void)sz;
  const struct http_rec *r = data;
  printf(
      "{\"t_ns\":%llu,\"pid\":%u,\"tid\":%u,\"fd\":%lld,\"len\":%u,\"data\":\"",
      r->ts_ns, r->pid, r->tid, r->fd, r->len);
  for (unsigned int i = 0; i < r->len && i < HTTP_CAP; i++)
    printf("%02x", r->data[i]);
  printf("\"}\n");
  fflush(stdout);
  return 0;
}

int main(int argc, char **argv) {
  unsigned int pid = (argc > 1) ? (unsigned int)atoi(argv[1]) : 0;

  libbpf_set_print(libbpf_quiet);
  signal(SIGINT, on_sig);
  signal(SIGTERM, on_sig);

  struct http_label_bpf *skel = http_label_bpf__open();
  if (!skel) {
    fprintf(stderr, "open failed\n");
    return 1;
  }
  skel->rodata->target_pid = pid;
  if (http_label_bpf__load(skel)) {
    fprintf(stderr, "load failed (verifier?)\n");
    http_label_bpf__destroy(skel);
    return 1;
  }
  // The SSL_read uprobes have no auto-attach target in their SEC; skip them in the
  // skeleton attach and attach them manually to libssl below.
  bpf_program__set_autoattach(skel->progs.up_ssl_read, false);
  bpf_program__set_autoattach(skel->progs.uret_ssl_read, false);
  bpf_program__set_autoattach(skel->progs.up_ssl_read_ex, false);
  bpf_program__set_autoattach(skel->progs.uret_ssl_read_ex, false);
  if (http_label_bpf__attach(skel)) {
    fprintf(stderr, "attach failed\n");
    http_label_bpf__destroy(skel);
    return 1;
  }
  // TLS path: read the request after decryption (one layer above the socket). OpenSSL 3 / Python 3.12
  // use SSL_read_ex; older stacks use SSL_read. Attach both; whichever the engine calls fires.
  {
    const char *libssl = getenv("EKV_LIBSSL");
    if (!libssl)
      libssl = "/opt/conda/lib/libssl.so.3";
    LIBBPF_OPTS(bpf_uprobe_opts, oe, .func_name = "SSL_read", .retprobe = false);
    LIBBPF_OPTS(bpf_uprobe_opts, ox, .func_name = "SSL_read", .retprobe = true);
    LIBBPF_OPTS(bpf_uprobe_opts, oe2, .func_name = "SSL_read_ex", .retprobe = false);
    LIBBPF_OPTS(bpf_uprobe_opts, ox2, .func_name = "SSL_read_ex", .retprobe = true);
    bpf_program__attach_uprobe_opts(skel->progs.up_ssl_read, (int)pid, libssl, 0, &oe);
    bpf_program__attach_uprobe_opts(skel->progs.uret_ssl_read, (int)pid, libssl, 0, &ox);
    struct bpf_link *le =
        bpf_program__attach_uprobe_opts(skel->progs.up_ssl_read_ex, (int)pid, libssl, 0, &oe2);
    struct bpf_link *lx =
        bpf_program__attach_uprobe_opts(skel->progs.uret_ssl_read_ex, (int)pid, libssl, 0, &ox2);
    fprintf(stderr, "SSL_read[_ex] uprobe on %s: %s\n", libssl,
            (le && lx) ? "attached" : "FAILED (TLS capture disabled)");
  }
  fprintf(stderr, "eKV http_label running (pid filter=%u, Ctrl-C to stop)\n",
          pid);

  struct ring_buffer *rb =
      ring_buffer__new(bpf_map__fd(skel->maps.rb), on_evt, NULL, NULL);
  if (!rb) {
    fprintf(stderr, "ringbuf new failed\n");
    http_label_bpf__destroy(skel);
    return 1;
  }
  int rc = drain_and_report(rb, bpf_map__fd(skel->maps.drops), &exiting, 0);
  ring_buffer__free(rb);
  http_label_bpf__destroy(skel);
  return rc;
}
