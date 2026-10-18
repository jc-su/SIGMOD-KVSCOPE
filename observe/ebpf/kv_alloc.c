// Userspace loader for the eKV KV-pool allocation probe. Attaches a uprobe +
// uretprobe to a device-allocation entry point (default libcudart.so:cudaMalloc),
// PID-filtered, and prints one JSONL record per allocation: {t_ns, pid, addr, size}.
// The analysis takes the pool-scale allocation as the KV base for
// BlockAddressMap.from_alloc — address->block with no in-process tensor read.
//
//   sudo ./kv_alloc <pid> [binary:symbol] [-m min_bytes]
#include <errno.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "kv_alloc.h"
#include "ringbuf_report.h"
#include "kv_alloc.skel.h"
#include <bpf/libbpf.h>

static volatile int exiting;
static void on_sig(int s) { (void)s; exiting = 1; }
static int quiet(enum libbpf_print_level l, const char *f, va_list a) {
  if (l == LIBBPF_DEBUG) return 0;
  return vfprintf(stderr, f, a);
}

static int on_evt(void *ctx, void *data, size_t sz) {
  (void)ctx; (void)sz;
  const struct alloc_rec *r = data;
  printf("{\"t_ns\":%llu,\"pid\":%u,\"addr\":%llu,\"size\":%llu}\n",
         r->ts_ns, r->pid, r->addr, r->size);
  fflush(stdout);
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <pid> [binary:symbol] [-m min_bytes]\n", argv[0]);
    return 2;
  }
  unsigned int pid = (unsigned int)atoi(argv[1]);
  const char *probe = "libcudart.so:cudaMalloc";
  unsigned long long min_size = 0;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "-m") && i + 1 < argc) min_size = strtoull(argv[++i], NULL, 10);
    else probe = argv[i];
  }
  const char *colon = strrchr(probe, ':');
  if (!colon) { fprintf(stderr, "probe must be binary:symbol\n"); return 2; }
  char bin[512];
  size_t bl = (size_t)(colon - probe);
  if (bl >= sizeof bin) bl = sizeof bin - 1;
  memcpy(bin, probe, bl); bin[bl] = 0;
  const char *sym = colon + 1;

  libbpf_set_print(quiet);
  signal(SIGINT, on_sig);
  signal(SIGTERM, on_sig);

  struct kv_alloc_bpf *skel = kv_alloc_bpf__open();
  if (!skel) { fprintf(stderr, "open failed\n"); return 1; }
  skel->rodata->target_pid = pid;
  skel->rodata->min_size = min_size;
  if (kv_alloc_bpf__load(skel)) {
    fprintf(stderr, "load failed (verifier?)\n");
    kv_alloc_bpf__destroy(skel);
    return 1;
  }

  LIBBPF_OPTS(bpf_uprobe_opts, en, .func_name = sym, .retprobe = false);
  LIBBPF_OPTS(bpf_uprobe_opts, ex, .func_name = sym, .retprobe = true);
  struct bpf_link *l1 = bpf_program__attach_uprobe_opts(skel->progs.alloc_enter, (int)pid, bin, 0, &en);
  struct bpf_link *l2 = bpf_program__attach_uprobe_opts(skel->progs.alloc_exit, (int)pid, bin, 0, &ex);
  if (!l1 || !l2) {
    fprintf(stderr, "attach %s:%s failed: %d\n", bin, sym, -errno);
    if (l1) bpf_link__destroy(l1);
    if (l2) bpf_link__destroy(l2);
    kv_alloc_bpf__destroy(skel);
    return 1;
  }
  fprintf(stderr, "eKV kv_alloc: pid=%u probe=%s:%s min_size=%llu (Ctrl-C to stop)\n",
          pid, bin, sym, min_size);

  struct ring_buffer *rbuf =
      ring_buffer__new(bpf_map__fd(skel->maps.rb), on_evt, NULL, NULL);
  if (!rbuf) {
    fprintf(stderr, "ringbuf new failed\n");
    bpf_link__destroy(l1); bpf_link__destroy(l2);
    kv_alloc_bpf__destroy(skel);
    return 1;
  }
  int rc = drain_and_report(rbuf, bpf_map__fd(skel->maps.drops), &exiting, 0);
  ring_buffer__free(rbuf);
  bpf_link__destroy(l1);
  bpf_link__destroy(l2);
  kv_alloc_bpf__destroy(skel);
  return rc;
}
