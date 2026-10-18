// Userspace loader for the eKV storage-tier read probe. Attaches syscall tracepoints
// (read/pread64), PID- and size-filtered, and prints one JSONL record per qualifying
// read: {t_ns, pid, tid, fd, bytes, dur_ns}. The analysis joins each read to the
// following H2D on the same thread -> full tier miss cost = tier_read + H2D.
//
//   sudo ./storage_read <pid> [-m min_bytes]
#include <errno.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "storage_read.h"
#include "ringbuf_report.h"
#include "storage_read.skel.h"
#include <bpf/libbpf.h>

static volatile int exiting;
static void on_sig(int s) { (void)s; exiting = 1; }
static int quiet(enum libbpf_print_level l, const char *f, va_list a) {
  if (l == LIBBPF_DEBUG) return 0;
  return vfprintf(stderr, f, a);
}

static int on_evt(void *ctx, void *data, size_t sz) {
  (void)ctx; (void)sz;
  const struct read_rec *r = data;
  printf("{\"t_ns\":%llu,\"pid\":%u,\"tid\":%u,\"fd\":%llu,\"bytes\":%llu,\"dur_ns\":%llu}\n",
         r->ts_ns, r->pid, r->tid, r->fd, r->bytes, r->dur_ns);
  fflush(stdout);
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <pid> [-m min_bytes]\n", argv[0]);
    return 2;
  }
  unsigned int pid = (unsigned int)atoi(argv[1]);
  unsigned long long min_size = 1u << 20;   // default 1 MiB: KV chunks, not metadata
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "-m") && i + 1 < argc) min_size = strtoull(argv[++i], NULL, 10);
  }

  libbpf_set_print(quiet);
  signal(SIGINT, on_sig);
  signal(SIGTERM, on_sig);

  struct storage_read_bpf *skel = storage_read_bpf__open();
  if (!skel) { fprintf(stderr, "open failed\n"); return 1; }
  skel->rodata->target_pid = pid;
  skel->rodata->min_size = min_size;
  if (storage_read_bpf__load(skel)) {
    fprintf(stderr, "load failed (verifier?)\n");
    storage_read_bpf__destroy(skel);
    return 1;
  }
  if (storage_read_bpf__attach(skel)) {   // auto-attach the syscall tracepoints
    fprintf(stderr, "attach failed\n");
    storage_read_bpf__destroy(skel);
    return 1;
  }
  fprintf(stderr, "eKV storage_read: pid=%u min_size=%llu (Ctrl-C to stop)\n", pid, min_size);

  struct ring_buffer *rbuf =
      ring_buffer__new(bpf_map__fd(skel->maps.rb), on_evt, NULL, NULL);
  if (!rbuf) {
    fprintf(stderr, "ringbuf new failed\n");
    storage_read_bpf__destroy(skel);
    return 1;
  }
  int rc = drain_and_report(rbuf, bpf_map__fd(skel->maps.drops), &exiting, 0);
  ring_buffer__free(rbuf);
  storage_read_bpf__destroy(skel);
  return rc;
}
