// Userspace loader for the eKV zero-touch readout. Reads the pass's published
// descriptor (/tmp/ekv_pass_<pid>.json), attaches a uprobe to a function the engine
// calls each step, lifts the capture (+timer) buffer via bpf_probe_read_user, and
// prints {slots, cycles} — the SAME output as `python observe/zero_touch.py <pid>`,
// but via eBPF instead of /proc.
//
//   sudo ./ekv_readout <engine_pid> [binary:symbol] [-d seconds]
//
// Default probe is libcuda.so.1:cuLaunchKernel (the driver launch every Triton
// kernel hits). For a non-CUDA target pass an explicit binary:symbol.
#include <errno.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "ekv_readout.h"
#include "ringbuf_report.h"
#include "ekv_readout.skel.h"
#include <bpf/libbpf.h>

static volatile int exiting;
static void on_sig(int s) { (void)s; exiting = 1; }
static int quiet(enum libbpf_print_level l, const char *f, va_list a) {
  if (l == LIBBPF_DEBUG) return 0;
  return vfprintf(stderr, f, a);
}

/* latest snapshot of each buffer (the pass overwrites a fixed buffer each step) 
 * NOTE (2026-07-02): the TIMER half of this tool is RETIRED -- the pass timer moved to DEVICE
 * memory (host-mapped per-CTA atomics were a ~24% PCIe storm; commit 78ad5c9), and eBPF cannot
 * read device memory. Timer and attention-statistic readout now lives in observe/daemon.py over the pass's exported
 * CUDA-IPC handles. The CAPTURE half here (host-mapped block-table buffer) remains valid, as do
 * eBPF's movement collectors and the sync-anchor timing role.
 */
static unsigned char cap_buf[EKV_BUF_CAP]; static unsigned int cap_len;
static unsigned char tim_buf[EKV_BUF_CAP]; static unsigned int tim_len;
static int g_ncap, g_max_rows;

/* minimal flat-JSON integer field reader for the published descriptor */
static long long jget(const char *s, const char *key) {
  char pat[64];
  snprintf(pat, sizeof pat, "\"%s\"", key);
  const char *p = strstr(s, pat);
  if (!p) return 0;
  p = strchr(p, ':');
  if (!p) return 0;
  return strtoll(p + 1, NULL, 10);
}

static int g_have_timer = 0;      /* set from timer_bytes: a tick = cap record then timer record */
static long g_nprinted = 0;
static void print_snapshot(void);

static int on_evt(void *ctx, void *data, size_t sz) {
  (void)ctx; (void)sz;
  const struct ekv_buf_rec *r = data;
  unsigned int n = r->len > EKV_BUF_CAP ? EKV_BUF_CAP : r->len;
  if (r->kind == 0) {
    memcpy(cap_buf, r->data, n); cap_len = n;
    if (!g_have_timer && cap_len) { print_snapshot(); g_nprinted++; }   /* no-timer: cap is the tick */
  } else {
    memcpy(tim_buf, r->data, n); tim_len = n;
    if (cap_len) { print_snapshot(); g_nprinted++; }   /* stream one snapshot per completed tick */
  }
  return 0;
}

/* parse + print exactly like observe/zero_touch.py (parse_slots + read_pass_cycles):
 * capture is i32[max_rows*ncap], active row = row[0] != -1 (the 0xff UNWRITTEN
 * sentinel the pass memsets); timer is i64[max_rows], active = cycles > 0. */
static void print_snapshot(void) {
  const int *a = (const int *)cap_buf;
  int n = (int)(cap_len / 4);
  int first = 1;
  printf("{\"slots\":{");
  for (int s = 0; s < g_max_rows && (s + 1) * g_ncap <= n; s++) {
    const int *row = a + (size_t)s * g_ncap;
    if (row[0] == -1) continue;
    printf(first ? "\"%d\":[" : ",\"%d\":[", s);
    first = 0;
    for (int j = 0; j < g_ncap; j++) printf(j ? ",%d" : "%d", row[j]);
    printf("]");
  }
  printf("},\"cycles\":{");
  const long long *t = (const long long *)tim_buf;
  int tn = (int)(tim_len / 8);
  first = 1;
  for (int s = 0; s < g_max_rows && s < tn; s++)
    if (t[s] > 0) { printf(first ? "\"%d\":%lld" : ",\"%d\":%lld", s, t[s]); first = 0; }
  printf("}}\n");
  fflush(stdout);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <pid> [binary:symbol] [-d seconds]\n", argv[0]);
    return 2;
  }
  unsigned int pid = (unsigned int)atoi(argv[1]);
  const char *probe = "libcuda.so.1:cuLaunchKernel";
  int dur = 0;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "-d") && i + 1 < argc) dur = atoi(argv[++i]);
    else probe = argv[i];
  }
  const char *colon = strrchr(probe, ':');
  if (!colon) { fprintf(stderr, "probe must be binary:symbol\n"); return 2; }
  char bin[512];
  size_t bl = (size_t)(colon - probe);
  if (bl >= sizeof bin) bl = sizeof bin - 1;
  memcpy(bin, probe, bl); bin[bl] = 0;
  const char *sym = colon + 1;

  char path[128];
  const char *rdir = getenv("EKV_RUNTIME_DIR");
  snprintf(path, sizeof path, "%s/ekv_pass_%u.json", rdir && *rdir ? rdir : "/tmp", pid);
  FILE *jf = fopen(path, "r");
  if (!jf) { fprintf(stderr, "no %s (pass running with EKV_PASS_AUTO?)\n", path); return 1; }
  char js[8192];
  size_t jn = fread(js, 1, sizeof js - 1, jf);
  js[jn] = 0;
  fclose(jf);
  /* The pass publishes one JSON line per instrumented kernel variant (per-kernel buffers). Pick the
   * LAST capture line that carries a timer -- the steady-decode attention variant is compiled last,
   * so its buffer is the one that updates each step. (A single-variant kernel has just one line.) */
  char *src = js, *lastcap = NULL;
  for (char *ln = strtok(js, "\n"); ln; ln = strtok(NULL, "\n"))
    if (jget(ln, "host_va") > 0 && jget(ln, "timer_host_va") > 0)
      lastcap = ln;
  if (lastcap)
    src = lastcap;
  unsigned long long host_va = (unsigned long long)jget(src, "host_va");
  unsigned int bytes = (unsigned int)jget(src, "bytes");
  unsigned long long tv = (unsigned long long)jget(src, "timer_host_va");
  unsigned int tb = (unsigned int)jget(src, "timer_bytes");
  g_have_timer = (tb > 0);
  g_ncap = (int)jget(src, "ncap");
  g_max_rows = (int)jget(src, "max_rows");
  if (!host_va || !bytes || g_ncap <= 0 || g_max_rows <= 0) {
    fprintf(stderr, "bad descriptor in %s\n", path); return 1;
  }
  if (bytes >= EKV_BUF_CAP) {
    fprintf(stderr, "buffer %u >= EKV_BUF_CAP %u; rebuild with a larger cap\n", bytes, EKV_BUF_CAP);
    return 1;
  }

  libbpf_set_print(quiet);
  signal(SIGINT, on_sig);
  signal(SIGTERM, on_sig);

  struct ekv_readout_bpf *skel = ekv_readout_bpf__open();
  if (!skel) { fprintf(stderr, "open failed\n"); return 1; }
  skel->rodata->target_pid = pid;
  skel->rodata->cap_va = host_va;
  skel->rodata->cap_bytes = bytes;
  skel->rodata->timer_va = tv;
  skel->rodata->timer_bytes = tb;
  const char *iv = getenv("EKV_READOUT_INTERVAL_NS");   /* throttle between snapshots (default 50ms) */
  if (iv)
    skel->rodata->interval_ns = strtoull(iv, NULL, 10);
  if (ekv_readout_bpf__load(skel)) {
    fprintf(stderr, "load failed (verifier?)\n");
    ekv_readout_bpf__destroy(skel);
    return 1;
  }

  LIBBPF_OPTS(bpf_uprobe_opts, uo, .func_name = sym, .retprobe = false);
  struct bpf_link *link =
      bpf_program__attach_uprobe_opts(skel->progs.ekv_tick, (int)pid, bin, 0, &uo);
  if (!link) {
    fprintf(stderr, "attach uprobe %s:%s failed: %d\n", bin, sym, -errno);
    ekv_readout_bpf__destroy(skel);
    return 1;
  }

  struct ring_buffer *rbuf =
      ring_buffer__new(bpf_map__fd(skel->maps.rb), on_evt, NULL, NULL);
  if (!rbuf) {
    fprintf(stderr, "ringbuf new failed\n");
    bpf_link__destroy(link);
    ekv_readout_bpf__destroy(skel);
    return 1;
  }
  fprintf(stderr, "eKV readout: pid=%u probe=%s:%s cap=%uB timer=%uB\n",
          pid, bin, sym, bytes, tb);

  int rc = drain_and_report(rbuf, bpf_map__fd(skel->maps.drops), &exiting, dur);
  if (!g_nprinted)
    fprintf(stderr, "no snapshot (did the engine call %s during the window?)\n", sym);

  ring_buffer__free(rbuf);
  bpf_link__destroy(link);
  ekv_readout_bpf__destroy(skel);
  return rc ? rc : (cap_len ? 0 : 1);
}
