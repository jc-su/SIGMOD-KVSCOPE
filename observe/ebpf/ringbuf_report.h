/* Userspace poll loop shared by the single-ring-buffer collectors (kv_alloc, storage_read,
 * ekv_readout, http_label). Each BPF program counts its failed ring buffer reservations in a one-entry
 * array map named `drops`; this loop drains the ring buffer and prints that count at exit as
 * "ringbuf drops: N" (observe.health.parse_ringbuf_drops reads it), so a lost record is never
 * silent. */
#ifndef EKV_RINGBUF_REPORT_H
#define EKV_RINGBUF_REPORT_H

#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <errno.h>
#include <stdio.h>
#include <time.h>

/* Poll `rb` until *exiting is set or, if dur_s > 0, until dur_s seconds have passed. Then drain
 * the records already submitted and print the drop count. Returns 0, or 1 if a poll, the final
 * drain, or the counter read failed. */
static int drain_and_report(struct ring_buffer *rb, int drops_fd, volatile int *exiting, int dur_s) {
  int rc = 0;
  struct timespec t0;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  while (!*exiting) {
    int n = ring_buffer__poll(rb, 200);
    if (n == -EINTR)
      continue;
    if (n < 0) {
      fprintf(stderr, "ring buffer poll failed: %d\n", n);
      rc = 1;
      break;
    }
    if (dur_s) {
      struct timespec now;
      clock_gettime(CLOCK_MONOTONIC, &now);
      if (now.tv_sec - t0.tv_sec >= dur_s)
        break;
    }
  }
  if (rc == 0 && ring_buffer__consume(rb) < 0) {
    fprintf(stderr, "final ring buffer drain failed\n");
    rc = 1;
  }
  fflush(stdout);
  __u32 k = 0;
  __u64 drops = 0;
  if (bpf_map_lookup_elem(drops_fd, &k, &drops)) {
    fprintf(stderr, "cannot read the ring buffer drop counter: errno %d\n", errno);
    return 1;
  }
  fprintf(stderr, "ringbuf drops: %llu\n", (unsigned long long)drops);
  return rc;
}

#endif /* EKV_RINGBUF_REPORT_H */
