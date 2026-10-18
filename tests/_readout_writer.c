// Test stand-in for the GPU pass: allocate a capture buffer + a timer buffer, fill
// them with a KNOWN pattern, publish /tmp/ekv_pass_<pid>.json exactly as the pass
// does, then call ekv_readout_tick() in a loop so the eBPF reader's uprobe fires in
// this process's context. No GPU/CUDA — the bpf_probe_read_user the reader performs
// is identical whether the buffer is host-pinned-mapped (pass) or plain mmap (here),
// so this faithfully validates the readout mechanism. Active rows: s<ACTIVE; row s =
// [s*100 + j]; cycles[s] = (s+1)*1e6. Unused rows are the 0xff/-1 UNWRITTEN sentinel.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#define NCAP 128
#define MAXROWS 8
#define ACTIVE 4

__attribute__((noinline)) void ekv_readout_tick(void) { asm volatile("" ::: "memory"); }

int main(void) {
  int cap_n = MAXROWS * NCAP;
  int *buf = mmap(NULL, (size_t)cap_n * 4, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  long long *tbuf = mmap(NULL, (size_t)MAXROWS * 8, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (buf == MAP_FAILED || tbuf == MAP_FAILED) return 1;
  for (int s = 0; s < MAXROWS; s++) {
    for (int j = 0; j < NCAP; j++) buf[s * NCAP + j] = (s < ACTIVE) ? (s * 100 + j) : -1;
    tbuf[s] = (s < ACTIVE) ? (long long)(s + 1) * 1000000 : 0;
  }
  int pid = (int)getpid();
  char path[64];
  snprintf(path, sizeof path, "/tmp/ekv_pass_%d.json", pid);
  FILE *f = fopen(path, "w");
  if (!f) return 1;
  fprintf(f,
          "{\"pid\":%d,\"host_va\":%llu,\"dev_ptr\":0,\"bytes\":%d,\"ncap\":%d,"
          "\"max_rows\":%d,\"timer_host_va\":%llu,\"timer_bytes\":%d}\n",
          pid, (unsigned long long)(uintptr_t)buf, cap_n * 4, NCAP, MAXROWS,
          (unsigned long long)(uintptr_t)tbuf, MAXROWS * 8);
  fclose(f);
  fprintf(stderr, "writer pid=%d ready\n", pid);
  for (;;) { ekv_readout_tick(); usleep(2000); }
  return 0;
}
