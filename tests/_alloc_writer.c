// Test stand-in for cudaMalloc: a function with the same (void** devPtr, size_t)
// shape that writes a KNOWN device pointer to *devPtr, called in a loop so the
// kv_alloc uprobe+uretprobe fires. No CUDA — the arg-stash-at-entry +
// bpf_probe_read_user(*devPtr)-at-return capture is identical to the real cudaMalloc.
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#define KNOWN_ADDR 0x0000700000000000ULL
#define KNOWN_SIZE 268435456ULL /* 256 MB — pool-scale */

__attribute__((noinline)) int fake_alloc(void **p, unsigned long size) {
  (void)size;
  *p = (void *)(uintptr_t)KNOWN_ADDR;
  asm volatile("" ::: "memory");
  return 0;
}

int main(void) {
  fprintf(stderr, "alloc-writer pid=%d ready\n", (int)getpid());
  for (;;) {
    void *p = 0;
    fake_alloc(&p, KNOWN_SIZE);
    usleep(5000);
  }
  return 0;
}
