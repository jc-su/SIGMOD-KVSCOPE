// Overhead microbench for the kv_copy eBPF uprobe: time a tight cudaMemcpyAsync
// loop, run once WITHOUT the collector (baseline) and once WITH it attached. The
// per-op delta is the uprobe + BPF-program cost charged to each enqueue.
//   nvcc -O2 tests/_memcpy_overhead.c -lcudart -o /tmp/ekv_memcpy_oh
//   /tmp/ekv_memcpy_oh [N]
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
  int N = argc > 1 ? atoi(argv[1]) : 200000;
  size_t sz = 4096;
  void *d = NULL, *h = NULL;
  cudaMalloc(&d, sz);
  cudaHostAlloc(&h, sz, 0);
  cudaStream_t s;
  cudaStreamCreate(&s);

  for (int i = 0; i < 2000; i++)  // warmup
    cudaMemcpyAsync(h, d, sz, cudaMemcpyDeviceToHost, s);
  cudaStreamSynchronize(s);

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  for (int i = 0; i < N; i++) {
    cudaMemcpyAsync(h, d, sz, cudaMemcpyDeviceToHost, s);
    if ((i & 1023) == 0)
      cudaStreamSynchronize(s);  // bound the in-flight queue, like real offload
  }
  cudaStreamSynchronize(s);
  clock_gettime(CLOCK_MONOTONIC, &t1);

  double ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
  printf("memcpy_async %d ops  %.1f ns/op  (%.0f ops/s)\n", N, ns / N, N / (ns / 1e9));
  return 0;
}
