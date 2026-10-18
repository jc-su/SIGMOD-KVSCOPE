// Drives eKV's event-query completion path with REAL not-ready polls, to validate the
// ready-bracket: a large H2D copy (fetch) is still running when we start polling
// cudaEventQuery, so the first polls return cudaErrorNotReady and the last one returns
// cudaSuccess. eKV should then emit a completion with ready_lo_ns = the last not-ready poll
// and ready_window_ns = the poll gap (the residual uncertainty on the true GPU-ready instant).
//   nvcc -o /tmp/ekv_evtready tests/_event_ready.cu   (GPU1 via CUDA_VISIBLE_DEVICES=1)
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <unistd.h>

#define CHK(x) do { cudaError_t e_=(x); if (e_!=cudaSuccess){fprintf(stderr,"FAIL %s: %s\n",#x,cudaGetErrorString(e_));return 1;} } while(0)

int main() {
  const size_t SZ = 256ull * 1024 * 1024;            // 256 MiB: ~10ms H2D, so polls see not-ready
  CHK(cudaSetDevice(0));                              // device 0 == GPU1 under CUDA_VISIBLE_DEVICES=1; loads libcudart
  // Barrier: libcudart is now mapped. Announce the pid and pause so the runner can attach
  // kv_copy SCOPED TO THIS PID (no system-wide libcudart uprobe -> no GPU0 co-tenant overhead).
  printf("EKV_TEST_PID %d\n", (int)getpid());
  fflush(stdout);
  sleep(4);
  void *h = nullptr, *d = nullptr;
  CHK(cudaMallocHost(&h, SZ));                        // pinned host (async H2D requires it)
  memset(h, 0xCD, SZ);
  CHK(cudaMalloc(&d, SZ));
  cudaStream_t s; CHK(cudaStreamCreate(&s));
  cudaEvent_t ev; CHK(cudaEventCreate(&ev));

  for (int it = 0; it < 3; it++) {
    CHK(cudaMemcpyAsync(d, h, SZ, cudaMemcpyHostToDevice, s));   // H2D = fetch (DRAM -> GPU)
    CHK(cudaEventRecord(ev, s));                                 // KV_BIND: copies -> event
    int polls = 0, notready = 0;
    cudaError_t q;
    do {                                                         // poll until done
      q = cudaEventQuery(ev);
      polls++;
      if (q == cudaErrorNotReady) notready++;
      else if (q != cudaSuccess) { fprintf(stderr, "FAIL query: %s\n", cudaGetErrorString(q)); return 1; }
    } while (q == cudaErrorNotReady);
    printf("iter %d: polls=%d notready=%d\n", it, polls, notready);
  }
  CHK(cudaStreamSynchronize(s));
  cudaEventDestroy(ev); cudaStreamDestroy(s); cudaFree(d); cudaFreeHost(h);
  printf("event-ready workload done\n");
  return 0;
}
