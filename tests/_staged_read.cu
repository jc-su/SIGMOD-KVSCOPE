// Exercise the STAGED KV fetch path storage -> DRAM -> GPU, to validate that eKV observes BOTH
// legs and that they are CORRELATABLE: storage_read captures the read() (storage -> host DRAM) and
// kv_copy captures the following cudaMemcpyAsync H2D (DRAM -> GPU) on the SAME thread, so the
// analysis can join read(tid) -> H2D(tid) into the full tiered-miss cost (tier_read + H2D).
//   nvcc -o /tmp/ekv_staged tests/_staged_read.cu   (GPU1 via CUDA_VISIBLE_DEVICES=1)
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

#define CHK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){fprintf(stderr,"FAIL %s: %s\n",#x,cudaGetErrorString(e_));return 1;} } while(0)

int main() {
  const size_t SZ = 4ull * 1024 * 1024;            // 4 MiB > storage_read's 1 MiB KV-chunk threshold
  const char *path = "/tmp/ekv_staged_kv.bin";

  // a KV "tier file" with known content
  int fd = open(path, O_CREAT | O_RDWR, 0644);
  if (fd < 0) { perror("open"); return 1; }
  char *tmp = (char *)malloc(SZ); memset(tmp, 0xAB, SZ);
  if (write(fd, tmp, SZ) != (ssize_t)SZ) { perror("write"); return 1; }
  fsync(fd); free(tmp);

  CHK(cudaSetDevice(0));                            // device 0 == GPU1 under CUDA_VISIBLE_DEVICES=1; loads libcudart
  // barrier: announce pid + pause so the runner can attach storage_read + kv_copy SCOPED to this pid
  printf("EKV_TEST_PID %d\n", (int)getpid());
  fflush(stdout);
  sleep(4);

  // leg 1: storage -> DRAM. read() into PINNED host memory (the syscall fires regardless of buffer type).
  void *hbuf = nullptr;
  CHK(cudaMallocHost(&hbuf, SZ));
  lseek(fd, 0, SEEK_SET);
  ssize_t nr = read(fd, hbuf, SZ);                 // <-- storage_read captures this (bytes, dur, tid)
  printf("read %zd bytes (storage -> DRAM)\n", nr);

  // leg 2: DRAM -> GPU. the H2D the analysis joins to the read above (same thread).
  void *dbuf = nullptr;
  CHK(cudaMalloc(&dbuf, SZ));
  cudaStream_t s; CHK(cudaStreamCreate(&s));
  CHK(cudaMemcpyAsync(dbuf, hbuf, SZ, cudaMemcpyHostToDevice, s));   // <-- kv_copy captures this (H2D)
  CHK(cudaStreamSynchronize(s));
  printf("H2D %zu bytes (DRAM -> GPU) done\n", SZ);

  cudaStreamDestroy(s); cudaFree(dbuf); cudaFreeHost(hbuf); close(fd);
  printf("staged workload done\n");
  return (nr == (ssize_t)SZ) ? 0 : 2;
}
