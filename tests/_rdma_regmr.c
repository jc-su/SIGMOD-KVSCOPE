// Minimal RDMA driver for the eBPF control-plane test: open the first HCA and
// register an 8 MiB memory region 3 times. ibv_reg_mr is a LOCAL op (no link
// needed), so this works even with the NIC link DOWN. cc -o _rdma_regmr
// _rdma_regmr.c -libverbs
#include <infiniband/verbs.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
  int n = 0;
  struct ibv_device **list = ibv_get_device_list(&n);
  if (!list || n == 0) {
    fprintf(stderr, "no IB/RDMA devices\n");
    return 1;
  }
  struct ibv_context *ctx = ibv_open_device(list[0]);
  if (!ctx) {
    fprintf(stderr, "ibv_open_device failed\n");
    return 1;
  }
  struct ibv_pd *pd = ibv_alloc_pd(ctx);
  if (!pd) {
    fprintf(stderr, "ibv_alloc_pd failed\n");
    return 1;
  }
  size_t len = 8u << 20; // 8 MiB
  void *buf = malloc(len);
  memset(buf, 0, len);
  for (int i = 0; i < 3; i++) {
    struct ibv_mr *mr = ibv_reg_mr(pd, buf, len,
                                   IBV_ACCESS_LOCAL_WRITE |
                                       IBV_ACCESS_REMOTE_WRITE |
                                       IBV_ACCESS_REMOTE_READ);
    if (!mr) {
      fprintf(stderr, "ibv_reg_mr failed\n");
      return 1;
    }
    printf("registered MR %d on %s: addr=%p length=%zu\n", i,
           ibv_get_device_name(list[0]), buf, len);
    ibv_dereg_mr(mr);
  }
  usleep(1000000); // let the collector flush
  return 0;
}
