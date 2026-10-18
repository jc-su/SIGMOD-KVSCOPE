// Attach-verify workload for the provider post_send uprobe: create an RC QP, take
// it to INIT (local-only), and post a few send WRs. Without a peer the posts return
// an error, but the provider's post_send IS invoked on entry, so the eKV uprobe
// fires — proving the runtime self-discovered offset is correct. (No NIC link
// needed; this is capture-mechanism validation, not a real transfer.)
//   cc tests/_provider_wr_workload.c -libverbs -o /tmp/ekv_provider_wr && /tmp/ekv_provider_wr
#include <infiniband/verbs.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main(void) {
  int n = 0;
  struct ibv_device **list = ibv_get_device_list(&n);
  if (!list || n == 0) {
    fprintf(stderr, "no RDMA device\n");
    return 1;
  }
  struct ibv_context *ctx = ibv_open_device(list[0]);
  struct ibv_pd *pd = ibv_alloc_pd(ctx);
  struct ibv_cq *cq = ibv_create_cq(ctx, 16, NULL, NULL, 0);
  char buf[4096];
  struct ibv_mr *mr = ibv_reg_mr(pd, buf, sizeof(buf), IBV_ACCESS_LOCAL_WRITE);
  struct ibv_qp_init_attr qa = {
      .send_cq = cq, .recv_cq = cq, .qp_type = IBV_QPT_RC,
      .cap = {.max_send_wr = 16, .max_recv_wr = 16,
              .max_send_sge = 1, .max_recv_sge = 1}};
  struct ibv_qp *qp = ibv_create_qp(pd, &qa);
  if (!qp) {
    fprintf(stderr, "qp create failed\n");
    return 1;
  }
  struct ibv_qp_attr a;
  memset(&a, 0, sizeof(a));
  a.qp_state = IBV_QPS_INIT;
  a.pkey_index = 0;
  a.port_num = 1;
  a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
  ibv_modify_qp(qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);

  // print the two SGE addresses so the captured post_send.addr can be checked.
  unsigned long a0 = (unsigned long)buf, a1 = (unsigned long)(buf + 2048);
  fprintf(stderr, "SGE addrs: wr0=%lu wr1=%lu (each 64 bytes)\n", a0, a1);

  for (int i = 0; i < 5; i++) {
    // a 2-WR LINKED LIST (wr0 -> wr1) per call: exercises wr->next walking and
    // distinct per-WR SGE addresses (the join key).
    struct ibv_sge sge0 = {.addr = a0, .length = 64, .lkey = mr ? mr->lkey : 0};
    struct ibv_sge sge1 = {.addr = a1, .length = 64, .lkey = mr ? mr->lkey : 0};
    struct ibv_send_wr wr0, wr1, *bad = NULL;
    memset(&wr0, 0, sizeof(wr0));
    memset(&wr1, 0, sizeof(wr1));
    wr1.wr_id = 0x2000 + i;
    wr1.sg_list = &sge1;
    wr1.num_sge = 1;
    wr1.opcode = IBV_WR_SEND;
    wr0.wr_id = 0x1000 + i;
    wr0.sg_list = &sge0;
    wr0.num_sge = 1;
    wr0.opcode = IBV_WR_SEND;
    wr0.next = &wr1;
    int rc = ibv_post_send(qp, &wr0, &bad);  // provider post_send invoked -> uprobe fires
    fprintf(stderr, "post_send[%d] 2-WR list wr_ids=0x%lx,0x%lx rc=%d\n",
            i, (unsigned long)wr0.wr_id, (unsigned long)wr1.wr_id, rc);
    usleep(100000);
  }
  return 0;
}
