/* Shared record for the eKV KV-pool allocation probe (kv_alloc).
 *
 * Captures device allocations (cudaMalloc / cuMemAlloc) so the KV pool's base
 * address + size can be recovered from the data plane — feeding
 * BlockAddressMap.from_alloc(base, num_blocks, stride) so address->block resolution
 * needs NO in-process KV-tensor read (the engine-agnostic, zero-touch transfer face).
 * The KV pool is the large allocation; the loader/analysis picks it (largest, or
 * size matching the model's KV bytes). */
#ifndef EKV_KV_ALLOC_H
#define EKV_KV_ALLOC_H

struct alloc_rec {
  unsigned long long ts_ns;
  unsigned int pid;
  unsigned int tid;
  unsigned long long addr;   /* the allocated device pointer (*devPtr at return) */
  unsigned long long size;   /* bytes requested */
};

#endif /* EKV_KV_ALLOC_H */
