"""Independent counter validation for KVScope's summed-CTA-residency signal.

One profiled attention-like launch exposes both the compiler-pass timer and Nsight Compute DRAM
bytes.  Environment selects request composition and physical locality; the evaluation runner
executes several cells and compares the two measurements with planted logical KV reads.
"""
import json
import os

import torch
import triton
import triton.language as tl


from ekv_vllm_plugin import install_pass  # noqa: E402

install_pass()


@triton.jit
def metric_attention(block_table, seq_nblks, kv_cache, query, output,
                     QB: tl.constexpr, W: tl.constexpr, D: tl.constexpr,
                     NPHYS: tl.constexpr, REPS: tl.constexpr):
    program = tl.program_id(0)
    head = tl.program_id(1)
    request = program // QB
    positions = tl.arange(0, W)
    physical = tl.load(block_table + request * W + positions)
    count = tl.load(seq_nblks + request)
    dims = tl.arange(0, D)
    q = tl.load(query + request * D + dims)
    acc = 0.0
    for rep in range(0, REPS):
        for pos in range(0, count):
            block = tl.sum(tl.where(positions == pos, physical, 0)) + rep * NPHYS
            k = tl.load(kv_cache + block * D + dims)
            acc += tl.sum(q * k)
    tl.store(output + program * tl.num_programs(1) + head, acc)


def main():
    requests, query_blocks, heads = 8, 2, 2
    width, dims, repeats, nphys = 64, 128, 16, 65536
    profile = os.environ.get("EKV_METRIC_PROFILE", "skew")
    locality = os.environ.get("EKV_METRIC_LOCALITY", "disjoint")
    os.environ.setdefault("EKV_NCAP", str(width))
    os.environ.setdefault("EKV_MAX_ROWS", "128")
    profiles = {
        "short": [4] * requests,
        "balanced": [24] * requests,
        "skew": [4, 8, 12, 16, 24, 32, 48, 64],
        "long": [56] * requests,
    }
    counts_host = profiles[profile]
    torch.manual_seed(183)
    if locality == "shared":
        common = torch.randperm(nphys - 1, device="cuda", dtype=torch.int64)[:width] + 1
        table = common.to(torch.int32).repeat(requests, 1)
    else:
        table = torch.randint(1, nphys, (requests, width), dtype=torch.int32, device="cuda")
    counts = torch.tensor(counts_host, dtype=torch.int32, device="cuda")
    kv = torch.randn(repeats * nphys * dims, dtype=torch.float16, device="cuda")
    query = torch.randn(requests * dims, dtype=torch.float16, device="cuda")
    output = torch.zeros(requests * query_blocks * heads, dtype=torch.float32, device="cuda")
    grid = (requests * query_blocks, heads)

    from observe.zero_touch import read_pass_cycles
    metric_attention[grid](table, counts, kv, query, output, QB=query_blocks, W=width, D=dims,
                           NPHYS=nphys, REPS=repeats)
    torch.cuda.synchronize()
    before = read_pass_cycles(os.getpid())
    torch.cuda.cudart().cudaProfilerStart()
    metric_attention[grid](table, counts, kv, query, output, QB=query_blocks, W=width, D=dims,
                           NPHYS=nphys, REPS=repeats)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    after = read_pass_cycles(os.getpid())
    cycles = {str(k): int(after.get(k, 0) - before.get(k, 0)) for k in after}
    logical_bytes = sum(counts_host) * query_blocks * heads * repeats * dims * 2
    print("EKV_METRIC " + json.dumps({
        "profile": profile, "locality": locality, "counts": counts_host,
        "logical_kv_bytes": logical_bytes, "request_cycles": cycles,
        "summed_cta_cycles": sum(cycles.values()),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
