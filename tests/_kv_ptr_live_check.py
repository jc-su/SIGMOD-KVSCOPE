"""LIVE composition on the REAL vLLM kernel_unified_attention (no eBPF/sudo needed):
the pass finds key_cache_ptr (gap 4), captures its runtime value, publishes it; we
verify the published KV pointer lands inside the engine's real KV-cache allocation, and
that identify_kv_pool(kv_ptr) + BlockAddressMap.from_alloc built from it resolve real KV
addresses — the transfer face's addr->block, composed on a live engine with no
in-process tensor read in the data path. Oracle = the engine's kv_caches ranges (read
in-process, validation only).

  TRITON_EXT_ENABLED=1 TRITON_PLUGIN_PATHS=.../libeKVPlugin.so EKV_PASS_AUTO=1 \
  PYTHONPATH=triton-src LD_PRELOAD=libstdc++ CUDA_VISIBLE_DEVICES=1 python this.py
"""
import os
import sys

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
os.environ["EKV_PASS_AUTO"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from observe import BlockAddressMap, identify_kv_pool, zero_touch  # noqa: E402


from ekv_vllm_plugin import install_pass  # noqa: E402

install_pass()

_kv = {"ranges": []}


def _install_oracle():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GR
    orig = GR.execute_model

    def execute_model(runner, *a, **k):
        out = orig(runner, *a, **k)
        try:
            if not _kv["ranges"]:
                rng = []
                for t in (getattr(runner, "kv_caches", None) or []):
                    try:
                        rng.append((int(t.data_ptr()), int(t.numel() * t.element_size())))
                    except Exception:  # noqa: BLE001
                        pass
                if rng:
                    _kv["ranges"] = rng
        except Exception:  # noqa: BLE001
            pass
        return out
    GR.execute_model = execute_model


def main():
    _install_oracle()
    from vllm import LLM, SamplingParams
    nblocks = int(os.environ.get("EKV_GPU_BLOCKS", "2048"))
    llm = LLM(model="facebook/opt-125m", enforce_eager=True, attention_backend="TRITON_ATTN",
              gpu_memory_utilization=0.30, max_model_len=2048, enable_prefix_caching=True,
              enable_chunked_prefill=False, max_num_seqs=64, num_gpu_blocks_override=nblocks)
    llm.generate(["Zebra quantum waterfall marble engine drifting"] * 4,
                 SamplingParams(max_tokens=16, temperature=0.0, ignore_eos=True))

    kv_ptr = zero_touch.read_pass_kv_ptr(os.getpid())
    ranges = _kv["ranges"]
    print(f"pass kv_ptr = {hex(kv_ptr) if kv_ptr else None};  engine kv_caches ranges = {len(ranges)}")
    assert kv_ptr, "FAIL: pass published no KV pointer (key_cache_ptr not found on the live kernel?)"
    assert ranges, "FAIL: oracle captured no kv_caches ranges (harness issue, not the pass)"

    # (4) live: the captured pointer lies inside a REAL KV-cache allocation
    inside = next(((b, s) for (b, s) in ranges if b <= kv_ptr < b + s), None)
    print(f"kv_ptr inside a real kv_caches alloc: {inside is not None}")
    assert inside, f"kv_ptr {hex(kv_ptr)} not in any kv_caches range {[(hex(b), s) for b, s in ranges]}"

    # pool-id live: among the real allocs + a deliberately-larger fake 'weights' alloc,
    # identify_kv_pool picks the KV one by the pointer — not by size.
    allocs = [dict(addr=0x10_000000000, size=8 << 30)] + [dict(addr=b, size=s) for (b, s) in ranges]
    pool = identify_kv_pool(allocs, kv_ptr=kv_ptr)
    assert pool and pool[0] <= kv_ptr < pool[0] + pool[1], pool
    print(f"identify_kv_pool(kv_ptr) -> real KV alloc base={hex(pool[0])} size={pool[1] >> 20}MB "
          f"(not the 8 GB 'weights')")

    # from_alloc on the real base resolves addr->block live (synthetic stride from the pool)
    base, size = pool
    stride = max(1, size // nblocks)
    m = BlockAddressMap.from_alloc(base, nblocks, stride)
    blk = m.resolve(base + 3 * stride + 7)
    assert blk is not None and blk.block_id == 3, blk
    print(f"from_alloc(real KV base): addr->block resolves (block 3), stride={stride}B")
    print("PASS: live on kernel_unified_attention — pass captured key_cache_ptr, it lies in the "
          "engine's real KV alloc, identify_kv_pool pins it (not by size), from_alloc resolves "
          "addr->block. Transfer-face plumbing composed live; no in-process tensor read in the path.")


if __name__ == "__main__":
    main()
