"""Validation of the TMA / descriptor op-set (the op-set lesson): the eKV index_lookup walk now
recognizes Hopper TMA gathers (`tt.descriptor_load` / `tt.descriptor_gather`) as KV value gathers, in the
SAME one `K(addr)` principle as an addptr load. The KV pool is reached through `make_tensor_descriptor`'s
base (the existing region-aware walk already threads operands), and the index reaches via the descriptor
coords / x_offsets.

The box is an A6000 (sm_86, NO Hopper) -- Triton lowers TMA away on it, so a compiled kernel yields plain
loads, not descriptor ops. So this validates on RAW TTIR (the method an earlier plugin uses for its own synthetic-only
TMA): a hand-written module with a real `tt.descriptor_gather` KV gather (x_offsets = the loaded block ids
= the index map; the descriptor wraps kv_cache; the gathered KV feeds a tt.reduce contraction). The pass
runs in DETECT-ONLY mode (no CUDA alloc), and PASS = it identifies block_table (arg 0) as the index map,
kv_cache (arg 1) as the KV base, via the descriptor gather.

  TRITON_EXT_ENABLED=1 TRITON_PLUGIN_PATHS=.../libeKVPlugin.so EKV_ENABLE=1 EKV_PASS_AUTO=1 \
  PYTHONPATH=triton-src LD_PRELOAD=libstdc++ python this.py   (no GPU needed)
"""
import os
import tempfile

TTIR = """
module {
  tt.func public @attn_tma(%block_table: !tt.ptr<i32>, %kv_cache: !tt.ptr<f16>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i32 = arith.constant 32 : i32
    %c32_i64 = arith.constant 32 : i64
    %c2048_i32 = arith.constant 2048 : i32
    %nb = arith.constant 16 : i32
    %pid = tt.get_program_id x : i32
    %rowoff = arith.muli %pid, %nb : i32
    %btrow = tt.addptr %block_table, %rowoff : !tt.ptr<i32>, i32
    %rng = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %btsp = tt.splat %btrow : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %btptr = tt.addptr %btsp, %rng : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %phys = tt.load %btptr : tensor<16x!tt.ptr<i32>>
    %desc = tt.make_tensor_descriptor %kv_cache, [%c2048_i32, %c32_i32], [%c32_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<1x32xf16>
    %k = tt.descriptor_gather %desc[%phys, %c0_i32] : (!tt.tensordesc<1x32xf16>, tensor<16xi32>, i32) -> tensor<16x32xf16>
    %qk = "tt.reduce"(%k) <{axis = 1 : i32}> ({
    ^bb0(%a: f16, %b: f16):
      %s = arith.addf %a, %b : f16
      tt.reduce.return %s : f16
    }) : (tensor<16x32xf16>) -> tensor<16xf16>
    tt.return
  }
}
"""


def main():
    os.environ["EKV_ENABLE"] = "1"
    os.environ["EKV_PASS_AUTO"] = "1"
    os.environ["EKV_DETECT_ONLY"] = "1"

    tf = tempfile.NamedTemporaryFile("w", suffix=".ttir", delete=False)
    tf.write(TTIR)
    tf.close()

    # capture the C-level stderr the pass prints to
    errf = tempfile.NamedTemporaryFile(delete=False)
    saved = os.dup(2)
    os.dup2(errf.fileno(), 2)
    try:
        from triton._C.libtriton import ir, passes
        ctx = ir.context()
        ir.load_dialects(ctx)
        mod = ir.parse_mlir_module(tf.name, ctx)
        mod.context = ctx
        pm = ir.pass_manager(ctx)
        __import__('runtime.ekv_plugin', fromlist=['_resolve_ekv_pass'])._resolve_ekv_pass(passes)(pm)
        pm.run(mod, "tma")
    finally:
        os.dup2(saved, 2)
        errf.flush()
    err = open(errf.name).read()

    line = next((ln for ln in err.splitlines() if "eKV-detect" in ln), "")
    print("detect:", line or "(none)")
    ok = ("kernel=attn_tma" in line and "idxArg=0" in line and "kvArg=1" in line
          and "gather=tt.descriptor_gather" in line)
    print("PASS: eKV detected the TMA descriptor_gather KV gather structurally (index=block_table, "
          "base=kv_cache via the descriptor) -- the same one K(addr) principle, descriptor op-set."
          if ok else "FAIL: TMA descriptor gather not detected as the index_lookup")


if __name__ == "__main__":
    main()
