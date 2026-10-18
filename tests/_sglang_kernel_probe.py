"""BRING-UP PROBE (research, not a passing gate): does the eKV Triton pass GENERALIZE to a SECOND
engine + kernel + memory model? Runs SGLang's Triton attention backend under the source-triton + the
pass. Must be run via tests/_sglang_kernel_run.sh (which sets the source-triton + libstdc++ preload +
the pass env). The `if __name__ == "__main__"` mask is REQUIRED -- SGLang SPAWNS its scheduler, and a
without mask enforcement main re-imports and re-spawns (RuntimeError: bootstrapping phase).

FINDING (2026-06-22, SGLang 0.5.12) -- VALIDATED-GENERAL, run via tests/_sglang_kernel_run.sh (which
adds the sitecustomize bootstrap; without it register() never runs -> 0 buffers, a BOOTSTRAP miss, not a
pass-logic failure). After the eKVTimer.cpp CSR/GEMV/i64 refactor:
  * #3 SUBPROCESS bootstrap is ZERO-TOUCH -- register() runs in 4 pids (main + spawned scheduler +
    workers) via sitecustomize; it is a PLUGIN REGISTER (idiomatic: an sglang.srt.plugins entry point),
    not an engine touch. Contribution intact.
  * The pass detects SGLang's attention kernels NAME-FREE: prefill `_fwd_kernel` (GEMM, tt.dot) AND
    decode `_fwd_kernel_stage1` (GEMV, tt.reduce, NO tt.dot, the dominant path) both select via the
    contraction rule (dot OR reduce); leaf=kv_indices (i64), base=K_Buffer; `_fwd_kernel_stage2` and the
    token-pool helpers excluded.
  * CORRECTNESS proven by tests/_sglang_oracle_run.sh: buffer[r]==kv_indices[kv_indptr[r]:] for 96/96
    decode requests vs SGLang's own arrays, names poisoned. See GENERALIZATION.md S8.
"""
import os
import sys

import sglang


def main():
    # EKV_SGLANG_MODEL lets the oracle swap in a GQA model (e.g. Qwen2.5-0.5B, kv_group_num>1) to
    # exercise the grouped-GQA decode kernel `_fwd_grouped_kernel_stage1`; default opt-125m is MHA.
    model = os.environ.get("EKV_SGLANG_MODEL", "facebook/opt-125m")
    try:
        llm = sglang.Engine(model_path=model, attention_backend="triton",
                            mem_fraction_static=0.45, disable_cuda_graph=True, log_level="warning")
    except Exception as e:  # noqa: BLE001
        print("ENGINE_INIT_FAIL:", repr(e)[:700])
        sys.exit(2)
    try:
        out = llm.generate("The capital of France is", {"max_new_tokens": 8, "temperature": 0.0})
        print("GEN_OK:", (out["text"] if isinstance(out, dict) else out))
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
