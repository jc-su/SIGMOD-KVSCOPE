"""U6 / CS5: ISOLATION audit -- the system as a runtime correctness checker. From the captured
block tables alone (no engine hook), verify that no request's attention reads a block that is
PRIVATE to another request. A block is private to request R if, across the whole run, it appears
only in rows whose content-anchor is R; a violation is a block appearing under >=2 distinct
anchors that is NOT a legitimately-shared prefix block (prefix caching). Distinct-prompt requests
(no prefix caching) must show ZERO cross-request private-block reads -- the invariant a use-after-
free / allocator-reuse-before-overwrite bug would break, invisible to the engine's own state.

The experiment runner archives stdout and parsed counters in its immutable run directory.
PASS = 0 private-block cross-reads on distinct prompts.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _probe_common  # noqa: E402, F401 (sets sys.path/env)


def main():
    from observe.capture import Capture

    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(model="facebook/opt-125m", gpu_memory_utilization=0.35, max_model_len=2048,
              enable_prefix_caching=False,          # DISTINCT prompts, no sharing -> full isolation
              attention_backend="TRITON_ATTN")
    llm.generate(["warm up the attention compile now " * 4], SamplingParams(max_tokens=8, temperature=0.0))
    capture = Capture(os.getpid())

    # distinct-content requests of varied length, several waves (exercise allocator reuse)
    block_anchors: dict = {}                             # block -> set of anchors seen under it
    row_of_anchor: dict = {}                            # anchor -> its full block set (accumulated)
    for wave in range(4):
        prompts = [f"Wave {wave} doc {i} unique-{wave}-{i}. "
                   + " ".join(f"line {wave}-{i}-{j} topic {i} item {j}." for j in range(5 + i * 6))
                   for i in range(6)]
        llm.generate(prompts, SamplingParams(max_tokens=48, min_tokens=48, temperature=0.0,
                                             ignore_eos=True))
        torch.cuda.synchronize()
        rows = capture.rows()
        # per-tick anchor = first block (distinct prompts -> no prefix collision, first block unique)
        for s, row in rows.items():
            a = row[0]
            row_of_anchor.setdefault(a, set()).update(row)
            for b in set(row):
                block_anchors.setdefault(b, set()).add(a)

    # a block read by >=2 distinct anchors, with prefix caching OFF and distinct prompts, is a
    # cross-request read of memory that "belongs" to another request -> an isolation violation.
    cross = {b: anchors for b, anchors in block_anchors.items() if len(anchors) >= 2}
    n_blocks = len(block_anchors)
    n_cross = len(cross)
    n_requests = len(row_of_anchor)

    print(f"EKV_ISOLATION requests={n_requests} distinct_blocks={n_blocks} "
          f"cross_request_private_reads={n_cross}", flush=True)
    ok = n_cross == 0 and n_blocks > 50
    print("PASS: the capture plane certifies request isolation -- 0 blocks read across requests "
          "(prefix caching off, distinct prompts); a use-after-free / reuse-before-overwrite bug "
          "would surface here, invisible to the engine's own state"
          if ok else f"FAIL: {n_cross} cross-request block reads (blocks={n_blocks})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
