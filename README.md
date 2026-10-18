# KVScope

KVScope gives per-request observability and control of the KV cache in a continuous-batching LLM
serving engine (vLLM, SGLang). The engine's source and its attention kernel stay unchanged. KVScope
works at the block table, which maps each request to the KV blocks the attention kernel reads:

- **Observe.** A compiler pass in the Triton JIT pipeline finds the block-table lookup by data flow
  and records, per request row, the blocks read, the GPU cycles of its thread blocks, and the
  log-sum-exp of each tile of keys, from which the exact attention share of each block follows.
- **Enforce.** A view of the block table, built at a safe point between decode steps, gives each
  decode row the blocks it may read, and the engine's own kernel reads the view.
- **Decide.** An external controller reads the records, ranks each request's blocks by measured
  attention, and commits leased decisions through a shared-memory mailbox.
- **Attribute.** eBPF probes on the CUDA copy calls charge copies between memory tiers to requests.

The code keeps the project's working name, **eKV**: the package is `ekv`, and every environment
variable starts with `EKV_`.

## Terms used in the code

| Paper | Code |
|---|---|
| measurement step, measurement request | audit step, audit (`runtime/audit.py`) |
| measurement kernel | audit kernel (`runtime/audit_kernel.py`) |
| view and safe point | scheduled publication (`EKV_SCHEDULED_PUBLICATION`) |
| generation, lease | generation, lease (`observe/publication.py`) |
| measured ranking, sum objective | policy `uniform`, policy `sum` |

## Where each part of the paper is

| Paper section | Code |
|---|---|
| 3.2 Interface, Listing 1 | `observe/controller.py`, `observe/publication.py` |
| 3.3 Compiler pass: lookup by data flow, row capture, cycles, tile log-sum-exp | `mlir_pass/lib/eKVTimer.cpp`, `mlir_pass/lib/eKVPlugin.cpp`, installer `runtime/ekv_plugin.py` |
| 3.3 Exact statistic (normalization outside the GPU) | `observe/tile_shares.py`, readers `observe/capture.py`, `observe/cycles.py` |
| 3.3 Measurement kernel for non-Triton backends (FlashInfer) | `runtime/audit_kernel.py` |
| 3.3 Copies between memory tiers | `observe/ebpf/kv_copy.bpf.c`, `observe/ebpf_source.py`, `observe/ledger.py`, `observe/reloads.py` |
| 3.4 View of the block table, start-up check | `runtime/block_view.py` (`require_supported_attention`) |
| 3.5 Budget, step-time target | `observe/allocation.py`, `observe/controller.py`, `observe/daemon.py` |
| 3.6 Generations, seqlock, request keys, lease | `observe/publication.py`, `runtime/audit.py` |
| 4 vLLM plugin entry point | `ekv_vllm_plugin.py`, `pyproject.toml` |
| 4 SGLang adapter | `runtime/sglang_hooks.py`, `runtime/sglang_view.py`, `runtime/sglang_compact.py`, `runtime/sglang_site/` |

## Requirements

- Linux, an NVIDIA GPU, Python 3.11 or newer.
- vLLM 0.21, PyTorch 2.11, Triton 3.6. The tested GPU is an RTX A6000.
- For the compiler pass: a Triton source checkout that matches the installed Triton, its LLVM/MLIR
  bundle, CMake, and g++ 12.
- For SGLang: SGLang 0.5.12.
- For the eBPF probes (optional): Linux 5.8 or newer with BTF (`/sys/kernel/btf/vmlinux`), Clang 14
  or newer, libelf, and the privileges to load BPF programs. `scripts/build_ebpf.sh` builds libbpf
  and bpftool from the running kernel's source tree into `third_party/`.

## Build

```bash
pip install -e .                                   # registers the vLLM general plugin
TRITON_SRC=/path/to/triton LLVM_HOME=/path/to/llvm bash scripts/build_plugin.sh   # build/libeKVPlugin.so
bash scripts/build_ebpf.sh                         # optional
```

## Run with vLLM

Engine process:

```bash
export EKV_ENABLE=1
export EKV_SCHEDULED_PUBLICATION=1                 # the view and the safe point
export EKV_PASS_AUTO=1 EKV_ATTN_SHARE=1 EKV_TIMER=1 # the compiler pass and its records
export TRITON_EXT_ENABLED=1
export TRITON_PLUGIN_PATHS=$PWD/build/libeKVPlugin.so
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export PYTHONPATH=/path/to/triton/python:$PWD
export TRITON_CACHE_DIR=/tmp/ekv_triton_cache_$$   # one cache per process: the pass bakes in addresses
vllm serve meta-llama/Llama-3.1-8B-Instruct
```

On a backend without a Triton attention kernel, replace `EKV_ATTN_SHARE=1` with
`EKV_AUDIT_KERNEL=1`. A fixed-rule policy, such as keeping the newest blocks, needs the view alone
(`EKV_SCHEDULED_PUBLICATION=1`).

Controller, in a separate process with the same `EKV_RUNTIME_DIR` and `EKV_CONTROL_DIR`:

```bash
# a fixed budget: keep half of each request's blocks
python -m observe.daemon --follow --duration 0 --policy uniform --budget 0.5 --lease-seconds 0.5

# the isolation policy of the paper: a 40 ms step-time target on requests above 256 blocks
python -m observe.daemon --follow --duration 0 --policy uniform --target-step-ms 40 \
  --min-budget 0.2 --cap-above-blocks 256 --health-file /tmp/ekv_health.json
```

For a tensor-parallel engine, pass every worker's process id with `--pid`. If the controller
stops, every request returns to full reads when the last lease expires (0.5 s by default).
`deploy/ekv-daemon.service` is an example systemd unit.

## Environment variables

| Variable | Default | Meaning |
|---|---:|---|
| `EKV_ENABLE` | off | enable the plugin in the engine process |
| `EKV_PASS_AUTO` | off | install the compiler pass |
| `EKV_SCHEDULED_PUBLICATION` | off | install the block-table view and the safe point |
| `EKV_ATTN_SHARE` | off | record the tile statistic on measurement steps, with the compiler pass |
| `EKV_AUDIT_KERNEL` | off | record the tile statistic with the measurement kernel, on any backend |
| `EKV_TIMER` | off | per-request GPU-cycle accumulation |
| `EKV_LSE_LAYERS`, `EKV_LSE_ROWS` | 64 / 8 | layers and query rows the statistic table holds |
| `EKV_NCAP`, `EKV_MAX_ROWS` | 128 / 1024 | captured row width and scheduler-row count |
| `EKV_RUNTIME_DIR`, `EKV_CONTROL_DIR` | `/tmp`, `/dev/shm` | where the pass publishes its buffers and where the mailbox lives |
| `EKV_DEBUG` | off | report the pass's matching and instrumentation decisions |

## Tests

```bash
bash tests/run_cpu.sh        # 254 checks, no GPU: controller, allocation, view, mailbox protocol, eBPF parsing
bash tests/run_gpu.sh        # live GPU checks: statistic against the true attention share, cycles, isolation
```

Set `PY=/path/to/python` to choose the interpreter. The scripts under `tests/_*` are the individual
live checks; each `_*_run.sh` sets up its environment, and the paths in them are placeholders
(`/path/to/triton`, `/opt/conda`) to adjust to your installation.

## Layout

```text
mlir_pass/        Triton compiler pass (C++, out of tree; loaded through TRITON_PLUGIN_PATHS)
runtime/          code in the engine process: view, safe point, measurement steps, SGLang adapter
observe/          controller, mailbox, allocation, record readers, ledger, eBPF probes
ekv_vllm_plugin.py  vLLM general-plugin entry point
scripts/          build scripts
tests/            CPU and live-GPU tests
deploy/           example systemd unit for the controller
```
