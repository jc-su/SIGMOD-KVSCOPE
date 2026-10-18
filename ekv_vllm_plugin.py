"""vLLM general-plugin entry point. It installs, in every vLLM process (engine core and each
worker), what the environment asks for:

  * the compiler pass (EKV_PASS_AUTO=1), which observes: it runs inside the Triton JIT pipeline
    and does not modify engine state;
  * the block-table view and its safe point (EKV_SCHEDULED_PUBLICATION=1), which controls: a
    model-runner boundary hook exports the row-to-request association, applies completed
    controller generations, and gives each pure-decode batch its per-request view of the block
    table before attention executes. It needs no compiler pass;
  * the audit kernel (EKV_AUDIT_KERNEL=1), which observes on any attention backend: a kernel of
    its own records the attention statistic after each layer, where the compiler pass can only
    record it inside a Triton attention kernel;
  * the release of blocks (EKV_RELEASE=1), which turns reads saved into memory saved: the engine's
    scheduler returns the blocks the controller has kept out of a request's reads to the block pool.

The pass self-identifies the attention kernel (by its block-table-named arg + a kernel
name pattern), self-allocates + publishes its capture (+timer) buffer, and self-configures
stride/num_seqs from the kernel's own runtime args. Everything else — request<->block,
per-request compute and transfer latency — is read outside the engine by eBPF
(observe/ebpf/{http_label,kv_copy}) + the /proc read in observe/zero_touch.py.
The Triton plugin ABI only registers passes, so this entry point also inserts the compiler pass
into the JIT pipeline.

Opt-in via EKV_ENABLE=1. The pass needs the source-built triton (plugin ABI):
TRITON_EXT_ENABLED=1 + TRITON_PLUGIN_PATHS=.../libeKVPlugin.so + the pass tuning env
(EKV_NCAP/EKV_MAX_ROWS/EKV_HEAD_AXIS/EKV_SLOT_AXIS, EKV_TIMER for cycles). Detection is purely
structural (no argument names or configured kernel names): the pass finds the KV-index lookup
from the gather dataflow.

Registered in pyproject:
  [project.entry-points."vllm.general_plugins"]  ekv = "ekv_vllm_plugin:register"
"""

import os
import sys
import uuid

# (process id, key) of the process that made the key; see process_compile_key.
_compile_key: tuple[int, str] | None = None


def process_compile_key() -> str:
    """Return the compile-cache key of the instrumented Triton pipeline in this process.

    The pass allocates its buffers when it compiles a kernel and writes their addresses into the
    kernel binary, so a binary is valid only in the process that compiled it. Triton adds the first
    value that the stages hook returns to the key of its on-disk cache, and the second to the key of
    its in-memory cache. The hook returns this key for both. Thus a process never runs a binary that
    another process compiled: not as a cache hit, and not when two processes that share a cache
    directory (the workers of a tensor-parallel engine) compile the same kernel at the same time and
    the second replaces the files of the first before the first loads them. The random part keeps a
    later process that gets the same process id from matching the binaries of an earlier one, and a
    forked child makes a key of its own."""
    global _compile_key
    pid = os.getpid()
    if _compile_key is None or _compile_key[0] != pid:
        _compile_key = (pid, f"ekv-auto-{pid}-{uuid.uuid4().hex}")
    return _compile_key[1]


def _flag(name: str) -> bool:
    """Value-aware, matching the pass's envFlag: "0"/"false"/empty disable."""
    v = os.environ.get(name, "")
    return v not in ("", "0", "false")


def register() -> None:
    """Install what the environment asks for: the compiler pass (observation), the block-table
    view and its safe point (control), or both. They are independent: the view needs no pass."""
    if not _flag("EKV_ENABLE"):
        return
    if _flag("EKV_PASS_AUTO"):
        install_pass()
    if _flag("EKV_SCHEDULED_PUBLICATION"):
        from observe.publication import install_vllm_safe_point

        patched = install_vllm_safe_point()
        if not patched:
            raise RuntimeError("no supported vLLM GPU model runner for the block-table view")
        print(
            f"[eKV] block-table view and safe point installed on {patched} model runner(s)", file=sys.stderr
        )
    if _flag("EKV_AUDIT_KERNEL"):
        if not _flag("EKV_SCHEDULED_PUBLICATION"):
            raise RuntimeError(
                "EKV_AUDIT_KERNEL needs EKV_SCHEDULED_PUBLICATION: the view chooses the rows to audit"
            )
        from runtime.block_view import install_audit_kernel

        patched = install_audit_kernel()
        print(
            f"[eKV] the attention statistic is recorded by a kernel after each layer of {patched} backend(s)",
            file=sys.stderr,
        )
    if _flag("EKV_RELEASE"):
        if not _flag("EKV_SCHEDULED_PUBLICATION"):
            raise RuntimeError(
                "EKV_RELEASE needs EKV_SCHEDULED_PUBLICATION: the view must stop reading a block first"
            )
        from runtime.release import install_scheduler_release

        install_scheduler_release()
        print("[eKV] blocks the controller releases are returned to the engine's pool", file=sys.stderr)


def install_pass() -> None:
    """Insert the compiler pass into the Triton JIT pipeline of this process.

    The vLLM plugin calls it in each vLLM process, and ``runtime.sglang_site`` calls it in SGLang's scheduler."""
    pid = os.getpid()
    try:
        from triton import knobs
        from triton._C.libtriton import ir, passes

        def _hook(self=None, stages=None, options=None, language=None, capability=None):
            # A call with no arguments asks for (on-disk cache key, in-memory cache key).
            if all(a is None for a in (stages, options, language, capability)):
                key = process_compile_key()
                return (key, key)
            if "ttir" in stages:
                # KEEP-ALL ACROSS THE WHOLE PIPELINE: instrumentation must never break a compile. Two
                # failure classes: (1) the eKV pass itself raises (e.g. on a foreign fused kernel
                # the structural detection false-matched); (2) instrumentation succeeds but produces
                # IR a LATER stage rejects. Both decline loudly and recompile the original chain.
                orig_stages = dict(stages)
                order = list(stages.keys())
                state = {}

                def _clean_chain(upto_idx, md):
                    x = orig_stages[order[0]](state["src"], md)
                    for n in order[1 : upto_idx + 1]:
                        x = orig_stages[n](x, md)
                    return x

                def _make(idx, name, fn):
                    def run(arg, md, _fn=fn, _idx=idx, _name=name):
                        if _idx == 0:
                            state["src"] = arg
                            state["instrumented"] = False
                            m = _fn(arg, md)
                            try:
                                pm = ir.pass_manager(m.context)
                                from runtime.ekv_plugin import _resolve_ekv_pass

                                _resolve_ekv_pass(passes)(pm)
                                pm.run(m, "ekv_auto")
                                state["instrumented"] = True
                                return m
                            except Exception as exc:  # class (1)
                                print(
                                    f"[eKV] instrumentation DECLINED (pass failed): "
                                    f"{type(exc).__name__}: {exc} -- using original IR",
                                    file=sys.stderr,
                                )
                                return _fn(arg, md)
                        try:
                            return _fn(arg, md)
                        except Exception as exc:  # class (2)
                            if not state.get("instrumented"):
                                raise  # not instrumentation-induced: surface it
                            print(
                                f"[eKV] instrumentation DECLINED "
                                f"(stage {_name} failed on instrumented IR): "
                                f"{type(exc).__name__} -- recompiling clean chain",
                                file=sys.stderr,
                            )
                            state["instrumented"] = False
                            return _clean_chain(_idx, md)

                    return run

                for _i, _n in enumerate(order):
                    stages[_n] = _make(_i, _n, orig_stages[_n])
            key = process_compile_key()
            return (key, key)

        knobs.runtime.add_stages_inspection_hook = _hook
        import triton as _t

        from runtime.ekv_plugin import _resolve_ekv_pass

        ok = _resolve_ekv_pass(passes) is not None
        print(
            f"[eKV] automatic compiler-pass bootstrap installed (pid {pid}) "
            f"triton={_t.__file__} ekv_pass_available={ok}",
            file=sys.stderr,
        )
    except Exception as exc:  # noqa: BLE001  (never break the engine)
        print(f"[eKV] auto-pass bootstrap failed: {type(exc).__name__}: {exc}", file=sys.stderr)
