"""Install the eKVTimer Triton pass into the live JIT pipeline and own the
device control block. Mirrors an earlier plugin installer, but is
simpler: eKV adds NO kernel argument and never touches the launch site — the
pass bakes the control-block address in as a constant.

Requirements (the .so is loaded by Triton's plugin host):
  * a Triton built with the out-of-tree pass-plugin ABI (PR #8401, >= 3.7/3.8)
    and TRITON_EXT_ENABLED=1;
  * env TRITON_EXT_ENABLED=1 and TRITON_PLUGIN_PATHS=.../libeKVPlugin.so.

Usage:
    from runtime.ekv_plugin import timing
    with timing(max_slots=256, axis=0, allowlist=["unified_attention"]) as ctrl:
        run_some_triton_kernels()
        cycles_per_slot = ctrl.read()        # u64[max_slots], per-CTA clock64
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os

import torch

# Function-attribute names — must match mlir_pass/include/eKV/Passes.h.
_ATTR_CTRL = "ekv.ctrl_ptr"
_ATTR_AXIS = "ekv.slot_axis"
_ATTR_MAXSLOTS = "ekv.max_slots"
_ATTR_BT_CTRL = "ekv.bt_ctrl_ptr"
_ATTR_BT_ARG = "ekv.bt_arg"
_ATTR_BT_ROWSTRIDE = "ekv.bt_row_stride"
_ATTR_BT_NCAP = "ekv.bt_ncap"
_ATTR_BT_HEADAXIS = "ekv.bt_head_axis"
_ATTR_BT_MAXROWS = "ekv.bt_max_rows"


class BlockTableBlock:
    """An i32[max_slots * ncap] device buffer. The eKV pass snapshots, per
    slot=program_id(axis), the first `ncap` block_table entries the attention kernel
    reads — recovering request<->block GPU-side, where the map is materialized (works
    even when the engine keeps it device-resident, e.g. SGLang). Stable data_ptr()."""

    def __init__(self, max_slots: int, ncap: int, device: str = "cuda"):
        self.max_slots = int(max_slots)
        self.ncap = int(ncap)
        self.buf = torch.full((self.max_slots * self.ncap,), -1, dtype=torch.int32,
                              device=device)
        self.ptr = int(self.buf.data_ptr())

    def reset(self) -> None:
        self.buf.fill_(-1)

    def read(self):
        """Host copy as int32[max_slots, ncap]; row s = the blocks slot s's query read."""
        return self.buf.detach().to("cpu").clone().view(self.max_slots, self.ncap)


class ControlBlock:
    """A u64[max_slots] device accumulator. Each instrumented CTA atomic-adds its
    clock64 cycle delta into acc[program_id(axis)]. Same storage across resets,
    so its data_ptr() is stable for the process (CUDA-graph safe)."""

    def __init__(self, max_slots: int = 1024, device: str = "cuda"):
        self.max_slots = int(max_slots)
        self.buf = torch.zeros(self.max_slots, dtype=torch.int64, device=device)
        self.ptr = int(self.buf.data_ptr())

    def reset(self) -> None:
        self.buf.zero_()

    def read(self):
        """Host copy of the per-slot cycle accumulator (torch.int64[max_slots])."""
        return self.buf.detach().to("cpu").clone()


_state = {"on": False}


def _resolve_ekv_pass(passes):
    """The eKV pass entry across Triton plugin-API generations, as a callable(pm) or None.

    PR #8401 (<= mid-3.8) auto-loaded TRITON_PLUGIN_PATHS at import and exposed each plugin
    pass as `passes.plugin.<name>`. Current triton-src instead exposes
    `passes.plugin.extend_with(path)`, which loads the .so on demand and registers
    `passes.plugin.add_<name>(pm, args)`. The .so ABI (tritonGetPluginInfo, API v2) is
    unchanged -- only the Python surface moved -- so one resolver covers both."""
    plug = getattr(passes, "plugin", None)
    if plug is None:
        return None
    if hasattr(plug, "ekv"):                                  # the auto-registered surface
        return plug.ekv
    if not hasattr(plug, "add_ekv") and hasattr(plug, "extend_with"):
        for p in os.environ.get("TRITON_PLUGIN_PATHS", "").split(":"):
            if p:
                plug.extend_with(p)                           # idempotent per path
    if hasattr(plug, "add_ekv"):                              # the extend_with surface
        return lambda pm: plug.add_ekv(pm, [])
    return None


def _attach_and_run(module, *, ctrl_ptr, axis, max_slots, allowlist, bt=None):
    """Attach the manifest attrs to the live TTIR module and run the native pass
    on it directly — no text round-trip (same approach as an earlier plugin).
    `bt` (or None) is a dict of block-table-capture params."""
    from triton._C.libtriton import ir, passes

    ctx = module.context
    b = ir.builder(ctx)

    names = [n for n in (allowlist or []) if module.has_function(n)]
    if not names:
        if allowlist:                 # allowlist given but no name in THIS module -> skip it
            return module             # (the hook is global; only touch named kernels)
        names = [module.get_entry_func_name()]   # no allowlist -> instrument the entry func

    for name in names:
        f = module.get_function(name)
        if ctrl_ptr:
            f.set_attr(_ATTR_CTRL, b.get_string_attr(str(ctrl_ptr)))
            f.set_attr(_ATTR_MAXSLOTS, b.get_int32_attr(int(max_slots)))
        f.set_attr(_ATTR_AXIS, b.get_int32_attr(int(axis)))
        if bt:
            f.set_attr(_ATTR_BT_CTRL, b.get_string_attr(str(bt["ctrl_ptr"])))
            f.set_attr(_ATTR_BT_ARG, b.get_int32_attr(int(bt["arg"])))
            f.set_attr(_ATTR_BT_ROWSTRIDE, b.get_int32_attr(int(bt["row_stride"])))
            f.set_attr(_ATTR_BT_NCAP, b.get_int32_attr(int(bt["ncap"])))
            if bt.get("head_axis") is not None:
                f.set_attr(_ATTR_BT_HEADAXIS, b.get_int32_attr(int(bt["head_axis"])))
            if bt.get("max_rows") is not None:
                f.set_attr(_ATTR_BT_MAXROWS, b.get_int32_attr(int(bt["max_rows"])))

    pm = ir.pass_manager(ctx)
    _resolve_ekv_pass(passes)(pm)
    pm.run(module, "ekv_timer")
    return module


def install(ctrl: "ControlBlock | None" = None, *, axis: int = 0, allowlist=None,
            bt: dict | None = None) -> None:
    """Install the compile hook. Instrumentation runs at the TTIR stage (layout
    free, so the pass never deals with TTGIR encodings). Idempotent. Pass `ctrl`
    for the clock64 timer, `bt` (dict: ctrl_ptr, arg, row_stride, ncap, head_axis)
    for GPU-side request<->block capture, or both."""
    from triton import knobs

    if not (os.environ.get("TRITON_EXT_ENABLED") and os.environ.get("TRITON_PLUGIN_PATHS")):
        raise RuntimeError(
            "eKV needs TRITON_EXT_ENABLED=1 and TRITON_PLUGIN_PATHS pointing at "
            "libeKVPlugin.so (the plugin is loaded by Triton's host).")
    try:
        from triton._C.libtriton import passes
        if _resolve_ekv_pass(passes) is None:
            raise AttributeError
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "the eKV pass is unavailable in the running Triton (neither passes.plugin.ekv "
            "nor extend_with/add_ekv): build triton-src with TRITON_EXT_ENABLED=1 "
            "(>= 3.7) and point TRITON_PLUGIN_PATHS at libeKVPlugin.so.") from e

    if ctrl is None and bt is None:
        raise ValueError("install needs ctrl (timer) and/or bt (block-table capture)")
    kw = dict(ctrl_ptr=(ctrl.ptr if ctrl else 0), axis=int(axis),
              max_slots=(ctrl.max_slots if ctrl else 0),
              allowlist=sorted(allowlist) if allowlist else None, bt=bt)
    # The manifest must be part of the compile-cache key (documented
    # add_stages_inspection_hook contract: no-arg call returns (key, hash)), so
    # changing ctrl_ptr / axis / allowlist recompiles instead of reusing a stale
    # uninstrumented kernel. The pass also writes addresses of this process into the
    # kernel, so the key also names the process (see process_compile_key).
    from ekv_vllm_plugin import process_compile_key

    manifest = "ekv:" + json.dumps(kw, sort_keys=True)

    def _key() -> tuple[str, str]:
        key = f"{manifest}:{process_compile_key()}"
        return key, hashlib.sha256(key.encode()).hexdigest()

    def _ttir(module):
        return _attach_and_run(module, **kw)

    def hook(self=None, stages=None, options=None, language=None, capability=None):
        if all(a is None for a in (stages, options, language, capability)):
            return _key()
        orig_ttir = stages.get("ttir")
        if orig_ttir is not None:
            stages["ttir"] = lambda src, md, _o=orig_ttir: _ttir(_o(src, md))
        return _key()

    knobs.runtime.add_stages_inspection_hook = hook
    _state["on"] = True


def uninstall() -> None:
    from triton import knobs
    knobs.runtime.add_stages_inspection_hook = None
    _state["on"] = False


def is_installed() -> bool:
    return _state["on"]


@contextlib.contextmanager
def timing(max_slots: int = 1024, axis: int = 0, allowlist=None, device: str = "cuda"):
    """Scoped install. Yields the ControlBlock; read() it after your kernels run."""
    ctrl = ControlBlock(max_slots=max_slots, device=device)
    install(ctrl, axis=axis, allowlist=allowlist)
    try:
        yield ctrl
    finally:
        uninstall()


@contextlib.contextmanager
def capturing(*, max_slots: int, ncap: int, bt_arg: int, row_stride: int, axis: int = 0,
              head_axis=None, max_rows=None, allowlist=None, device: str = "cuda", ctrl=None):
    """Scoped GPU-side request<->block capture. Yields a BlockTableBlock; read() it
    after the kernels run -> int32[max_slots, ncap], row s = the physical blocks the
    query block in slot s read. `max_rows` (the source block_table's row count) guards
    padding programs whose slot exceeds it. Pass a ControlBlock as `ctrl` to also time."""
    bt_blk = BlockTableBlock(max_slots=max_slots, ncap=ncap, device=device)
    bt = dict(ctrl_ptr=bt_blk.ptr, arg=int(bt_arg), row_stride=int(row_stride),
              ncap=int(ncap), head_axis=head_axis, max_rows=max_rows)
    install(ctrl, axis=axis, allowlist=allowlist, bt=bt)
    try:
        yield bt_blk
    finally:
        uninstall()
