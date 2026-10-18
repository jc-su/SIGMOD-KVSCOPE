"""Injectors for the eKV eval, loaded via PYTHONPATH so they run in EVERY python the serve stack
spawns -- crucially the engine-core CHILD process. One opt-in post-import patch:

  * Ground-truth oracle (only when EKV_ORACLE_OUT is set). Patches Scheduler.schedule to record, every
    step, each running request's real (prompt_token_ids, physical block_ids) from the engine's own
    KVCacheManager, flushed to EKV_ORACLE_OUT.<pid> (per-process). For the fidelity check only; unset
    for overhead runs so the server is fully stock.
"""
import os
import sys
import importlib.abc
import importlib.util

_HANDLERS = {}
_INSTALLED = [False]


class _Finder(importlib.abc.MetaPathFinder):
    """Let the real machinery load a target module, then run its handler after exec."""
    def find_spec(self, name, path=None, target=None):
        fn = _HANDLERS.get(name)
        if fn is None:
            return None
        for f in list(sys.meta_path):
            if f is self:
                continue
            spec = f.find_spec(name, path, target)
            if spec and spec.loader:
                _orig = spec.loader.exec_module

                def exec_module(module, _o=_orig, _fn=fn):
                    _o(module)
                    try:
                        _fn(module)
                    except Exception:
                        pass

                spec.loader.exec_module = exec_module
                return spec
        return None


def _register(name, fn):
    _HANDLERS[name] = fn
    if not _INSTALLED[0]:
        sys.meta_path.insert(0, _Finder())
        _INSTALLED[0] = True


# --- ground-truth oracle (opt-in) ---
if os.environ.get("EKV_ORACLE_OUT"):
    import json
    import threading
    import time

    _OUT = os.environ["EKV_ORACLE_OUT"]
    _snaps = []
    _started = [False]

    def _flush_loop():
        out = f"{_OUT}.{os.getpid()}"
        while True:
            time.sleep(1.0)
            try:
                tmp = out + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(_snaps[-40000:], f)
                os.replace(tmp, out)
            except Exception:
                pass

    def _install_oracle(mod):
        Sched = getattr(mod, "Scheduler", None)
        if Sched is None or getattr(Sched, "_ekv_patched", False):
            return
        _orig = Sched.schedule

        def _patched(self):
            out = _orig(self)
            snap = {}
            for req in list(self.running):
                try:
                    bids = self.kv_cache_manager.get_block_ids(req.request_id)
                    snap[req.request_id] = {"p": list(req.prompt_token_ids or []),
                                            "b": list(bids[0]) if bids else []}
                except Exception:
                    pass
            _snaps.append([time.monotonic_ns(), snap])
            if not _started[0]:
                _started[0] = True
                threading.Thread(target=_flush_loop, daemon=True).start()
            return out

        Sched.schedule = _patched
        Sched._ekv_patched = True

    _register("vllm.v1.core.sched.scheduler", _install_oracle)
