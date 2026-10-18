"""What KVScope installs in an SGLang process, chosen by the environment as in ``ekv_vllm_plugin``.

  EKV_PASS_AUTO=1             the compiler pass, which observes the attention kernel from inside the Triton JIT
  EKV_SCHEDULED_PUBLICATION=1 the block-table view and its safe point in the scheduler, which control the reads
  EKV_SITE_MODULES=a,b        extra modules to import, which a check uses to observe the engine from inside

``register`` runs at interpreter start-up, before SGLang is imported, so a patch to an SGLang module waits for the
module's first import (``when_imported``).
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
from collections.abc import Callable, Sequence
from types import ModuleType

_waiting: dict[str, list[Callable[[ModuleType], None]]] = {}


class _AfterImport(importlib.abc.MetaPathFinder):
    """Let the normal finder load a module, then run the handlers that wait for it."""

    def find_spec(
        self, name: str, path: Sequence[str] | None = None, target: ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        if name not in _waiting:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            spec = finder.find_spec(name, path, target)
            if spec is None or spec.loader is None:
                continue
            execute = spec.loader.exec_module

            def exec_module(module: ModuleType, _execute=execute, _name=name) -> None:
                _execute(module)
                for handler in _waiting.pop(_name):
                    handler(module)

            spec.loader.exec_module = exec_module  # ty: ignore[invalid-assignment]  # wraps the loader's own method
            return spec
        return None


def when_imported(name: str, handler: Callable[[ModuleType], None]) -> None:
    """Call ``handler(module)`` right after module ``name`` is first imported, or now if it already is."""
    if name in sys.modules:
        handler(sys.modules[name])
        return
    if not any(isinstance(finder, _AfterImport) for finder in sys.meta_path):
        sys.meta_path.insert(0, _AfterImport())
    _waiting.setdefault(name, []).append(handler)


def _flag(name: str) -> bool:
    return os.environ.get(name, "") not in ("", "0", "false")


def register() -> None:
    if _flag("EKV_PASS_AUTO"):
        from ekv_vllm_plugin import install_pass

        install_pass()
    if _flag("EKV_SCHEDULED_PUBLICATION"):
        from runtime.sglang_view import install

        install()
    for name in filter(None, os.environ.get("EKV_SITE_MODULES", "").split(",")):
        importlib.import_module(name)
