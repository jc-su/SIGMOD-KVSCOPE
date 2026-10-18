"""The import hook of ``runtime.sglang_hooks``: a patch waits for the first import of its module, and runs once."""

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest

from runtime import sglang_hooks


@pytest.fixture
def module_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.syspath_prepend(str(tmp_path))
    return tmp_path


def test_handler_runs_after_the_module_body(module_dir: Path) -> None:
    (module_dir / "ekv_hooked_one.py").write_text("VALUE = 1\n")
    seen: list[int] = []
    sglang_hooks.when_imported("ekv_hooked_one", lambda module: seen.append(module.VALUE))
    assert seen == []  # nothing runs until the module is imported
    importlib.import_module("ekv_hooked_one")
    assert seen == [1]  # the handler saw the finished module
    importlib.reload(sys.modules["ekv_hooked_one"])
    assert seen == [1]  # and it runs once


def test_handler_can_patch_the_module(module_dir: Path) -> None:
    (module_dir / "ekv_hooked_two.py").write_text("def answer():\n    return 1\n")

    def patch(module: ModuleType) -> None:
        original = module.answer
        module.__dict__["answer"] = lambda: original() + 41

    sglang_hooks.when_imported("ekv_hooked_two", patch)
    assert importlib.import_module("ekv_hooked_two").answer() == 42


def test_an_imported_module_is_patched_at_once(module_dir: Path) -> None:
    (module_dir / "ekv_hooked_three.py").write_text("VALUE = 3\n")
    importlib.import_module("ekv_hooked_three")
    seen: list[int] = []
    sglang_hooks.when_imported("ekv_hooked_three", lambda module: seen.append(module.VALUE))
    assert seen == [3]
