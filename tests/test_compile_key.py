"""The compile-cache key of the instrumented Triton pipeline belongs to one process.

The pass writes the addresses of the buffers it allocates into the kernel binary, so a binary compiled by
one process faults with an MMU error in any other process. Triton files a compiled kernel in its on-disk
cache under the key that the stages hook returns, so that key must differ between processes, also between
the tensor-parallel workers that share one cache directory and a forked child of a process.
"""

import os

import pytest

import ekv_vllm_plugin


def test_key_is_stable_within_a_process() -> None:
    assert ekv_vllm_plugin.process_compile_key() == ekv_vllm_plugin.process_compile_key()


def test_key_names_the_process() -> None:
    assert ekv_vllm_plugin.process_compile_key().startswith(f"ekv-auto-{os.getpid()}-")


def test_forked_child_gets_its_own_key(monkeypatch: pytest.MonkeyPatch) -> None:
    parent = ekv_vllm_plugin.process_compile_key()
    child = os.getpid() + 1
    monkeypatch.setattr(ekv_vllm_plugin.os, "getpid", lambda: child)  # same module state, another pid
    key = ekv_vllm_plugin.process_compile_key()
    assert key.startswith(f"ekv-auto-{child}-")
    assert key != parent


def test_a_new_process_with_a_reused_id_gets_another_key(monkeypatch: pytest.MonkeyPatch) -> None:
    first = ekv_vllm_plugin.process_compile_key()
    monkeypatch.setattr(ekv_vllm_plugin, "_compile_key", None)  # a fresh interpreter with the same pid
    assert ekv_vllm_plugin.process_compile_key() != first
