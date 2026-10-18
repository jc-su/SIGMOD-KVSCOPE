"""Manage eBPF kv_alloc/kv_copy attachment and guaranteed detachment.

This replaces the bash attach/pkill chains whose leaked uprobes on libcuda poisoned
every subsequent CUDA process on this box (observed as CUBLAS_INTERNAL_ERROR at warmup). Kills are
by exact PID, never by pattern, so a cleanup can never take out its own caller.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Literal, NamedTuple, Self

EKV = Path(__file__).resolve().parents[1]
LIBCUDA = "/usr/lib/x86_64-linux-gnu/libcuda.so.1"
STOP_TIMEOUT_S = 30.0  # seconds a collector gets to exit after SIGINT (see EbpfSession.detach)


def mapped_libcudarts(maps: str) -> list[str]:
    """Every CUDA runtime library in a ``/proc/<pid>/maps`` text, sorted and without repeats.

    A probe attaches to a file, not to a process, so the collector must be given each file the target
    calls. One process can map several runtimes: torch 2.11+cu130 calls ``libcudart.so.13`` while a
    dependency maps ``libcudart.so.12``. Raises ``RuntimeError`` if there is none."""
    found = sorted(
        {
            fields[-1]
            for fields in (line.split() for line in maps.splitlines())
            if len(fields) >= 6 and Path(fields[-1]).name.startswith("libcudart.so")
        }
    )
    if not found:
        raise RuntimeError("the process maps no libcudart.so: the copy collector has nothing to attach to")
    return found


def _collector_pid(sudo_pid: int, name: str) -> int:
    """The pid of the collector ``name`` that the ``sudo`` process ``sudo_pid`` started.

    With ``Defaults use_pty`` in sudoers, sudo runs the command under a monitor process and does not
    relay a signal that another process sends to sudo, so the collector itself must get the signal.
    The collector is a descendant of ``sudo_pid`` whose command name is ``name``."""
    frontier = [sudo_pid]
    while frontier:
        pid = frontier.pop()
        if Path(f"/proc/{pid}/comm").read_text().strip() == name:
            return pid
        frontier += [int(child) for child in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
    raise RuntimeError(f"no {name} process under sudo pid {sudo_pid}")


class _Collector(NamedTuple):
    name: str
    sudo: subprocess.Popen[bytes]
    pid: int


class EbpfSession:
    """Attach lifecycle collectors to ``pid`` and expose their capture paths."""

    def __init__(
        self, pid: int, out_dir: Path, alloc: bool = True, copy: bool = True, metadata: bool = False
    ) -> None:
        self.pid = pid
        self.out = Path(out_dir)
        self.want_alloc, self.want_copy = alloc, copy
        self.metadata = bool(metadata)
        self.kv_alloc = self.out / "kv_alloc.jsonl"
        self.kv_copy = self.out / "kv_copy.jsonl"
        # the kv_copy collector's stderr: its attach log and, after detach, its loss counters
        # (read them with observe.parse_collector_report)
        self.kv_copy_report = self.kv_copy.with_suffix(".err")
        # the kv_alloc collector's stderr; after detach it ends with "ringbuf drops: N"
        # (read it with observe.parse_ringbuf_drops)
        self.kv_alloc_report = self.kv_alloc.with_suffix(".err")
        self._collectors: list[_Collector] = []

    def _start(self, name: str, command: list[str], output: Path) -> None:
        """Start a collector, and fail with its own error message if it does not stay up.

        A collector that cannot load its programs (a kernel without a map type it needs, or a missing
        privilege) exits at once. Its output would then be empty, which looks like a workload that did
        no copies. So the error output is kept, and an early exit is an error."""
        errors = output.with_suffix(".err")
        p = subprocess.Popen(command, stdout=open(output, "w"), stderr=open(errors, "w"))  # noqa: SIM115
        time.sleep(1.0)
        if p.poll() is not None:
            raise RuntimeError(
                f"the {name} collector exited with code {p.returncode} while attaching to pid {self.pid}: "
                f"{errors.read_text().strip()[-400:]}"
            )
        self._collectors.append(_Collector(name, p, _collector_pid(p.pid, name)))

    def attach_alloc(self) -> None:
        self._start(
            "kv_alloc",
            [
                "sudo",
                str(EKV / "observe/ebpf/kv_alloc"),
                str(self.pid),
                f"{LIBCUDA}:cuMemAlloc_v2",
                "-m",
                "1048576",
            ],
            self.kv_alloc,
        )

    def attach_copy(self) -> None:
        cmd = ["sudo"]
        if self.metadata:
            cmd += ["env", "EKV_META=1"]
        runtimes = mapped_libcudarts(Path(f"/proc/{self.pid}/maps").read_text())
        cmd += [str(EKV / "observe/ebpf/kv_copy"), ":".join(runtimes), str(self.pid)]
        self._start("kv_copy", cmd, self.kv_copy)

    def detach(self) -> None:
        """Stop every collector by its exact pid and wait for it to exit.

        SIGINT lets a collector drain its buffer, print its loss counters, and detach its probes.
        Detaching is slow: kv_copy removes about 40 uprobes, which takes about 3 s on Linux 5.15.
        A collector that has not exited STOP_TIMEOUT_S later gets SIGKILL, and then this raises. A
        collector that outlives the detach leaks its uprobes on libcuda, which breaks every later
        CUDA process on this host, so that also raises."""
        for c in self._collectors:
            subprocess.run(["sudo", "kill", "-INT", str(c.pid)], capture_output=True, check=False)
        killed: list[str] = []
        for c in self._collectors:
            try:
                c.sudo.wait(timeout=STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                subprocess.run(["sudo", "kill", "-9", str(c.pid)], capture_output=True, check=False)
                c.sudo.wait(timeout=5)
                killed.append(c.name)
        alive = [f"{c.name} pid {c.pid}" for c in self._collectors if Path(f"/proc/{c.pid}").exists()]
        self._collectors.clear()
        if alive:
            raise RuntimeError(f"eBPF collectors still alive after detach: {alive}")
        if killed:
            raise RuntimeError(f"eBPF collectors did not stop on SIGINT and were killed: {killed}")

    def __enter__(self) -> Self:
        if self.want_alloc:
            self.attach_alloc()
        if self.want_copy:
            self.attach_copy()
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.detach()
        return False
