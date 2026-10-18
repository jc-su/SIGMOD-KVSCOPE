"""The copy collector must attach to every CUDA runtime the target maps, and fail when there is none."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observe.ebpf_attach import mapped_libcudarts

NV = "/opt/conda/lib/python3.12/site-packages/nvidia"
MAPS = f"""\
55d0c0a00000-55d0c0a01000 r--p 00000000 08:02 1 /usr/bin/python3.12
7f0000000000-7f0000100000 r--p 00000000 08:02 2 {NV}/cu13/lib/libcudart.so.13
7f0000100000-7f0000200000 r-xp 00100000 08:02 2 {NV}/cu13/lib/libcudart.so.13
7f0000300000-7f0000400000 r-xp 00000000 08:02 3 {NV}/cuda_runtime/lib/libcudart.so.12
7f0000500000-7f0000600000 r-xp 00000000 08:02 4 /usr/lib/x86_64-linux-gnu/libcuda.so.610.57.04
7f0000700000-7f0000800000 rw-p 00000000 00:00 0
7f0000900000-7f0000a00000 r-xp 00000000 08:02 5 /opt/lib/libcudart_static_helper.so
"""


def test_every_runtime_once_sorted() -> None:
    assert mapped_libcudarts(MAPS) == [
        f"{NV}/cu13/lib/libcudart.so.13",
        f"{NV}/cuda_runtime/lib/libcudart.so.12",
    ]


def test_driver_and_anonymous_mappings_are_not_runtimes() -> None:
    maps = "\n".join(line for line in MAPS.splitlines() if "libcudart.so" not in line)
    with pytest.raises(RuntimeError, match="maps no libcudart.so"):
        mapped_libcudarts(maps)
