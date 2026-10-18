#!/usr/bin/env bash
# Build the vendored libbpf (static, >= 1.0) + the vendored bpftool (must parse the RUNNING
# kernel's BTF) + the eKV eBPF collectors. Sources default to the running kernel's build tree
# (/lib/modules/$(uname -r)/build/tools); override LIBBPF_SRC / BPFTOOL_SRC if elsewhere.
#
# Kernel: Linux 5.8 or newer with BTF (/sys/kernel/btf/vmlinux). The collectors use the BPF ring
# buffer (5.8) and CO-RE; vmlinux.h is generated from the running kernel. Tested on 5.15 and 7.1.1.
#
# Why vendored, not distro: the collectors need libbpf >= 1.0 (func_name uprobe resolution and the
# generic SEC("uprobe")); the Ubuntu 22.04 distro libbpf is 0.5. A distro bpftool built for an older
# kernel cannot parse a newer kernel's BTF (e.g. the 5.15 bpftool fails on 7.1.1 BTF: "failed to
# load BTF ... Invalid argument"). CC=/usr/bin/gcc throughout so the build finds
# /usr/include/libelf.h (a conda gcc on PATH would not).
set -euo pipefail
EKV="$(cd "$(dirname "$0")/.." && pwd)"
KTREE="/lib/modules/$(uname -r)/build/tools"
LIBBPF_SRC="${LIBBPF_SRC:-$KTREE/lib/bpf}"
BPFTOOL_SRC="${BPFTOOL_SRC:-$KTREE/bpf/bpftool}"
DEST="$EKV/third_party/libbpf"

if [ ! -f "$DEST/usr/lib64/libbpf.a" ]; then
  echo "building libbpf (static) from $LIBBPF_SRC"
  [ -d "$LIBBPF_SRC" ] || { echo "set LIBBPF_SRC to a libbpf>=1.8 src dir"; exit 1; }
  rm -rf "$DEST"
  # NB: this Makefile uses lowercase 'prefix' and has no 'install_uapi_headers' target.
  make -s -C "$LIBBPF_SRC" CC=/usr/bin/gcc BUILD_STATIC_ONLY=1 \
    OBJDIR=/tmp/ekv_libbpf_obj DESTDIR="$DEST" prefix=/usr install
fi

if [ ! -x "$EKV/third_party/bpftool/bpftool" ]; then
  echo "building bpftool from $BPFTOOL_SRC"
  [ -d "$BPFTOOL_SRC" ] || { echo "set BPFTOOL_SRC to the kernel tree tools/bpf/bpftool"; exit 1; }
  make -s -C "$BPFTOOL_SRC" CC=/usr/bin/gcc HOSTCC=/usr/bin/gcc
  mkdir -p "$EKV/third_party/bpftool"
  cp "$BPFTOOL_SRC/bpftool" "$EKV/third_party/bpftool/bpftool"
fi

make -C "$EKV/observe/ebpf"
echo "---"
echo "built: $EKV/observe/ebpf/kv_copy"
echo "run:   sudo $EKV/observe/ebpf/kv_copy <path-to-libcudart.so> [pid]"
echo "test:  bash $EKV/scripts/test_ebpf.sh"
