#!/usr/bin/env bash
# Build libeKVPlugin.so OUT OF TREE against an existing Triton checkout.
# Does NOT modify Triton's source tree (no edits to examples/plugins/CMakeLists).
# The .so is loaded at run time via TRITON_PLUGIN_PATHS.
#
# Build settings default to this box (read from triton-src/build/.../CMakeCache.txt);
# override via env if your checkout differs.
set -euo pipefail

EKV="$(cd "$(dirname "$0")/.." && pwd)"
TRITON_SRC="${TRITON_SRC:-/path/to/triton}"
CXX="${CXX:-/usr/bin/g++-12}"
LLVM_HOME="${LLVM_HOME:-}"
if [ -z "$LLVM_HOME" ]; then
  for candidate in /path/to/.triton/llvm/llvm-*; do
    if [ -f "$candidate/lib/cmake/mlir/MLIRConfig.cmake" ] &&
       [ -f "$candidate/lib/cmake/llvm/LLVMConfig.cmake" ]; then
      LLVM_HOME="$candidate"
    fi
  done
fi
[ -n "$LLVM_HOME" ] || {
  echo "FAIL: no Triton LLVM/MLIR bundle found; set LLVM_HOME" >&2
  exit 1
}

echo "eKV        : $EKV"
echo "triton-src : $TRITON_SRC"
echo "g++        : $CXX"
echo "llvm/mlir  : $LLVM_HOME"

cmake -S "$EKV/mlir_pass" -B "$EKV/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DTRITON_SRC="$TRITON_SRC" \
  -DMLIR_DIR="$LLVM_HOME/lib/cmake/mlir" \
  -DLLVM_DIR="$LLVM_HOME/lib/cmake/llvm"

cmake --build "$EKV/build" -j"$(nproc)"

echo "---"
ls -l "$EKV/build/libeKVPlugin.so"
cat <<EOF

Built (out of tree; Triton source untouched). To use it, run your Triton/vLLM
process against the matching triton-src with:

  export TRITON_EXT_ENABLED=1
  export TRITON_PLUGIN_PATHS=$EKV/build/libeKVPlugin.so
  # the triton-src .so needs the system libstdc++ (newer than conda's):
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
  export PYTHONPATH=$TRITON_SRC/python:$EKV

  python $EKV/tests/test_timer_kernel.py
EOF
