cmake_minimum_required(VERSION 3.20)
project(eKVPlugin LANGUAGES CXX)

# Out-of-tree Triton pass plugin. Builds libeKVPlugin.so against an EXISTING
# Triton checkout WITHOUT modifying Triton's source tree: we only consume its
# headers + bundled MLIR/LLVM and link its libtriton.so. The .so is loaded at
# run time via TRITON_PLUGIN_PATHS and inserted into the JIT pipeline from
# Python via knobs.runtime.add_stages_inspection_hook (see runtime/ekv_plugin.py).
#
# Configure example (values from the target Triton's CMakeCache.txt):
#   cmake -S mlir_pass -B build \
#     -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
#     -DTRITON_SRC=/path/to/triton \
#     -DMLIR_DIR=/path/to/.triton/llvm/llvm-62b7cf96-ubuntu-x64-2/lib/cmake/mlir \
#     -DLLVM_DIR=/path/to/.triton/llvm/llvm-62b7cf96-ubuntu-x64-2/lib/cmake/llvm
#   cmake --build build -j

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(TRITON_SRC "" CACHE PATH "Triton source checkout to build against (the exact Triton the .so loads into)")
if(NOT TRITON_SRC)
  message(FATAL_ERROR "Pass -DTRITON_SRC=<triton checkout>")
endif()
file(GLOB TRITON_BUILD_DIRS "${TRITON_SRC}/build/cmake.*")
list(GET TRITON_BUILD_DIRS 0 TRITON_BUILD)
message(STATUS "TRITON_SRC   = ${TRITON_SRC}")
message(STATUS "TRITON_BUILD = ${TRITON_BUILD}")

# The same bundled MLIR/LLVM the target Triton was built with.
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR} ${LLVM_CMAKE_DIR})

find_library(TRITON_LIB triton
  PATHS "${TRITON_SRC}/python/triton/_C" "${TRITON_BUILD}"
  REQUIRED)
message(STATUS "libtriton    = ${TRITON_LIB}")

add_library(eKVPlugin SHARED
  lib/eKVPlugin.cpp
  lib/eKVTimer.cpp
)

target_include_directories(eKVPlugin PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include   # eKV/Passes.h
  ${TRITON_SRC}/include                 # triton/Dialect/..., triton/Tools/PluginUtils.h
  ${TRITON_BUILD}/include               # tablegen-generated *.h.inc
  ${TRITON_SRC}                         # PluginUtils.h does #include "python/src/ir.h"
  ${TRITON_SRC}/third_party/nvidia/include
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)

# Match LLVM/MLIR build flags (no RTTI), keep symbols local except the entry point.
target_compile_options(eKVPlugin PRIVATE
  -fvisibility=hidden -fvisibility-inlines-hidden -fno-rtti
  -Wno-error=deprecated-declarations)
target_compile_definitions(eKVPlugin PRIVATE ${LLVM_DEFINITIONS})

# Link libtriton (contains the Triton dialects + MLIR). On Linux any residual
# undefined symbols resolve at dlopen into the Triton host process.
target_link_libraries(eKVPlugin PRIVATE ${TRITON_LIB})
target_link_options(eKVPlugin PRIVATE -Wl,--allow-shlib-undefined)

set_target_properties(eKVPlugin PROPERTIES OUTPUT_NAME "eKVPlugin")
