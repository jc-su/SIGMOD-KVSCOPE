//===- eKV/Passes.h - eKV MLIR pass plugin API ------------------*- C++ -*-===//
//
// eKV ships an MLIR pass that runs inside Triton's compile pipeline as a real
// out-of-tree plugin (the tritonGetPluginInfo() ABI in triton/Tools/PluginUtils.h,
// PR #8401, merged 2025-11-22; Triton >= 3.7/3.8 built with -DTRITON_EXT_ENABLED=1).
//
//   eKVTimer  — bracket each *selected* kernel with a clock64 read at entry and
//               every return, and atomic-add the per-CTA elapsed cycles into a
//               fixed-address device control block, indexed by
//               slot = program_id(axis).
//
// Methodology mirrors an earlier Triton plugin (native typed Triton ops + OpTy::create,
// linked against `triton`), but the emission is strictly simpler than that plugin's
// check-emission: no taint / owner-table / mask / redirect — just a timer
// bracket and one scalar atomic.
//
// Config is read from function attributes set on the LIVE module by the Python
// installer (runtime/ekv_plugin.py); with no attrs the pass is a no-op, so it
// only ever touches kernels eKV explicitly selected. The control-block address
// is baked in as a constant, so NO kernel launch argument is added and the
// engine's launch site is never touched (unlike that plugin, whose per-launch cap
// tensors require a launch wrapper).
//
//===----------------------------------------------------------------------===//
#ifndef EKV_PASSES_H
#define EKV_PASSES_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace mlir {
class ModuleOp;
}  // namespace mlir

namespace ekv {

/// Per-CTA clock64 timing pass. Defined in eKVTimer.cpp.
std::unique_ptr<mlir::Pass> createEKVTimerPass();

/// Register eKV passes with MLIR's global registry (so triton-opt -ekv-timer
/// works). Defined in eKVPlugin.cpp.
void registerEKVPasses();

namespace attrs {
/// String attr on a tt.func: the decimal device address of the u64[max_slots]
/// control block. Presence of this attr SELECTS the function for instrumentation.
constexpr llvm::StringLiteral kCtrlPtr = "ekv.ctrl_ptr";
/// i32 attr: the program_id axis that identifies the per-request slot (default 0).
constexpr llvm::StringLiteral kSlotAxis = "ekv.slot_axis";
/// i32 attr: number of slots in the control block (default 1024).
constexpr llvm::StringLiteral kMaxSlots = "ekv.max_slots";
/// Unit attribute marking a function already instrumented (idempotency marker).
constexpr llvm::StringLiteral kInstrumented = "ekv.instrumented";

// --- request<->block capture (GPU-side, for engines whose KV-location map is
// device-resident, e.g. SGLang): snapshot the block-table row the kernel reads,
// per slot=program_id(axis), into a second device control block. Captured WHERE
// the map is materialized (the kernel reads it to gather KV), so it works whether
// the engine builds the map on the host (vLLM) or the device (SGLang). Scalar,
// unrolled K entries -> stage-agnostic (no tensor layout encoding needed).
/// String attr: decimal device address of the i32[max_slots * bt_ncap] block-table
/// output buffer. Presence SELECTS the function for block-table capture.
constexpr llvm::StringLiteral kBtCtrlPtr = "ekv.bt_ctrl_ptr";
/// i32 attr: index of the kernel argument that is the block_table pointer.
constexpr llvm::StringLiteral kBtArg = "ekv.bt_arg";
/// i32 attr: source row stride (max blocks per request = block_table column count).
constexpr llvm::StringLiteral kBtRowStride = "ekv.bt_row_stride";
/// i32 attr: number of block-table entries to capture per slot (K, the dst width).
constexpr llvm::StringLiteral kBtNcap = "ekv.bt_ncap";
/// i32 attr (optional): a program_id axis to head-deduplicate on; the capture store
/// is masked to that axis == 0. Absent -> capture on every invocation (idempotent).
constexpr llvm::StringLiteral kBtHeadAxis = "ekv.bt_head_axis";
/// i32 attr (optional): number of rows in the source block_table. When set, the
/// capture is masked to slot < max_rows, so padding programs whose slot exceeds the
/// real row count (e.g. vLLM unified-attention's q-block grid is an upper bound on
/// num_seqs) neither read past the block_table nor write past the output buffer.
constexpr llvm::StringLiteral kBtMaxRows = "ekv.bt_max_rows";
}  // namespace attrs

}  // namespace ekv
#endif  // EKV_PASSES_H
