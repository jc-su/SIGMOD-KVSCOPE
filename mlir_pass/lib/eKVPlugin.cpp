//===- eKVPlugin.cpp - eKV out-of-tree Triton pass plugin entry ----------===//
//
// Exports the C-linkage entry point tritonGetPluginInfo() (the contract in
// triton/Tools/PluginUtils.h, PR #8401). Triton loads the .so from
// TRITON_PLUGIN_PATHS and exposes the pass two ways:
//   * triton-opt -ekv-timer file.mlir          (via the registerPass callback)
//   * passes.plugin.ekv(pm) from Python         (via the addPass callback),
//     which runtime/ekv_plugin.py runs from inside the JIT pipeline through
//     knobs.runtime.add_stages_inspection_hook.
//
// Modeled directly on an earlier Triton plugin (already built into this exact
// Triton, examples/plugins/) and the in-tree example examples/plugins/
// TritonPlugin.cpp.
//
//===----------------------------------------------------------------------===//

#include "eKV/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "triton/Tools/PluginUtils.h"

namespace ekv {
void registerEKVPasses() {
  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createEKVTimerPass(); });
}
}  // namespace ekv

namespace {

// AddPassCallback: append eKV passes to a live PassManager.
void addEKVPasses(mlir::PassManager *pm, const std::vector<std::string> &args) {
  pm->addPass(ekv::createEKVTimerPass());
}

// RegisterPassCallback.
void registerEKVPassesCb() { ekv::registerEKVPasses(); }

const char *kPluginName = "eKVPlugin";
const char *kPluginVersion = "0.1.0";
const char *kPassName = "ekv";

}  // namespace

TRITON_PLUGIN_API mlir::triton::plugin::PluginInfo *tritonGetPluginInfo() {
  using namespace mlir::triton::plugin;
  static PassInfo pass = {kPassName, kPluginVersion, addEKVPasses,
                          registerEKVPassesCb};
  static PassInfo passes[] = {pass};
  static PluginInfo info = {TRITON_PLUGIN_API_VERSION,
                            kPluginName,
                            kPluginVersion,
                            passes,
                            /*numPasses=*/1,
                            /*dialects=*/nullptr,
                            /*numDialects=*/0,
                            /*ops=*/nullptr,
                            /*numOps=*/0,
                            TRITON_VERSION};
  return &info;
}
