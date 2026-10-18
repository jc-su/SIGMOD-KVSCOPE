//===- eKVTimer.cpp - structural instrumentation for KV attention kernels -===//
//
// One structural instrumentation site -- the KV index lookup that an attention kernel must
// perform (index map -> KV gather -> contraction, detected without symbol names) -- carries every
// capability:
//
//   CAPTURE   buffer[request] = the kernel's index row (block table / CSR indices)    [default]
//   TIMER     per-request CTA-residency cycles (clock64 deltas, atomic-add per row)  [EKV_TIMER]
//   QLEN      cu_seqlens_q snapshot (recompute-vs-cache-hit signal)                  [EKV_QLEN]
//   KV_PTR    the KV pool base pointer (exact pool identification)                   [always]
//   ARITH     closed-form map (kv_base + seq*stride) for non-paged kernels            [auto]
//   SHARE     per-tile log attention weight, for the audit steps the serving     [EKV_ATTN_SHARE]
//             process flags: u = running max + log(tile sum of exp(score - max)), stored per
//             (layer, request row, block, query row). The pass only observes. A control
//             decision reaches the kernel through the block table the engine passes it
//             (runtime/block_view.py), not through the kernel.
//
// Every inserted effect is a commutative-monoid write (atomic add), an idempotent write, or a pure
// read -- the admission rule that makes all of it CUDA-graph-replay-safe. Buffers are allocated
// per COMPILE and published as one JSON line per instrumentation in /tmp/ekv_pass_<pid>.json;
// the external readers (observe/zero_touch.py, observe/ebpf/ekv_readout) disambiguate. The
// formal model and its pinning tests: MODEL.md.
//
// clock64 is the per-SM cycle counter (high resolution, sub-ns at ~GHz). It is
// NOT synchronized across SMs, but we only ever sum (end - start) DELTAS, each
// computed on one SM, so the per-slot total is a well-defined additive resource
// (CTA-residency cycles). This is the same counter Triton's own Proton profiler
// reads for its CYCLE metric (third_party/proton .../TargetInfo.cpp). The
// alternative %globaltimer is a fixed-freq ns clock but its default update
// cadence is ~1us (32ns only when a profiler forces it) -- too coarse for a
// per-CTA bracket -- so we use clock64 for the width and leave a per-kernel ns
// anchor to the host side.
//
//===----------------------------------------------------------------------===//

#include "eKV/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <dlfcn.h>
#include <map>
#include <mutex>
#include <string>
#include <unistd.h>

namespace ekv {
namespace {

namespace tt = mlir::triton;
namespace arith = mlir::arith;
using mlir::Location;
using mlir::OpBuilder;
using mlir::Type;
using mlir::Value;

// --- zero-touch self-configuration (the DEFAULT path; no Python installer) --------
// By default the pass identifies the attention kernel structurally (findIndexLookup: the index
// map -> KV gather -> contraction triad), self-allocates its capture buffer via CUDA (dlopen'd at
// compile time), and publishes the buffer address for the external eBPF reader -- so request<->block
// is captured with zero engine touch. The explicit-config attr path is opt-in via EKV_LEGACY (the
// oracle that validates this detection).

static int getEnvInt(const char *k, int dflt) {
  const char *v = std::getenv(k);
  if (!v) return dflt;
  int n = std::atoi(v);
  if (n <= 0 || n > (1 << 22)) {                 // a bad env var must not become a 0-size or
    std::fprintf(stderr, "[eKV] %s=%s invalid; using %d\n", k, v, dflt);  // multi-GB allocation
    return dflt;
  }
  return n;
}

// The original Python parameter name survives into TTIR as the arg's NameLoc
// (e.g. %block_tables_ptr ... loc("block_tables_ptr")). Return it, or "".
static std::string argName(mlir::Value arg) {
  mlir::Location loc = arg.getLoc();
  if (auto nl = mlir::dyn_cast<mlir::NameLoc>(loc))
    return nl.getName().str();
  return std::string();
}

// Bounded backward search over the SSA def chain: does `v` transitively derive from a
// get_program_id (wantRange=false) or a make_range (wantRange=true)? Used to recognize
// the block-table access pattern STRUCTURALLY (a per-slot, RANGED i32 load) — name- and
// KV-addressing-independent (we key on the index array's own load, not the KV gather).
static bool derivesFrom(mlir::Value v, int depth, bool wantRange) {
  if (depth <= 0 || !v) return false;
  mlir::Operation *op = v.getDefiningOp();
  if (!op) return false;                              // a block argument: stop
  if (wantRange ? mlir::isa<tt::MakeRangeOp>(op) : mlir::isa<tt::GetProgramIdOp>(op))
    return true;
  for (mlir::Value o : op->getOperands())
    if (derivesFrom(o, depth - 1, wantRange)) return true;
  return false;
}

// REGION-AWARE reachability (the generality lesson): if `v` is a loop-carried block argument -- an
// scf.for/scf.while/affine.for iter-arg, via the GENERIC mlir::LoopLikeOpInterface -- resolve it to its
// INIT operand so a backward walk crosses the loop boundary. An index/base advanced each step
// (`%p = addptr %p_prev, %stride` carried as an iter-arg) stays reachable to the arg it started from.
// One generic interface, not a per-dialect special-case. Returns the init for an iter-arg, else null.
static mlir::Value loopInitOf(mlir::Value v) {
  auto ba = mlir::dyn_cast<mlir::BlockArgument>(v);
  if (!ba) return {};
  auto loop = mlir::dyn_cast_or_null<mlir::LoopLikeOpInterface>(ba.getOwner()->getParentOp());
  if (!loop) return {};
  mlir::Block::BlockArgListType iterArgs = loop.getRegionIterArgs();
  mlir::OperandRange inits = loop.getInits();
  for (unsigned k = 0; k < iterArgs.size() && k < inits.size(); ++k)
    if (iterArgs[k] == v) return inits[k];
  return {};
}

// Does `v` transitively derive from the specific SSA value `target` (e.g. a kernel arg)?
// Region-aware: crosses loop iter-args to their inits, so a loop-carried index/base is still reached.
static bool derivesFromValue(mlir::Value v, mlir::Value target, int depth) {
  if (depth <= 0 || !v) return false;
  if (v == target) return true;
  mlir::Operation *op = v.getDefiningOp();
  if (!op) {                                          // a block argument
    if (mlir::Value init = loopInitOf(v))             // a loop iter-arg -> cross to its init
      return derivesFromValue(init, target, depth - 1);
    return false;                                     // a true entry arg -> stop
  }
  for (mlir::Value o : op->getOperands())
    if (derivesFromValue(o, target, depth - 1)) return true;
  return false;
}

// derivesDirectly = reachability that STOPS at a load boundary -- the operator that picks the LEAF of a
// reindexing chain (its MAXIMAL indirection level). When the walk reaches a tt.LoadOp result that is not
// `target`, it does NOT descend into that load's address: each load is one indirection level, and we want
// the index DIRECTLY below the gather, not an outer one. So the leaf (kv_indices) -- whose values address
// the KV gather directly (kv[ kv_loc*stride + ... ]) -- matches, while the delimiter (kv_indptr), which
// reaches the gather only THROUGH the leaf's load, does not (plain derivesFromValue crosses that load and
// wrongly matches both). Generalizes to any depth n (a radix/hierarchical page table) -- always the leaf.
// On vLLM's single-level block_table it is identical to derivesFromValue.
static bool derivesDirectly(mlir::Value v, mlir::Value target, int depth) {
  if (depth <= 0 || !v) return false;
  if (v == target) return true;
  mlir::Operation *op = v.getDefiningOp();
  if (!op) {                                     // a block argument
    if (mlir::Value init = loopInitOf(v))        // a loop iter-arg -> cross to its init (region-aware)
      return derivesDirectly(init, target, depth - 1);
    return false;
  }
  if (mlir::isa<tt::LoadOp>(op)) return false;   // a load result that isn't target -> different indirection level
  for (mlir::Value o : op->getOperands())
    if (derivesDirectly(o, target, depth - 1)) return true;
  return false;
}

// An index/offset array element type. Block tables and CSR index/indptr arrays are integer, but the
// WIDTH varies by engine: vLLM block_table is i32, SGLang kv_indices is i64 (and kv_indptr i32). A KV
// VALUE (the gather target) is never one of these (it is f16/bf16/fp8, or i8 for int8-quantized KV) --
// so isIndexInt distinguishes an index array/load from a value gather, width-agnostically.
static bool isIndexInt(mlir::Type t) { return t.isInteger(32) || t.isInteger(64); }

// Peel pure VIEWS off a value -- arith width/index casts and tt.splat/broadcast/bitcast -- so a
// kernel ARG that was cast (a stride widened to address width via arith.extsi) or a scalar base
// broadcast to a tensor of pointers is still recognized as that arg. A NO-OP on the validated
// kernels' uncast, scalar-addressed forms: it only EXTENDS recognition, never changes a match that
// already worked. (Fixes the casted-stride false-negative and the splat-base addptr miss.)
static mlir::Value stripView(mlir::Value v) {
  while (v) {
    mlir::Operation *op = v.getDefiningOp();
    if (op && mlir::isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp, arith::IndexCastOp,
                        tt::SplatOp, tt::BroadcastOp, tt::BitcastOp>(op))
      v = op->getOperand(0);
    else
      break;
  }
  return v;
}

// Is `v` the pointer `base`, or a pure view of it (splat/broadcast/cast)? Triton broadcasts a
// scalar base ptr arg to a tensor of ptrs before the row addptr -- addptr(splat(base), off) -- so
// plain pointer-equality on `base` would miss the row offset behind the splat.
static bool isBaseOrView(mlir::Value v, mlir::Value base) {
  return v == base || stripView(v) == base;
}

// A VALUE GATHER is the read of the KV pool itself -- the maximal-indirection value load on the slice.
// gatherView unifies the addressing STYLES so the rest of the analysis is style-agnostic (the
// generality lesson, the op-set side): a plain tt.load (address = its ptr) AND the Hopper TMA descriptor ops --
// tt.descriptor_load (address = the descriptor + the tile coords) and tt.descriptor_gather (the
// descriptor + x_offsets, the per-row gather indices = a paged gather done via TMA). On a gather op it
// fills `elem` (the loaded element type -- a KV VALUE, never an integer index) and `addrDeps` (the
// operands whose def-cones the index and the KV base flow through) and returns true; false otherwise.
// derivesFromValue/derivesDirectly then do the rest uniformly -- they already walk through
// make_tensor_descriptor's base operand, so the KV pool behind a descriptor is reached with no special
// case. Non-gather ops (and integer-valued "gathers", which are another index level) are rejected.
static bool gatherView(mlir::Operation *op, mlir::Type &elem,
                       llvm::SmallVector<mlir::Value, 4> &addrDeps) {
  auto setElem = [&](mlir::Type t) {
    auto rt = mlir::dyn_cast<mlir::RankedTensorType>(t);
    elem = rt ? rt.getElementType() : t;
  };
  if (auto ld = mlir::dyn_cast<tt::LoadOp>(op)) {
    setElem(ld.getType());
    addrDeps.push_back(ld.getPtr());
    return true;
  }
  if (auto dl = mlir::dyn_cast<tt::DescriptorLoadOp>(op)) {       // TMA tile load
    setElem(dl.getType());
    addrDeps.push_back(dl.getDesc());                            // -> make_tensor_descriptor -> KV base
    for (mlir::Value c : dl.getIndices()) addrDeps.push_back(c); // the tile coords carry the index
    return true;
  }
  if (auto dg = mlir::dyn_cast<tt::DescriptorGatherOp>(op)) {     // TMA scatter-gather (paged via TMA)
    setElem(dg.getType());
    addrDeps.push_back(dg.getDesc());
    addrDeps.push_back(dg.getXOffsets());                        // the per-row gather indices = the index map
    return true;
  }
  return false;
}

// Does the value loaded by `gather` flow into a CONTRACTION -- tt.dot (GEMM prefill) or tt.reduce
// (GEMV decode <q,k>)? That is what makes a gather a KV gather (vs a bias/scale gather). A free
// function so findKvGather can PREFER a contracted gather over a first-match non-contracted one
// (the V/bias-gather-first false-negative), and so the KV base it returns is a CONTRACTED pool.
static bool gatherFeedsContraction(tt::FuncOp func, mlir::Operation *gather) {
  if (!gather || gather->getNumResults() == 0) return false;
  mlir::Value gres = gather->getResult(0);
  bool feeds = false;
  func.walk([&](mlir::Operation *op) {
    if (feeds) return;
    if (!mlir::isa<tt::DotOp, tt::ReduceOp>(op)) return;
    for (mlir::Value operand : op->getOperands())
      if (derivesFromValue(operand, gres, 32)) { feeds = true; return; }
  });
  return feeds;
}

// Dataflow role of the KV-index lookup:
// do the VALUES loaded by `idxLoadResult` flow into a KV-gather whose address ALSO derives from a
// SEPARATE pointer arg? That is exactly the semantic role of the index map -- its loaded values index
// into the KV pool: addr_KV = kv_base + idx*stride + .... It keys on the index VALUES reaching the
// gather, NOT on how the index row is iterated (make_range / a loop / unrolled). It is the SOLE
// block-table characterization (no make_range / program_id special-case, no fallback): on vLLM's
// kernel_unified_attention it is UNIQUE -- of the kernel's i32 loads, only the block-table row's
// values reach the f16 KV gather (the scalar seq-len / num-seq loads do not; verified from the TTIR).
// THE shared gather characterization (one helper, two callers): find a VALUE gather (gatherView -- an
// addptr load OR a TMA descriptor op, loading the KV dtype, never an integer index level) whose address
// deps INCLUDE the index (per the caller's `indexReaches` test) AND a SEPARATE non-index value-ptr arg
// (the KV pool base, != `idxSkip`). For a descriptor gather the deps are {descriptor, coords/x_offsets}:
// the KV base reaches via the descriptor (make_tensor_descriptor, threaded by derivesFromValue), the
// index via the coords. Returns the gather op (null if none) + the base via `baseOut`. The two callers
// differ ONLY in the index test: derivesDirectly(dep, leafLoad) to DISCOVER the leaf index (the maximal
// indirection, not an outer delimiter), or derivesFromValue(dep, idxArg) once the index arg is KNOWN.
static mlir::Operation *findKvGather(tt::FuncOp func,
                                     llvm::function_ref<bool(mlir::Value)> indexReaches,
                                     mlir::Value idxSkip, mlir::Value *baseOut) {
  // Scan all gathers and prefer one that feeds an attention contraction, falling
  // back to the first triad match if none is contracted. First-match alone could pick a bias/scale
  // gather (or V before K) and then wrongly reject the kernel / publish the wrong KV base.
  mlir::Operation *res = nullptr, *anyMatch = nullptr;
  mlir::Value base, anyBase;
  func.walk([&](mlir::Operation *g) {
    if (res) return;                                              // already found a CONTRACTED match
    mlir::Type elem;
    llvm::SmallVector<mlir::Value, 4> deps;
    if (!gatherView(g, elem, deps) || isIndexInt(elem)) return;   // a KV-VALUE gather, not an index level
    bool idxR = false;
    for (mlir::Value d : deps)
      if (indexReaches(d)) { idxR = true; break; }
    if (!idxR) return;
    for (unsigned i = 0; i < func.getNumArguments(); ++i) {
      mlir::Value a = func.getArgument(i);
      if (a == idxSkip) continue;
      auto pt = mlir::dyn_cast<tt::PointerType>(a.getType());
      if (!pt || isIndexInt(pt.getPointeeType())) continue;       // KV base = value ptr, not an index array
      for (mlir::Value d : deps)
        if (derivesFromValue(d, a, 32)) {                         // a separate value base = the KV pool
          if (!anyMatch) { anyMatch = g; anyBase = a; }           // remember the first triad match
          if (gatherFeedsContraction(func, g)) { res = g; base = a; }  // PREFER a contracted gather
          return;
        }
    }
  });
  if (!res) { res = anyMatch; base = anyBase; }                   // none contracted -> first triad match
  if (baseOut) *baseOut = base;
  return res;
}

static bool loadFeedsKvGather(tt::FuncOp func, mlir::Value idxLoadResult, mlir::Value idxSrcArg) {
  // DISCOVER the leaf: the index VALUES must reach the gather DIRECTLY (not through a further index load).
  return findKvGather(func, [&](mlir::Value d) { return derivesDirectly(d, idxLoadResult, 32); },
                      idxSrcArg, nullptr) != nullptr;
}

// Self-allocate host-pinned, device-mapped buffers (the GPU writes them via the device
// ptr -> lands in host RAM, which the external eBPF/proc-mem reader reads with no
// in-process Python): a CAPTURE buffer (block_table, i32[max_rows*ncap]) and, when the
// timer is on, a TIMER buffer (per-slot clock64 cycles, u64[max_rows]). Publishes both
// addresses to /tmp/ekv_pass_<pid>.json. dlopen at compile time uses the engine's
// already-initialized CUDA context. Allocated once per process.
// CONSISTENCY/PERF (the host-mapped tradeoff): the device-side writes are RELAXED + GPU-scope, so
// they become host-visible at KERNEL COMPLETION (stream sync), not mid-flight -- the external reader
// MUST read these buffers AFTER the kernel/stream syncs (the capture is idempotent per request; the
// timer is a running sum, so a mid-flight read is partial). Host-mapped stores cross PCIe (slower
// than HBM) and the timer atomic contends across heads on the same slot; both are small and accepted
// for the zero-touch (no engine edit) goal.
// PER-KERNEL buffer sets. In a torch.compile'd engine SEVERAL kernels can legitimately match a
// index lookup (the attention kernel, its autotune/prefill variants, and foreign inductor gathers/GEMVs
// that share the structure). A process-global buffer set let the first match claim the buffers and
// every later match write into them -- cross-kernel pollution of the capture, timer, and KV pointer.
// Each instrumentation therefore allocates its OWN set and publishes its own json line, tagged with
// the kernel name; the external reader disambiguates. A set is ~capBytes+timer+qsl (~150 KB at the
// defaults), so even dozens of matching kernels cost a few MB.
// The tile log-sum-exp statistic. All compiled variants in one process and device share one block,
// because the statistic belongs to the serving process, not to a kernel specialization: a decode
// step runs one variant for every layer, and the serving process writes one flag and one table.
//   lse         f32 [2][layers][rows][width][groupRows]: plane 0 the tile max M, plane 1 the tile
//               sum l of the kernel's softmax; NaN until a launch writes it (device)
//   flag        u32, nonzero while the serving process wants collection (an audit step)
//   layerTable  u64 [layers], the KV-cache base address of each layer, filled by the serving
//               process, so a kernel finds its layer from its own KV pointer
// Each is exported with a CUDA-IPC handle for an out-of-process reader.
struct SharedShare {
  uint64_t lseDev = 0, flagDev = 0, layerTableDev = 0;
  char lseIpcHex[129] = {0}, flagIpcHex[129] = {0}, layerTableIpcHex[129] = {0};
  int layers = 0, rows = 0, width = 0, groupRows = 0;
};

struct PubSet {
  uint64_t capDev = 0, capHost = 0;
  char capShm[64] = {0};                              // /dev/shm name (unprivileged external read)
  // Per-request cycle timer: DEVICE memory (was host-mapped -- a per-CTA clock64 atomic to
  // host-mapped memory is a non-posted PCIe round trip EACH, ~24% on decode-heavy 3B; capture is
  // fine host-mapped because it writes once per LAUNCH, not per CTA). Device L2 atomics are cheap;
  // the reader does cudaMemcpy D2H in-process, or opens the
  // exported CUDA-IPC handle cross-process. timerHost stays 0 (no host mapping).
  uint64_t timerDev = 0, timerHost = 0;
  char timerIpcHex[129] = {0};
  uint64_t kvPtrDev = 0, kvPtrHost = 0;
  uint64_t qslDev = 0, qslHost = 0;
  // Evaluation-only exact launch count for the measured coverage matrix.  It is allocated and
  // emitted only under EKV_COVERAGE=1 and is never included in overhead arms.
  uint64_t coverageLaunchDev = 0;
  char coverageLaunchIpcHex[129] = {0};
  SharedShare share;                                  // the tile log-sum-exp statistic (EKV_ATTN_SHARE)
};
struct ArithPub {
  uint64_t kvPtrDev = 0, kvPtrHost = 0;
  uint64_t strideDev = 0, strideHost = 0;
  uint64_t timerDev = 0, timerHost = 0;
};
// The publication path: $EKV_RUNTIME_DIR/ekv_pass_<pid>.json (default /tmp). A descriptor FILE
// is the right pattern for a zero-touch publisher inside a foreign process (no server thread, no
// lifecycle, discoverable, crash-tolerant -- the CUDA-MPS / X11 / XDG-runtime precedent); the
// configurable directory is what fixes /tmp's warts (shared namespace, systemd PrivateTmp
// isolation). Production: RuntimeDirectory=ekv + EKV_RUNTIME_DIR=/run/ekv on BOTH engine and
// daemon (docs/OPERATIONS.md).
static void pubPath(char *buf, size_t n) {
  const char *dir = std::getenv("EKV_RUNTIME_DIR");
  std::snprintf(buf, n, "%s/ekv_pass_%d.json", dir && *dir ? dir : "/tmp", (int)getpid());
}

static void coveragePath(char *buf, size_t n) {
  const char *dir = std::getenv("EKV_RUNTIME_DIR");
  std::snprintf(buf, n, "%s/ekv_coverage_%d.json", dir && *dir ? dir : "/tmp", (int)getpid());
}

static bool g_jsonFirst = true;    // first publication truncates the json; later ones APPEND
static bool g_coverageFirst = true;
static std::mutex g_capMutex;

static void publishCoverage(const char *kernel, const char *classification, const char *reason,
                            uint64_t hostVa = 0, int shareSites = 0, int shareTileBlocks = 0,
                            bool shareNewestRows = false) {
  std::lock_guard<std::mutex> lk(g_capMutex);
  char path[256];
  coveragePath(path, sizeof(path));
  if (FILE *f = std::fopen(path, g_coverageFirst ? "w" : "a")) {
    ::chmod(path, 0600);
    g_coverageFirst = false;
    std::fprintf(f,
        "{\"pid\":%d,\"kernel\":\"%s\",\"host_va\":%llu,\"abi_version\":3,"
        "\"coverage_class\":\"%s\",\"coverage_reason\":\"%s\",\"attn_share_sites\":%d,"
        "\"attn_share_tile_blocks\":%d,\"attn_share_newest_rows\":%d}\n",
        (int)getpid(), kernel ? kernel : "", (unsigned long long)hostVa,
        classification ? classification : "untouched", reason ? reason : "", shareSites,
        shareTileBlocks, shareNewestRows ? 1 : 0);
    std::fclose(f);
  }
}

// dlopen'd cudaHostAlloc(Mapped) + cudaHostGetDevicePointer; returns device ptr (0 fail).
static uint64_t mapAlloc(size_t bytes, int fill, uint64_t *hostOut) {
  static void *h = dlopen("libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
  if (!h) h = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
  if (!h) { std::fprintf(stderr, "[eKV] dlopen cudart failed\n"); return 0; }
  static auto setFlags = (int (*)(unsigned))dlsym(h, "cudaSetDeviceFlags");
  static auto hostAlloc = (int (*)(void **, size_t, unsigned))dlsym(h, "cudaHostAlloc");
  static auto getDevPtr = (int (*)(void **, void *, unsigned))dlsym(h, "cudaHostGetDevicePointer");
  if (!hostAlloc || !getDevPtr) { std::fprintf(stderr, "[eKV] dlsym cudart failed\n"); return 0; }
  if (setFlags) setFlags(0x08u);                  // cudaDeviceMapHost (best-effort)
  void *host = nullptr;
  if (hostAlloc(&host, bytes, 0x02u) != 0 || !host) {  // cudaHostAllocMapped
    std::fprintf(stderr, "[eKV] cudaHostAlloc(%zu) failed\n", bytes); return 0; }
  std::memset(host, fill, bytes);
  void *dev = nullptr;
  if (getDevPtr(&dev, host, 0) != 0 || !dev) {
    std::fprintf(stderr, "[eKV] cudaHostGetDevicePointer failed\n"); return 0; }
  if (hostOut) *hostOut = (uint64_t)host;
  return (uint64_t)dev;
}

// CUDA-IPC handle for a device pointer, hex-encoded into `out[129]` (empty string on failure).
// Everything the external daemon must read or write exports one.
static void ipcHex(uint64_t devPtr, char *out) {
  out[0] = 0;
  if (!devPtr)
    return;
  static void *h = dlopen("libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
  if (!h) h = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
  static auto ipcGet = h ? (int (*)(void *, void *))dlsym(h, "cudaIpcGetMemHandle") : nullptr;
  unsigned char hdl[64] = {0};
  if (ipcGet && ipcGet(hdl, (void *)devPtr) == 0)
    for (int i = 0; i < 64; ++i)
      std::snprintf(out + 2 * i, 3, "%02x", hdl[i]);
}

// SHM-backed host-mapped alloc: the buffer lives in a named POSIX shared-memory segment
// (0600), mmap'd here and cudaHostRegister'd so the kernel can write it -- byte-identical to
// cudaHostAlloc(Mapped) on the device side (same posted-store PCIe path), but an EXTERNAL
// consumer mmaps /dev/shm/<name> directly instead of reading /proc/<pid>/mem, which removes
// the daemon's root/CAP_SYS_PTRACE requirement entirely (that read was its only root need).
// Falls back to mapAlloc when shm fails. Segments are unlinked at process exit; a SIGKILL can
// leak one (stale, 0600, skipped by liveness checks; ops cleanup in docs/OPERATIONS.md).
static char g_shmNames[16][64];
static int g_shmCount = 0;
static pid_t g_shmOwner = 0;
static void shmCleanup() {
  if (getpid() != g_shmOwner)
    return;  // a FORKED child (NCCL/CUDA helpers fork-and-exit) inherits this handler; letting
             // it run there unlinks the parent's live segments by name and breaks every
             // name-based reader (the daemon) while the in-process mappings keep working.
  for (int i = 0; i < g_shmCount; ++i)
    shm_unlink(g_shmNames[i]);
}
static uint64_t shmAlloc(size_t bytes, int fill, uint64_t *hostOut, char *nameOut, size_t nameLen) {
  static int counter = 0;
  std::snprintf(nameOut, nameLen, "/ekv_%d_%d", (int)getpid(), counter++);
  int fd = shm_open(nameOut, O_CREAT | O_EXCL | O_RDWR, 0600);
  if (fd < 0) { nameOut[0] = 0; return 0; }
  if (ftruncate(fd, (off_t)bytes) != 0) { close(fd); shm_unlink(nameOut); nameOut[0] = 0; return 0; }
  void *host = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (host == MAP_FAILED) { shm_unlink(nameOut); nameOut[0] = 0; return 0; }
  std::memset(host, fill, bytes);
  static void *h = dlopen("libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
  if (!h) h = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
  static auto reg = h ? (int (*)(void *, size_t, unsigned))dlsym(h, "cudaHostRegister") : nullptr;
  static auto gdp = h ? (int (*)(void **, void *, unsigned))dlsym(h, "cudaHostGetDevicePointer") : nullptr;
  if (!reg || !gdp || reg(host, bytes, 0x02u) != 0) {          // cudaHostRegisterMapped
    munmap(host, bytes); shm_unlink(nameOut); nameOut[0] = 0; return 0; }
  void *dev = nullptr;
  if (gdp(&dev, host, 0) != 0 || !dev) {
    munmap(host, bytes); shm_unlink(nameOut); nameOut[0] = 0; return 0; }
  if (g_shmCount < 16) {
    std::snprintf(g_shmNames[g_shmCount++], 64, "%s", nameOut);
    if (g_shmCount == 1) { g_shmOwner = getpid(); std::atexit(shmCleanup); }
  }
  if (hostOut) *hostOut = (uint64_t)host;
  return (uint64_t)dev;
}

// DEVICE-memory alloc (cudaMalloc + cudaMemset) for tables the KERNEL reads on its hot path.
static uint64_t devAlloc(size_t bytes, int fillByte) {
  static void *h = dlopen("libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
  if (!h) h = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
  if (!h) return 0;
  static auto dMalloc = (int (*)(void **, size_t))dlsym(h, "cudaMalloc");
  static auto dMemset = (int (*)(void *, int, size_t))dlsym(h, "cudaMemset");
  if (!dMalloc || !dMemset) return 0;
  void *dev = nullptr;
  if (dMalloc(&dev, bytes) != 0 || !dev) return 0;
  dMemset(dev, fillByte, bytes);
  return (uint64_t)dev;
}

// cudaMemset is STREAM-ORDERED, not host-synchronous: a kernel compiled mid-run (vLLM launches
// on non-blocking streams) can launch before its tables' memset lands -> uninitialized imp/tau
// -> nondeterministic drops (caught as wobbling round-1 hashes + sampler index asserts in
// default-MP mode, where a 5th attention variant compiles late). One device sync at the end of
// allocation closes it: the kernel's first launch strictly follows compilation return.
static void devSync() {
  static void *h = dlopen("libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
  if (!h) h = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
  static auto sync = h ? (int (*)())dlsym(h, "cudaDeviceSynchronize") : nullptr;
  if (sync) sync();
}

// Triton may compile several shape specializations for one serving worker. They have distinct
// capture buffers but share one statistic block (see SharedShare), kept per CUDA device in this
// process. selfAllocCapture holds g_capMutex while entering here, so the map and the first
// allocation are serialized with publication. Every variant must ask for the same dimensions; a
// variant that asks for others gets an empty block and is left without the statistic.
static SharedShare sharedShareForCurrentDevice(int layers, int rows, int width, int groupRows) {
  static std::map<int, SharedShare> shares;
  static void *h = dlopen("libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
  if (!h) h = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
  static auto getDevice = h ? (int (*)(int *))dlsym(h, "cudaGetDevice") : nullptr;
  int device = 0;
  if (getDevice && getDevice(&device) != 0)
    device = 0;
  auto it = shares.find(device);
  if (it != shares.end()) {
    const SharedShare &s = it->second;
    bool same = s.layers == layers && s.rows == rows && s.width == width && s.groupRows == groupRows;
    return same ? s : SharedShare();
  }
  SharedShare s;
  s.layers = layers, s.rows = rows, s.width = width, s.groupRows = groupRows;
  size_t cells = (size_t)layers * rows * width * groupRows;
  s.lseDev = devAlloc(2u * cells * 4u, 0xff);         // 0xFFFFFFFF is NaN: no launch wrote the cell
  s.flagDev = devAlloc(4u, 0x00);                     // collection off until the serving process asks
  s.layerTableDev = devAlloc((size_t)layers * 8u, 0x00);
  if (!s.lseDev || !s.flagDev || !s.layerTableDev)
    s = SharedShare();
  else {
    ipcHex(s.lseDev, s.lseIpcHex);
    ipcHex(s.flagDev, s.flagIpcHex);
    ipcHex(s.layerTableDev, s.layerTableIpcHex);
  }
  shares.emplace(device, s);
  return s;
}

static PubSet doSelfAlloc(size_t capBytes, size_t timerBytes, size_t qslBytes, int ncap,
                          int maxRows, int qslCount, int shareLayers, int shareGroupRows,
                          const char *kernel) {
  PubSet ps;
  ps.capDev = shmAlloc(capBytes, 0xff, &ps.capHost, ps.capShm, sizeof(ps.capShm));
  if (!ps.capDev)                                     // shm unavailable -> classic host-mapped
    ps.capDev = mapAlloc(capBytes, 0xff, &ps.capHost);  // (-1 sentinel for unwritten rows)
  if (!ps.capDev) return ps;
  if (timerBytes) {
    ps.timerDev = devAlloc(timerBytes, 0x00);           // DEVICE memory: cheap per-CTA atomics
    ipcHex(ps.timerDev, ps.timerIpcHex);                // the daemon READS via IPC
  }
  ps.kvPtrDev = mapAlloc(8, 0x00, &ps.kvPtrHost);     // KV-cache base ptr -> identifies the pool
  if (qslBytes)
    ps.qslDev = mapAlloc(qslBytes, 0xff, &ps.qslHost);  // cu_seqlens_q snapshot (-1 = unwritten)
  if (shareLayers > 0)
    ps.share = sharedShareForCurrentDevice(shareLayers, maxRows, ncap, shareGroupRows);
  if (std::getenv("EKV_COVERAGE")) {
    ps.coverageLaunchDev = devAlloc(8u, 0x00);
    ipcHex(ps.coverageLaunchDev, ps.coverageLaunchIpcHex);
  }
  devSync();                       // tables MUST be initialized before any launch
  char path[256];
  pubPath(path, sizeof(path));
  if (FILE *f = std::fopen(path, g_jsonFirst ? "w" : "a")) {
    ::chmod(path, 0600);   // the JSON carries IPC handles to device buffers: owner-only
    g_jsonFirst = false;
    std::fprintf(f, "{\"pid\":%d,\"kernel\":\"%s\",\"abi_version\":4,\"host_va\":%llu,\"dev_ptr\":%llu,\"bytes\":%zu,"
                    "\"ncap\":%d,\"max_rows\":%d,\"timer_host_va\":%llu,\"timer_bytes\":%zu,"
                    "\"timer_dev\":%llu,\"timer_ipc\":\"%s\","
                    "\"kv_ptr_host_va\":%llu,\"kv_ptr_bytes\":%d,"
                    "\"qsl_host_va\":%llu,\"qsl_count\":%d,"
                    "\"tile_lse_dev\":%llu,\"tile_lse_ipc\":\"%s\",\"tile_lse_layers\":%d,"
                    "\"tile_lse_group_rows\":%d,"
                    "\"stat_flag_dev\":%llu,\"stat_flag_ipc\":\"%s\","
                    "\"layer_table_dev\":%llu,\"layer_table_ipc\":\"%s\","
                    "\"coverage_launch_dev\":%llu,\"coverage_launch_ipc\":\"%s\","
                    "\"cap_shm\":\"%s\"}\n",
                 (int)getpid(), kernel ? kernel : "",
                 (unsigned long long)ps.capHost, (unsigned long long)ps.capDev,
                 capBytes, ncap, maxRows, (unsigned long long)ps.timerHost, timerBytes,
                 (unsigned long long)ps.timerDev, ps.timerIpcHex,
                 (unsigned long long)ps.kvPtrHost, ps.kvPtrDev ? 8 : 0,
                 (unsigned long long)ps.qslHost, ps.qslDev ? qslCount : 0,
                 (unsigned long long)ps.share.lseDev, ps.share.lseIpcHex, ps.share.layers,
                 ps.share.groupRows,
                 (unsigned long long)ps.share.flagDev, ps.share.flagIpcHex,
                 (unsigned long long)ps.share.layerTableDev, ps.share.layerTableIpcHex,
                 (unsigned long long)ps.coverageLaunchDev, ps.coverageLaunchIpcHex, ps.capShm);
    std::fclose(f);
  }
  int device = -1;
  {
    static void *h = dlopen("libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
    static auto getDevice = h ? (int (*)(int *))dlsym(h, "cudaGetDevice") : nullptr;
    if (getDevice) getDevice(&device);
  }
  std::fprintf(stderr, "[eKV] self-alloc capture kernel=%s device=%d host=0x%llx ncap=%d max_rows=%d "
               "timer=0x%llx kvptr=0x%llx qsl=0x%llx(%d) -> %s\n", kernel ? kernel : "?", device,
               (unsigned long long)ps.capHost, ncap, maxRows, (unsigned long long)ps.timerHost,
               (unsigned long long)ps.kvPtrHost, (unsigned long long)ps.qslHost, qslCount, path);
  return ps;
}

static PubSet selfAllocCapture(size_t capBytes, int ncap, int maxRows, size_t timerBytes,
                               size_t qslBytes, int qslCount, int shareLayers, int shareGroupRows,
                               const char *kernel) {
  // Per COMPILE, not per kernel name: sharing one buffer across a kernel's compiled shapes was
  // tried and REVERTED -- a single-sequence graph-capture/profile compile writes its dummy table
  // into row 0 last (last-writer-wins), polluting that request's identity ANCHOR. Per-compile
  // buffers stay clean, and the content-addressed reader (cycles.rows_match / attention_entries)
  // selects the real decode variant and, for cycles, merges variants' timers in the reader. A
  // failed alloc leaves THIS compile uninstrumented, loudly; the next compile allocates its own.
  std::lock_guard<std::mutex> lk(g_capMutex);
  PubSet ps = doSelfAlloc(capBytes, timerBytes, qslBytes, ncap, maxRows, qslCount, shareLayers,
                          shareGroupRows, kernel);
  if (!ps.capDev) {
    static int attempts = 0;
    std::fprintf(stderr, "[eKV] WARNING: capture-buffer alloc failed (attempt %d, CUDA context not "
                 "ready?) -- kernel %s is left UNINSTRUMENTED (loud skip, coverage gap "
                 "observable).\n", ++attempts, kernel ? kernel : "?");
  }
  return ps;
}

// Arithmetic-index fallback alloc: a fully-arithmetic kernel (no index array) has a CLOSED-FORM
// request->region map -- region(seq) = kv_base + seq*stride -- so there is NO per-request capture
// array to fill. We publish only the two scalars the formula needs: the KV base (kv_ptr) and the
// per-request stride. The external reader computes region(seq) for every request with no GPU buffer.
static ArithPub doSelfAllocArith(size_t timerBytes, int maxRows, const char *kernel) {
  ArithPub ap;
  ap.kvPtrDev = mapAlloc(8, 0x00, &ap.kvPtrHost);
  if (!ap.kvPtrDev) return ap;
  ap.strideDev = mapAlloc(8, 0x00, &ap.strideHost);
  if (timerBytes)
    ap.timerDev = mapAlloc(timerBytes, 0x00, &ap.timerHost);
  devSync();                       // tables MUST be initialized before any launch
  char path[256];
  pubPath(path, sizeof(path));
  if (FILE *f = std::fopen(path, g_jsonFirst ? "w" : "a")) {
    ::chmod(path, 0600);   // the JSON carries WRITE-capable IPC handles (mask tables): owner-only
    g_jsonFirst = false;
    std::fprintf(f, "{\"pid\":%d,\"kernel\":\"%s\",\"row_form\":\"arith\","
                    "\"kv_ptr_host_va\":%llu,\"kv_ptr_bytes\":%d,"
                    "\"arith_stride_host_va\":%llu,\"timer_host_va\":%llu,\"timer_bytes\":%zu,"
                    "\"max_rows\":%d}\n",
                 (int)getpid(), kernel ? kernel : "",
                 (unsigned long long)ap.kvPtrHost, ap.kvPtrDev ? 8 : 0,
                 (unsigned long long)ap.strideHost, (unsigned long long)ap.timerHost,
                 timerBytes, maxRows);
    std::fclose(f);
  }
  std::fprintf(stderr, "[eKV] self-alloc ARITH kernel=%s kvptr=0x%llx stride=0x%llx timer=0x%llx "
               "-> %s\n", kernel ? kernel : "?", (unsigned long long)ap.kvPtrHost,
               (unsigned long long)ap.strideHost, (unsigned long long)ap.timerHost, path);
  return ap;
}
static ArithPub selfAllocArith(size_t timerBytes, int maxRows, const char *kernel) {
  std::lock_guard<std::mutex> lk(g_capMutex);
  return doSelfAllocArith(timerBytes, maxRows, kernel);
}

class EKVTimerPass
    : public mlir::PassWrapper<EKVTimerPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EKVTimerPass)

  llvm::StringRef getArgument() const final { return "ekv-timer"; }
  llvm::StringRef getDescription() const final {
    return "Bracket each selected kernel with clock64 and atomic-add per-CTA "
           "elapsed cycles into a fixed-address device control block";
  }

  void runOnOperation() override {
    getOperation().walk([&](tt::FuncOp func) { instrument(func); });
  }

 private:
  // Read the 64-bit cycle counter via inline PTX. Equivalent to the Python
  // tl.inline_asm_elementwise("mov.u64 $0, %clock64;", "=l", [], tl.int64,
  // is_pure=False, pack=1) used to read special registers.
  static Value readClock64(OpBuilder &b, Location loc) {
    Type i64 = b.getI64Type();
    auto op = tt::ElementwiseInlineAsmOp::create(
        b, loc, mlir::TypeRange{i64},
        /*asm_string=*/"mov.u64 $0, %clock64;",
        /*constraints=*/"=l",
        /*pure=*/false,
        /*packed_element=*/1,
        /*args=*/mlir::ValueRange{});
    return op.getResult()[0];
  }

  static int getIntAttr(tt::FuncOp func, llvm::StringLiteral key, int dflt) {
    if (auto a = func->getAttrOfType<mlir::IntegerAttr>(key))
      return static_cast<int>(a.getInt());
    return dflt;
  }

  // GPU-side request<->block capture: snapshot the bt_ncap entries of the
  // block_table row for slot=program_id(axis) into the i32 output buffer, as ONE
  // coalesced masked vector load + store (tensor ops, no encoding -> Triton's own
  // layout passes assign the layout downstream, exactly as for user tensor ops).
  // The attention kernel reads block_table[slot, :] to gather KV, so this captures
  // the device-resident map where it is materialized. Inserted at `b` (kernel entry).
  //
  // Width W = next power of two >= ncap (make_range requires pow2); tail lanes
  // [ncap, W) are masked off. With head-dedup the SAME mask gates both the load and
  // the store, so heads != 0 do zero memory traffic (not just a suppressed write).
  // STRUCTURAL detection of the INDEX MAP -- the array whose loaded values are physical KV unit ids (the
  // address map index lookup's index I; GENERALIZATION.md). Engine-agnostic instances: vLLM block_table (paged,
  // block size B=16), SGLang kv_indices (token pool, B=1). Identified by dataflow role, name- and
  // iteration-independent: the index map is the integer pointer arg whose VECTOR load's values index the
  // KV pool -- its load feeds a KV-VALUE gather whose address ALSO derives from a SEPARATE value pointer
  // arg (the KV base): addr_KV = kv_base + unit_id*stride + .... ONE characterization (D1 based-on-arg +
  // D3 feeds-the-gather), no make_range / program_id, no fallback. The VALUES must reach the gather
  // address directly (derivesDirectly) -- this picks the leaf (kv_indices), not a CSR delimiter
  // (kv_indptr) that only contributes the offset. A fully-arithmetic index (no index array) is handled
  // separately by findArithmeticIndex (the closed-form region map); a TMA-hidden index remains out of reach
  // here. Returns the arg or -1 (this kernel is not index-based attention).
  int findIndexMapArgStructural(tt::FuncOp func) {
    for (unsigned i = 0; i < func.getNumArguments(); ++i) {
      Value arg = func.getArgument(i);
      auto pt = mlir::dyn_cast<tt::PointerType>(arg.getType());
      if (!pt || !isIndexInt(pt.getPointeeType())) continue;     // integer index array (block ids / slots): i32 or i64
      bool found = false;
      func.walk([&](tt::LoadOp ld) {
        if (found) return;
        // The index map's load is a VECTOR of indices (one physical slot per element) -- a true
        // per-element gather index. A SCALAR integer load is a CSR OFFSET (kv_indptr / qo_indptr): it
        // is ADDED to a base, not used as a per-element index, so it is not the index map. This is what
        // separates the KV index (kv_indices) from the query-side offset (qo_indptr), which is otherwise
        // structurally identical (an integer load feeding a float gather via a separate base ptr).
        auto lt = mlir::dyn_cast<mlir::RankedTensorType>(ld.getType());
        if (!lt || !isIndexInt(lt.getElementType())) return;
        if (derivesDirectly(ld.getPtr(), arg, 24) &&            // D1: the load is BASED on THIS arg
            loadFeedsKvGather(func, ld.getResult(), arg))         // D3: its values index the KV pool
          found = true;
        // D1 uses derivesDirectly (stop-at-load), NOT derivesFromValue: the leaf load (kv_indices) is
        // the BASE pointer of the gather's addptr, whereas the CSR delimiter (kv_indptr) appears only in
        // the OFFSET, behind its own scalar load -- derivesFromValue would cross that load and wrongly
        // match the delimiter as the index map. The base is reachable without crossing a load; the
        // delimiter is not.
      });
      if (found) return static_cast<int>(i);
    }
    return -1;
  }

  // The index-map LoadOp (the integer load whose values feed the KV gather, D3) that is BASED on `arg`
  // — its RESULT is the physical KV unit ids the gather is addressed by; its PTR carries the row base.
  tt::LoadOp indexMapLoad(tt::FuncOp func, Value arg) {
    tt::LoadOp res;
    // The integer load based on `arg` whose VALUES feed the KV gather (D3) -- the SAME dataflow role by
    // which the arg is found, so no make_range / program_id test (seq_idx, the row index, is hidden in
    // an scf.while). Its RESULT is the unit ids the gather is addressed by; its PTR carries the row base
    // (seq_idx*stride for paged, kv_indptr[seq] for CSR). One characterization, no fallback.
    func.walk([&](tt::LoadOp ld) {
      if (res) return;
      auto t = mlir::dyn_cast<mlir::RankedTensorType>(ld.getType());
      if (t && isIndexInt(t.getElementType()) &&               // a VECTOR index load (not a scalar offset)
          derivesDirectly(ld.getPtr(), arg, 24) &&             // BASED on arg (not a delimiter in the offset)
          loadFeedsKvGather(func, ld.getResult(), arg))
        res = ld;
    });
    return res;
  }

  // The KV-VALUE gather OP (addptr load or TMA descriptor) + its KV-base arg, given the KNOWN index arg:
  // the value gather whose address derives from idxArg AND a separate value-ptr arg (the KV pool base).
  // Style-agnostic via the shared findKvGather (descriptor / loop / depth handled there). Returns the op
  // (null if none) and the KV base arg via `baseOut`.
  mlir::Operation *kvGatherOp(tt::FuncOp func, int idxArgIdx, Value *baseOut = nullptr) {
    if (baseOut) *baseOut = Value();
    if (idxArgIdx < 0) return nullptr;
    Value idxArg = func.getArgument(idxArgIdx);
    // the index arg is KNOWN -> any reachability into the gather addr (transitive): derivesFromValue.
    return findKvGather(func, [&](mlir::Value d) { return derivesFromValue(d, idxArg, 32); },
                        idxArg, baseOut);
  }

  // The KV pool base arg (name-independent): the value-ptr the gather reads through, alongside the index.
  Value findKvCacheArgStructural(tt::FuncOp func, int idxArgIdx) {
    Value base;
    kvGatherOp(func, idxArgIdx, &base);
    return base;
  }

  // SELECTION (GEMV- and GEMM-agnostic): a real attention kernel CONTRACTS over the gathered KV --
  // QK^T via tt.dot (GEMM prefill, vLLM + SGLang _fwd_kernel) OR qk = sum(q*k) via tt.reduce (the GEMV
  // decode, SGLang _fwd_kernel_stage1). Either contraction CONSUMING the KV gather marks attention.
  // A slot-mapping helper or a partials-combine kernel (SGLang _fwd_kernel_stage2) gathers no KV and
  // contracts nothing -> excluded. This replaces the tt.dot-only discriminator (which missed the GEMV
  // decode) and, being grounded in the KV gather, is the SOLE selector (no kernel-name fallback, which
  // would re-admit stage2). Math: attention is a contraction <q, k> over gathered k -- dot or reduce.
  bool kvGatherFeedsContraction(tt::FuncOp func, int idxArgIdx) {
    // kvGatherOp now PREFERS a contracted gather, so check the selected one via the shared free
    // helper (the same predicate findKvGather uses to prefer it).
    return gatherFeedsContraction(func, kvGatherOp(func, idxArgIdx));
  }

  // === index lookup = the data-dependent REINDEXING: the ONE principle, realized at the IR level ===
  // Attention computes a contraction <q, I*(K)>, where I*(K) is the KV pool K REINDEXED by the index map
  // I: (request,pos) -> physical slot. the index lookup is that I. Operationally I is the index array immediately below
  // the MAXIMAL-INDIRECTION value gather on the kernel's backward slice. The three facets are not
  // independent axioms -- they JOINTLY PIN the unique reindexing, each removing one impostor:
  //   (1) the gather loads VALUES, via a separate value-ptr base (findKvCacheArgStructural): the pool K
  //       holds values, not offsets   [kills an index array mistaken for the pool];
  //   (2) the gather is CONTRACTED (kvGatherFeedsContraction: tt.dot | tt.reduce): this is what makes the
  //       pool "KV"                    [kills a bias/scale gather];
  //   (3) I is the LEAF index reaching the gather DIRECTLY (derivesDirectly, stop-at-load), at ANY chain
  //       depth -- block_table (paged, depth 1) or kv_indices (CSR, depth 2), never the kv_indptr delimiter.
  // GENERALITY (the generality lesson) -- the analysis is style-agnostic in two dimensions, one principle:
  //   * OP-SET (gatherView): the value gather is an addptr tt.load OR a Hopper TMA descriptor op
  //     (tt.descriptor_load / tt.descriptor_gather -- the descriptor carries the KV base via
  //     make_tensor_descriptor, the coords/x_offsets carry the index). Validated on a real descriptor_gather
  //     in tests/_tma_index_lookup_check.py (raw TTIR -- the box is sm_86, which lowers TMA away).
  //   * REACHABILITY: the walks are loop-region-aware -- they cross loop iter-args to their inits via the
  //     generic mlir::LoopLikeOpInterface (one mechanism for scf.for/while/affine.for), so the index is
  //     found whether it is computed into the base or the offset (both operands searched), at any
  //     indirection depth, and even when its load BASE is loop-carried (GPU-validated by
  //     tests/_loop_index_lookup_check.py).
  // Returns {idxArg, kvBase, idxLoad}; idxArg < 0 == no LOADED index. A fully-arithmetic index (no index load,
  // address = kv_base + program_id(slot)*stride) is then handled by findArithmeticIndex -- a positive
  // no-index-load classification, NOT a fallback (see instrumentArith), which publishes the closed-form
  // region map. What is still out of reach at this IR level (skipped loudly, never mis-bound): an index lookup with no
  // recoverable affine structure, or a TMA-hidden index -- those need L3 (runtime trace, NVBit). See
  // GENERALIZATION.md S5.
  struct IndexLookup {
    int idxArg = -1;
    mlir::Value kvBase;
    tt::LoadOp idxLoad;
  };
  IndexLookup findIndexLookup(tt::FuncOp func) {
    int idxArg = findIndexMapArgStructural(func);                  // (3) the leaf index of the reindexing
    if (idxArg < 0) return {};
    mlir::Value kvBase = findKvCacheArgStructural(func, idxArg);   // (1) the reindexed value pool K
    if (!kvBase || !kvGatherFeedsContraction(func, idxArg))        // (2) contracted => it is the KV pool
      return {};
    tt::LoadOp idxLoad = indexMapLoad(func, func.getArgument(idxArg));
    if (!idxLoad) return {};
    return {idxArg, kvBase, idxLoad};
  }

  // === ARITHMETIC INDEX (the L1 index-only boundary closer) ===
  // Reaches a get_program_id with the SPECIFIC axis (the request/slot axis) -- so we pick the
  // REQUEST's stride term out of a multi-axis address, not the head/other-axis term.
  bool reachesPidAxis(mlir::Value v, int axis, int depth) {
    if (depth <= 0 || !v) return false;
    mlir::Operation *op = v.getDefiningOp();
    if (!op) return false;
    if (auto pid = mlir::dyn_cast<tt::GetProgramIdOp>(op))
      return static_cast<int>(pid.getAxis()) == axis;
    for (mlir::Value o : op->getOperands())
      if (reachesPidAxis(o, axis, depth - 1)) return true;
    return false;
  }

  // The coefficient of program_id(slotAxis) in `off`: the MulIOp `pid(slotAxis) * stride` = the
  // per-request stride into the KV pool. NOT a make_range factor (that is the per-element gather
  // index within a request, not the request stride). Null if the offset has no such term.
  mlir::Value slotStride(mlir::Value v, int slotAxis, int depth) {
    if (depth <= 0 || !v) return {};
    mlir::Operation *op = v.getDefiningOp();
    if (!op) return {};
    if (auto mul = mlir::dyn_cast<arith::MulIOp>(op)) {
      mlir::Value l = mul.getLhs(), r = mul.getRhs();
      bool lSlot = reachesPidAxis(l, slotAxis, 16) && !derivesFrom(l, 16, true);
      bool rSlot = reachesPidAxis(r, slotAxis, 16) && !derivesFrom(r, 16, true);
      if (lSlot && !rSlot) return r;
      if (rSlot && !lSlot) return l;
    }
    for (mlir::Value o : op->getOperands())
      if (mlir::Value s = slotStride(o, slotAxis, depth - 1)) return s;
    return {};
  }

  // Does `v` derive from a VECTOR INTEGER tt.load -- a per-element gather index (the index-map load)?
  // Its presence in a KV gather's address means the kernel is INDEX-based (block_table / kv_indices),
  // even if the index path's finer detection missed the specifics. Stops AT the load (that IS the index
  // level); it does not descend into the load's own address.
  bool derivesFromVectorIntLoad(mlir::Value v, int depth) {
    if (depth <= 0 || !v) return false;
    mlir::Operation *op = v.getDefiningOp();
    if (!op) return false;
    if (auto ld = mlir::dyn_cast<tt::LoadOp>(op)) {
      auto rt = mlir::dyn_cast<mlir::RankedTensorType>(ld.getType());
      return rt && isIndexInt(rt.getElementType());
    }
    for (mlir::Value o : op->getOperands())
      if (derivesFromVectorIntLoad(o, depth - 1)) return true;
    return false;
  }

  // The arithmetic KV-address form: a fully arithmetic address (no index array) where the request's KV
  // region is computed as kv_base + program_id(slotAxis)*stride. This is NOT a blind fallback: it runs
  // after the index path fails, but it is a positive classification of the address form: it requires
  // the KV
  // gather's address to contain NO vector-integer index load (derivesFromVectorIntLoad). So a kernel
  // that IS index-based but whose index specifics the index path missed (a false negative) still has
  // its index load, the arithmetic path REJECTS it (records sawIndexGather), and the caller SKIPS
  // LOUDLY -- it is never silently re-bound as a formula. {kvBase, stride} only for a contracted gather
  // through a separate value-ptr base with a pid(slot)*stride term AND no index load; else {}.
  struct ArithmeticIndex { mlir::Value kvBase; mlir::Value stride; bool sawIndexGather = false; };
  ArithmeticIndex findArithmeticIndex(tt::FuncOp func, int slotAxis) {
    ArithmeticIndex res;
    func.walk([&](mlir::Operation *g) {
      if (res.kvBase || res.sawIndexGather) return;                 // decided: arith found, or it's index-based
      mlir::Type elem;
      llvm::SmallVector<mlir::Value, 4> deps;
      if (!gatherView(g, elem, deps) || isIndexInt(elem)) return;   // a KV-VALUE gather, not an index level
      if (!gatherFeedsContraction(func, g)) return;                 // contracted => it is the KV pool
      for (mlir::Value d : deps)                                    // POSITIVE test: an arithmetic index has NO
        if (derivesFromVectorIntLoad(d, 32)) { res.sawIndexGather = true; return; }  // index load -> index-based, not us
      for (unsigned i = 0; i < func.getNumArguments(); ++i) {
        mlir::Value a = func.getArgument(i);
        auto pt = mlir::dyn_cast<tt::PointerType>(a.getType());
        if (!pt || isIndexInt(pt.getPointeeType())) continue;       // value-ptr base, not an index array
        for (mlir::Value d : deps) {
          mlir::Value off = addptrOffsetOnBase(d, a, 32);           // the address offset on this base
          if (!off) continue;
          if (mlir::Value s = slotStride(off, slotAxis, 32)) {      // request term = pid(slot)*stride
            res.kvBase = a;
            res.stride = s;
            return;
          }
        }
      }
    });
    return res;
  }

  // STRUCTURAL cu_seqlens_q (name-independent): the i32 ptr arg whose two loads (at adjacent
  // offsets) feed a SUBTRACTION — qsl[i+1]-qsl[i] = the query length, inside the kernel's
  // find_seq_idx. seq_lens is loaded once (no pair-difference), so this disambiguates them with
  // no names. Returns the arg or null.
  Value findQueryStartStructural(tt::FuncOp func) {
    // cu_seqlens_q is what find_seq_idx BINARY-SEARCHES: the i32 ptr arg loaded INSIDE an scf.while
    // (the search) whose adjacent elements ALSO feed a subtraction (the query length q[i+1]-q[i]).
    // kv_indptr yields the same difference shape (its csr length) but is NOT searched in a while, so
    // the in-while requirement excludes it -- the old shape-only test could mistake kv_indptr for qsl.
    Value found;
    func.walk([&](mlir::scf::WhileOp w) {
      if (found) return;
      for (unsigned i = 0; i < func.getNumArguments(); ++i) {
        Value a = func.getArgument(i);
        auto pt = mlir::dyn_cast<tt::PointerType>(a.getType());
        if (!pt || !pt.getPointeeType().isInteger(32)) continue;
        bool inWhile = false;
        w.walk([&](tt::LoadOp ld) { if (derivesFromValue(ld.getPtr(), a, 24)) inWhile = true; });
        if (!inWhile) continue;                          // not the searched array -> not cu_seqlens_q
        bool diff = false;
        func.walk([&](arith::SubIOp sub) {
          if (derivesFromValue(sub.getLhs(), a, 24) && derivesFromValue(sub.getRhs(), a, 24))
            diff = true;                                 // adjacent-element difference = query length
        });
        if (diff) { found = a; return; }
      }
    });
    return found;
  }

  // STRUCTURAL row stride: the factor multiplied by the program_id-derived slot (NOT the
  // make_range part) in the block-table address — i.e. block_table_stride, extracted from the
  // index load's own addressing. Bounded walk of the load-pointer chain. Null if not a runtime
  // scalar (then the caller's constant is used). 0-depth/clean fallback by design.
  Value strideFromChain(mlir::Value v, int depth) {
    if (depth <= 0 || !v) return Value();
    mlir::Operation *op = v.getDefiningOp();
    if (!op) return Value();
    if (auto mul = mlir::dyn_cast<arith::MulIOp>(op)) {
      mlir::Value l = mul.getLhs(), r = mul.getRhs();
      bool lSlot = derivesFrom(l, 16, false) && !derivesFrom(l, 16, true);   // program_id, not range
      bool rSlot = derivesFrom(r, 16, false) && !derivesFrom(r, 16, true);
      if (lSlot && !rSlot) return r;
      if (rSlot && !lSlot) return l;
    }
    for (mlir::Value o : op->getOperands())
      if (Value s = strideFromChain(o, depth - 1)) return s;
    return Value();
  }

  // A value the kernel COMPUTES (has a defining op that is not a constant) -- vs a kernel argument
  // (block argument, no defining op) or a literal. seq_idx is computed (find_seq_idx); the row stride
  // is a kernel arg / constant. Used to disambiguate the two operands of the row-offset multiply.
  static bool isComputed(mlir::Value v) {
    v = stripView(v);                              // a cast/broadcast of an arg or constant is still that
    mlir::Operation *op = v.getDefiningOp();
    return op && !mlir::isa<arith::ConstantOp, arith::ConstantIntOp>(op);
  }

  // The OFFSET in the addptr(base, offset) whose base is exactly `base` (the row-base address
  // block_table + row_offset), searched in the def-cone of `v`. For the block-table load this returns
  // the i64 row offset = seq_idx * stride. Null if not found within depth.
  mlir::Value addptrOffsetOnBase(mlir::Value v, mlir::Value base, int depth) {
    if (depth <= 0 || !v) return mlir::Value();
    mlir::Operation *op = v.getDefiningOp();
    if (!op) {                                         // a block argument
      if (mlir::Value init = loopInitOf(v))            // a loop-carried base -> the row base before the loop
        return addptrOffsetOnBase(init, base, depth - 1);
      return mlir::Value();
    }
    if (auto ap = mlir::dyn_cast<tt::AddPtrOp>(op))
      if (isBaseOrView(ap.getPtr(), base)) return ap.getOffset();   // base, or splat/cast view of it
    for (mlir::Value o : op->getOperands())
      if (mlir::Value r = addptrOffsetOnBase(o, base, depth - 1)) return r;
    return mlir::Value();
  }

  // From the block-table load's address addptr(idxArg, seq_idx*stride), recover BOTH factors from the
  // SAME multiply (so they are mutually consistent -- the bug before was extracting stride and seq_idx
  // from independent searches that landed on different multiplies). seq_idx = the COMPUTED operand
  // (the kernel's find_seq_idx result); stride = the kernel-arg/constant operand. Capturing
  // buffer[seq_idx] de-duplicates a prefill sequence's per-query-block writes and aligns rows to
  // sequences (idempotent write); at decode seq_idx == program_id, so the validated path is unchanged.
  // Returns {seq_idx, stride}; either may be null (caller falls back to program_id@entry / env stride).
  std::pair<mlir::Value, mlir::Value> pagedRowFactors(tt::LoadOp ld, mlir::Value idxArg) {
    if (!ld) return {mlir::Value(), mlir::Value()};
    mlir::Value off = addptrOffsetOnBase(ld.getPtr(), idxArg, 24);
    if (!off) return {mlir::Value(), mlir::Value()};
    if (auto mul = mlir::dyn_cast_or_null<arith::MulIOp>(stripView(off).getDefiningOp())) {
      mlir::Value l = mul.getLhs(), r = mul.getRhs();
      bool lc = isComputed(l), rc = isComputed(r);
      if (lc && !rc) return {l, r};               // l = seq_idx, r = stride
      if (rc && !lc) return {r, l};               // r = seq_idx, l = stride
      return {mlir::Value(), mlir::Value()};      // can't disambiguate -> fall back
    }
    return {off, mlir::Value()};                   // no stride factor -> off is seq_idx directly
  }

  // The CSR start load in `v`'s def-cone: a SCALAR i32 tt.load that reads through a DISTINCT i32 ptr
  // arg (the delimiter kv_indptr, != the leaf kv_indices). This is cur_seq_kv_start_idx =
  // load(kv_indptr + cur_seq), the per-request additive base into the token pool. Stops at any other
  // load (don't cross unrelated indirection). Null on a vLLM block table (its row offset is
  // seq_idx*stride -- a multiply, no delimiter load), so the CSR path never fires there.
  tt::LoadOp csrStartLoad(mlir::Value v, mlir::Value leafArg, tt::FuncOp func, int depth) {
    if (depth <= 0 || !v) return tt::LoadOp();
    mlir::Operation *op = v.getDefiningOp();
    if (!op) return tt::LoadOp();
    if (auto ld = mlir::dyn_cast<tt::LoadOp>(op)) {
      if (isIndexInt(ld.getType())) {                        // SCALAR integer load (a tensor load -> not isInteger)
        for (unsigned i = 0; i < func.getNumArguments(); ++i) {
          mlir::Value a = func.getArgument(i);
          auto pt = mlir::dyn_cast<tt::PointerType>(a.getType());
          if (pt && isIndexInt(pt.getPointeeType()) && a != leafArg &&
              derivesFromValue(ld.getPtr(), a, 24))
            return ld;                                        // reads a distinct delimiter arg (kv_indptr)
        }
      }
      return tt::LoadOp();                                    // some other scalar load -> not the CSR start
    }
    for (mlir::Value o : op->getOperands())
      if (tt::LoadOp r = csrStartLoad(o, leafArg, func, depth - 1)) return r;
    return tt::LoadOp();
  }

  // CSR (token-pool) row addressing -- the SGLang analogue of vLLM's seq_idx*stride. The leaf index
  // load (kv_indices) is addressed by addptr(kv_indices, cur_seq_kv_start_idx + range). Returns
  // {seqIdx, startOffset}: seqIdx = cur_seq (the request, taken from the delimiter load's OWN address
  // addptr(kv_indptr, cur_seq), so it is axis-free); startOffset = cur_seq_kv_start_idx (the additive
  // base). Both null on a vLLM block table -> caller uses the multiplicative pagedRowFactors path. The
  // capture then writes buffer[seqIdx] = kv_indices[startOffset : startOffset+ncap] -- the SAME
  // buffer format as vLLM (request row -> its physical KV units), so the external reader is unchanged.
  std::pair<mlir::Value, mlir::Value> csrRow(tt::LoadOp leafLoad, mlir::Value leafArg, tt::FuncOp func) {
    if (!leafLoad) return {mlir::Value(), mlir::Value()};
    mlir::Value off = addptrOffsetOnBase(leafLoad.getPtr(), leafArg, 24);
    if (!off) return {mlir::Value(), mlir::Value()};
    tt::LoadOp start = csrStartLoad(off, leafArg, func, 24);
    if (!start) return {mlir::Value(), mlir::Value()};
    mlir::Value seqIdx;
    if (auto ap = mlir::dyn_cast_or_null<tt::AddPtrOp>(start.getPtr().getDefiningOp()))
      seqIdx = ap.getOffset();
    return {seqIdx, start.getResult()};
  }

  // STRUCTURAL num_seqs (name-independent): find_seq_idx is a binary search over cu_seqlens —
  // `right = num_seqs; while left < right { mid=...; load(query_start_len_ptr+mid); ... }`. So
  // num_seqs is the scalar func-arg INIT of the scf.while whose body loads the cu_seqlens arg
  // (the search bound). Distinctive (the only loop searching cu_seqlens). Returns the arg or null.
  Value findNumSeqsStructural(tt::FuncOp func, Value qslArg) {
    if (!qslArg) return Value();
    Value found;
    mlir::Block &entry = func.getBody().front();
    func.walk([&](mlir::scf::WhileOp w) {
      if (found) return;
      bool loadsQsl = false;
      w.walk([&](tt::LoadOp ld) {
        if (derivesFromValue(ld.getPtr(), qslArg, 24)) loadsQsl = true;
      });
      if (!loadsQsl) return;
      for (mlir::Value init : w.getInits()) {            // right = num_seqs (a scalar func arg)
        if (!init.getType().isInteger(32)) continue;
        if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(init))
          if (ba.getOwner() == &entry) { found = init; return; }
      }
    });
    return found;
  }

  // NOTE: the auto path's detection is now PURELY STRUCTURAL (the gather triad in instrumentAuto), so the
  // by-name finders that used to back it -- findIndexMapArg (name cross-check), kernelSelected (name gate),
  // findScalarArg (stride/num_seqs by name), findKvCacheArg (KV base by name), findQueryStartArg
  // (cu_seqlens_q by name) -- have been REMOVED, along with the silent EKV_ROW_STRIDE constant. A kernel's
  // index lookup is recovered from dataflow alone (index map -> KV gather -> contraction; row base from the
  // load's own addressing; num_seqs from the scf.while bound) or it is skipped loudly. EKV_*_NAMES are no
  // longer consulted. (The legacy attr path below is unaffected -- it takes explicit Python config, not
  // detection.)

  // Snapshot the cu_seqlens_q array (qslArg[0:count]) into the published i32 buffer ONCE — only
  // the first CTA (program_id(slotAxis)==0 [&& program_id(headAxis)==0]) writes — masked to the
  // valid length (num_seqs+1 at runtime when available, so no read past the array). One coalesced
  // masked load+store, like the block-table capture. The external reader computes per-seq
  // query_len = qsl[s+1]-qsl[s] -> the EXACT compute signal (recompute vs cache-hit), with no
  // need to reproduce the kernel's find_seq_idx (program_id != seq_idx here).
  void emitQslCopy(OpBuilder &b, Location loc, Value qslArg, uint64_t addr, int count,
                   int slotAxis, int headAxis, Value numSeqsVal) {
    if (addr == 0 || !qslArg || count <= 0) return;
    int W = 1;
    while (W < count) W <<= 1;
    Type i32 = b.getI32Type();
    Type i1 = b.getI1Type();
    Type i32Ptr = tt::getPointerType(i32);
    auto i32T = mlir::RankedTensorType::get({static_cast<int64_t>(W)}, i32);
    auto i1T = mlir::RankedTensorType::get({static_cast<int64_t>(W)}, i1);
    auto ptrT = mlir::RankedTensorType::get({static_cast<int64_t>(W)}, i32Ptr);
    auto toI32 = [&](Value v) {
      return v.getType() == i32 ? v : arith::TruncIOp::create(b, loc, i32, v).getResult();
    };
    Value range = tt::MakeRangeOp::create(b, loc, i32T, 0u, static_cast<uint32_t>(W));
    Value srcPtr = tt::AddPtrOp::create(b, loc, ptrT,
                                        tt::SplatOp::create(b, loc, ptrT, qslArg), range);
    Value dstInt = arith::ConstantIntOp::create(b, loc, static_cast<int64_t>(addr), 64);
    Value dstScalar = tt::IntToPtrOp::create(b, loc, i32Ptr, dstInt);
    Value dstPtr = tt::AddPtrOp::create(b, loc, ptrT,
                                        tt::SplatOp::create(b, loc, ptrT, dstScalar), range);
    // bound = num_seqs+1 (runtime arg) else the buffer count
    // bound = min(num_seqs+1, count): the qsl buffer holds `count` (= max_rows+1) entries, so even
    // with the runtime num_seqs we never write past it (num_seqs+1 > count at large batch -> OOB).
    Value boundC;
    if (numSeqsVal) {
      Value one = arith::ConstantIntOp::create(b, loc, 1, 32);
      boundC = arith::AddIOp::create(b, loc, toI32(numSeqsVal), one).getResult();
      boundC = arith::MinSIOp::create(b, loc, boundC,
                   arith::ConstantIntOp::create(b, loc, count, 32)).getResult();
    } else {
      boundC = arith::ConstantIntOp::create(b, loc, count, 32).getResult();
    }
    Value mask = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, range,
                                       tt::SplatOp::create(b, loc, i32T, boundC)).getResult();
    Value zero = arith::ConstantIntOp::create(b, loc, 0, 32);
    auto firstOnAxis = [&](int axis) {
      Value pid = tt::GetProgramIdOp::create(b, loc, axis);
      Value eq = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, pid, zero);
      mask = arith::AndIOp::create(b, loc, mask, tt::SplatOp::create(b, loc, i1T, eq)).getResult();
    };
    firstOnAxis(slotAxis);
    if (headAxis >= 0) firstOnAxis(headAxis);
    Value val = tt::LoadOp::create(b, loc, srcPtr, mask, tt::CacheModifier::NONE,
                                   tt::EvictionPolicy::NORMAL, /*isVolatile=*/false).getResult();
    tt::StoreOp::create(b, loc, dstPtr, val, /*mask=*/mask, tt::CacheModifier::NONE,
                        tt::EvictionPolicy::NORMAL, /*ignore_cta=*/true);
  }

  // Attr-based wrapper (the Python-installer path; kept for the validated tests).
  void captureBlockTable(tt::FuncOp func, Value slot, OpBuilder &b, Location loc) {
    auto btAttr = func->getAttrOfType<mlir::StringAttr>(attrs::kBtCtrlPtr);
    if (!btAttr) return;
    uint64_t idxAddr = std::strtoull(btAttr.getValue().str().c_str(), nullptr, 10);
    if (idxAddr == 0) return;
    int argIdx = getIntAttr(func, attrs::kBtArg, -1);
    int rowStride = getIntAttr(func, attrs::kBtRowStride, 0);
    int ncap = getIntAttr(func, attrs::kBtNcap, 0);
    int maxRows = getIntAttr(func, attrs::kBtMaxRows, 0);
    int headAxis = -1;
    if (auto ha = func->getAttrOfType<mlir::IntegerAttr>(attrs::kBtHeadAxis))
      headAxis = static_cast<int>(ha.getInt());
    emitCapture(func, slot, b, loc, argIdx, idxAddr, rowStride, ncap, maxRows, headAxis);
  }

  // The vectorized capture emission, parameterized (shared by the attr path and the
  // zero-touch auto path). headAxis < 0 -> no head dedup; maxRows <= 0 -> no row bounds check.
  void emitCapture(tt::FuncOp func, Value slot, OpBuilder &b, Location loc, int argIdx,
                   uint64_t idxAddr, int rowStride, int ncap, int maxRows, int headAxis,
                   Value strideVal = Value(), Value boundVal = Value(), Value csrRowOffset = Value(),
                   Value csrLen = Value()) {
    if (idxAddr == 0 || argIdx < 0 || argIdx >= static_cast<int>(func.getNumArguments()) || ncap <= 0)
      return;
    // The row base needs ONE source: csrRowOffset (CSR additive), strideVal (runtime stride, auto paged),
    // or the rowStride constant (the attr path's explicit config). The auto path passes rowStride=0.
    if (!csrRowOffset && !strideVal && rowStride <= 0)
      return;
    Value idxArg = func.getArgument(argIdx);
    if (!mlir::isa<tt::PointerType>(idxArg.getType())) return;  // must be a pointer arg
    // FALSE-MATCH CHECK: on a foreign kernel the detected slot/stride/CSR values can be TENSORS
    // (e.g. an inductor-fused embedding kernel whose gather triad mimics attention indexing). toI32's TruncIOp on
    // a tensor is invalid IR and kills the whole compile pipeline -- including in torch's spawned
    // compile workers where Python cannot catch it. Decline the emission instead.
    auto scalarInt = [](Value v) { return !v || v.getType().isIntOrIndex(); };
    if (!scalarInt(slot) || !scalarInt(strideVal) || !scalarInt(csrRowOffset) || !scalarInt(csrLen))
      return;

    int W = 1;
    while (W < ncap) W <<= 1;

    Type i32 = b.getI32Type();
    Type i1 = b.getI1Type();
    Type i32Ptr = tt::getPointerType(i32);
    auto i32T = mlir::RankedTensorType::get({static_cast<int64_t>(W)}, i32);
    auto i1T = mlir::RankedTensorType::get({static_cast<int64_t>(W)}, i1);
    auto ptrT = mlir::RankedTensorType::get({static_cast<int64_t>(W)}, i32Ptr);
    // The index array (block_table / kv_indices) is integer but its width VARIES: i32 (vLLM paged) or
    // i64 (SGLang token pool). Read it at its NATIVE width, then truncate to i32 for the buffer (a
    // physical slot/block id fits in i32). The dst buffer stays i32, so the external reader is unchanged.
    Type idxElem = mlir::cast<tt::PointerType>(idxArg.getType()).getPointeeType();
    auto idxPtrT = mlir::RankedTensorType::get({static_cast<int64_t>(W)}, tt::getPointerType(idxElem));

    auto toI32 = [&](Value v) {
      return v.getType() == i32 ? v : arith::TruncIOp::create(b, loc, i32, v).getResult();
    };
    slot = toI32(slot);   // seq_idx from the load's row offset is i64; the row math below is i32
    Value range = tt::MakeRangeOp::create(b, loc, i32T, 0u, static_cast<uint32_t>(W));
    // runtime stride (from the kernel's block_table_stride arg) when given, else const.
    Value strideC = strideVal ? toI32(strideVal)
                              : arith::ConstantIntOp::create(b, loc, rowStride, 32).getResult();
    Value ncapC = arith::ConstantIntOp::create(b, loc, ncap, 32);

    // src ptrs: idxArg + (row_base + range). row_base is the per-request offset into the index array:
    //   * paged/vLLM (multiplicative): slot*row_stride  -- block_table[seq_idx, :]
    //   * CSR/SGLang (additive): cur_seq_kv_start_idx    -- kv_indices[kv_indptr[seq] : ]
    // Both then snapshot ncap contiguous index entries; only the row base address differs.
    Value srcRow = csrRowOffset ? toI32(csrRowOffset)
                                : arith::MulIOp::create(b, loc, slot, strideC).getResult();
    Value srcOff = arith::AddIOp::create(
        b, loc, tt::SplatOp::create(b, loc, i32T, srcRow), range);
    Value srcBase = tt::SplatOp::create(b, loc, idxPtrT, idxArg);   // native index width (i32 or i64)
    Value srcPtr = tt::AddPtrOp::create(b, loc, idxPtrT, srcBase, srcOff);

    // dst ptrs: int_to_ptr(idxAddr) + (slot*ncap + range)
    Value dstRow = arith::MulIOp::create(b, loc, slot, ncapC);
    Value dstOff = arith::AddIOp::create(
        b, loc, tt::SplatOp::create(b, loc, i32T, dstRow), range);
    Value dstInt =
        arith::ConstantIntOp::create(b, loc, static_cast<int64_t>(idxAddr), 64);
    Value dstBaseScalar = tt::IntToPtrOp::create(b, loc, i32Ptr, dstInt);
    Value dstBase = tt::SplatOp::create(b, loc, ptrT, dstBaseScalar);
    Value dstPtr = tt::AddPtrOp::create(b, loc, ptrT, dstBase, dstOff);

    // mask = (range < ncap) [tail] AND (slot < max_rows) [row bounds] AND (head==0)
    // [dedup], each optional; null mask -> full unmasked vector.
    Value mask = Value();
    auto andMask = [&](Value m) {
      mask = mask ? arith::AndIOp::create(b, loc, mask, m).getResult() : m;
    };
    if (W != ncap) {  // tail lanes [ncap, W) never read/written
      Value ncapSplat = tt::SplatOp::create(b, loc, i32T, ncapC);
      andMask(arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, range, ncapSplat));
    }
    // CSR (ragged token pool): bound the read to THIS request's KV length (kv_indptr[r+1]-kv_indptr[r]).
    // Unlike vLLM's padded 2D block_table, kv_indices is ragged, so reading a fixed ncap past a short
    // request's slice runs off the array -> illegal access. This mask captures exactly the request's
    // physical slots (the rest stay -1) AND keeps every lane in-bounds. (Null for the paged path.)
    if (csrLen) {
      Value lenSplat = tt::SplatOp::create(b, loc, i32T, toI32(csrLen));
      andMask(arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, range, lenSplat));
    }
    // Paged 2D block_table: bound the snapshot to THIS row (the runtime stride). ncap wider than
    // the engine's row stride otherwise reads the NEXT rows' entries -- observed live as cross-row
    // bleed (a 30-block request reading as 120 = four rows' concatenation at ncap=512, stride=128),
    // which inflates n_blocks, deflates reuse_frac, and false-shares other rows' blocks. The stride
    // mask makes the snapshot row-exact for any ncap. (CSR is ragged: csrLen above bounds it.)
    if (!csrRowOffset) {
      Value strideSplat = tt::SplatOp::create(b, loc, i32T, strideC);
      andMask(arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, range, strideSplat));
    }
    // row bounds check: slot < bound: runtime num_seqs arg when given (exact, makes prefill's
    // slot=q_block reads safe), else the max_rows constant. This is what keeps the
    // capture in-bounds on every step including prefill/chunked.
    // slot < min(num_seqs, max_rows). num_seqs (structural) is the exact live row count, but the
    // buffer holds only max_rows rows -- so we ALSO cap by max_rows. Without the cap, num_seqs >
    // max_rows (large batch) would store PAST the host-mapped capture buffer (OOB). num_seqs absent
    // -> the bound is max_rows alone.
    Value bound;
    if (boundVal) {
      bound = toI32(boundVal);
      if (maxRows > 0)
        bound = arith::MinSIOp::create(b, loc, bound,
                    arith::ConstantIntOp::create(b, loc, maxRows, 32)).getResult();
    } else if (maxRows > 0) {
      bound = arith::ConstantIntOp::create(b, loc, maxRows, 32).getResult();
    }
    if (bound) {
      Value inRow = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, slot, bound);
      andMask(tt::SplatOp::create(b, loc, i1T, inRow));
    }
    if (headAxis >= 0) {  // head-dedup: only program_id(headAxis)==0 captures
      Value head = tt::GetProgramIdOp::create(b, loc, headAxis);
      Value zero = arith::ConstantIntOp::create(b, loc, 0, 32);
      Value headEq =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, head, zero);
      andMask(tt::SplatOp::create(b, loc, i1T, headEq));
    }

    // one coalesced load (at native index width) + store; null mask -> full-vector (no-mask builder).
    Value val = mask
                    ? tt::LoadOp::create(b, loc, srcPtr, mask, tt::CacheModifier::NONE,
                                         tt::EvictionPolicy::NORMAL, /*isVolatile=*/false)
                          .getResult()
                    : tt::LoadOp::create(b, loc, srcPtr, tt::CacheModifier::NONE,
                                         tt::EvictionPolicy::NORMAL, /*isVolatile=*/false)
                          .getResult();
    if (idxElem != i32)                          // i64 kv_indices -> i32 buffer (a physical slot id fits)
      val = arith::TruncIOp::create(b, loc, i32T, val).getResult();
    tt::StoreOp::create(b, loc, dstPtr, val, /*mask=*/mask, tt::CacheModifier::NONE,
                        tt::EvictionPolicy::NORMAL, /*ignore_cta=*/true);
  }

  // clock64 bracket: t0 at the current insert point, t1 at every return; atomic-add the
  // per-CTA cycle delta into ctrl[row] (u64). Shared by the attr path and the auto path.
  // The row is the kernel's OWN request index (seqIdx, when supplied and dominating the return), so
  // under a q-block/chunked-prefill grid every CTA of a sequence accumulates into that SEQUENCE's
  // row -- per-request cycles, not per-CTA; program_id(slotAxis) is the fallback where the kernel
  // has no computed request index (then CTA == request, e.g. the arithmetic-index closed form).
  // `gate` (optional i1, e.g. head-axis pid == 0) restricts the atomic to ONE representative head group:
  // the buffer is HOST-mapped, and a PCIe atomic is a serialized non-posted round trip, so per-CTA
  // adds across symmetric head CTAs multiply cost without adding information -- heads are symmetric,
  // shares are invariant to the constant head-group factor (observe/cycles contract).
  void emitTimer(tt::FuncOp func, Value slot, OpBuilder &b, Location loc, uint64_t addr, int maxRows,
                 Value seqIdx = Value(), Value gate = Value()) {
    if (addr == 0) return;
    Value t0 = readClock64(b, loc);
    Type i64 = b.getI64Type();
    Type i64Ptr = tt::getPointerType(i64);
    mlir::DominanceInfo dom(func);
    llvm::SmallVector<tt::ReturnOp> rets;
    func.walk([&](tt::ReturnOp r) { rets.push_back(r); });
    for (tt::ReturnOp ret : rets) {
      OpBuilder rb(ret);
      Location rloc = ret.getLoc();
      Value row = slot;
      if (seqIdx && seqIdx.getDefiningOp() &&
          dom.properlyDominates(seqIdx.getDefiningOp(), ret.getOperation()))
        row = seqIdx;
      if (!row.getType().isInteger(32)) {        // normalize for the addptr offset + the i32 row bound
        if (mlir::isa<mlir::IndexType>(row.getType()))
          row = arith::IndexCastOp::create(rb, rloc, rb.getI32Type(), row).getResult();
        else if (row.getType().getIntOrFloatBitWidth() > 32)
          row = arith::TruncIOp::create(rb, rloc, rb.getI32Type(), row).getResult();
        else
          row = arith::ExtSIOp::create(rb, rloc, rb.getI32Type(), row).getResult();
      }
      Value t1 = readClock64(rb, rloc);
      Value dur = arith::SubIOp::create(rb, rloc, t1, t0);
      Value baseInt = arith::ConstantIntOp::create(rb, rloc, static_cast<int64_t>(addr), 64);
      Value base = tt::IntToPtrOp::create(rb, rloc, i64Ptr, baseInt);
      Value elem = tt::AddPtrOp::create(rb, rloc, i64Ptr, base, row);
      // row bounds check: suppress the atomic when row >= max_rows, so a grid whose row index exceeds the
      // timer buffer (max_rows u64 slots) cannot OOB-write host memory. maxRows <= 0 -> no bounds check.
      Value mask;
      if (maxRows > 0)
        mask = arith::CmpIOp::create(rb, rloc, arith::CmpIPredicate::slt, row,
                   arith::ConstantIntOp::create(rb, rloc, maxRows, 32)).getResult();
      if (gate)
        mask = mask ? arith::AndIOp::create(rb, rloc, mask, gate).getResult() : gate;
      tt::AtomicRMWOp::create(rb, rloc, i64, tt::RMWOp::ADD, elem, dur,
                              /*mask=*/mask, tt::MemSemantic::RELAXED, tt::MemSyncScope::GPU);
    }
  }

  // === Locating the physical id and the logical position =====================================
  // The attention-share statistic is stored per logical block of each request, so the pass needs,
  // at the exponential of the softmax, the logical position of the tile and the scheduler row.
  //
  // The physical-id value at the gather makes this form-uniform: every recognized index form
  // (paged ranged, paged loop, CSR) must materialize the physical KV index -- the id that addresses KV --
  // immediately before its stride-scaling step. We locate it structurally: the idxLoad-derived
  // integer value consumed (through pure views) by a multiply whose other operand does NOT derive
  // from the index load (the stride), or used directly as an addptr offset.
  Value findIdValue(tt::FuncOp func, mlir::Operation *gather, Value idxRes) {
    Value gatherPtr = gather->getNumOperands() ? gather->getOperand(0) : Value();
    if (!gatherPtr) return idxRes;
    auto rankOf = [](Value v) {
      if (auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType()))
        return (int)t.getRank();
      return 0;
    };
    auto stripLocal = [](Value v) {           // stripView + expand_dims (a pure shape view here)
      while (v && v.getDefiningOp() &&
             mlir::isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp, arith::IndexCastOp,
                       tt::SplatOp, tt::BroadcastOp, tt::BitcastOp, tt::ExpandDimsOp>(
                 v.getDefiningOp()))
        v = v.getDefiningOp()->getOperand(0);
      return v;
    };
    Value best;
    int bestRank = 1 << 20;
    llvm::SmallVector<Value, 16> work{idxRes};
    llvm::SmallPtrSet<void *, 32> seen;
    int steps = 0;
    while (!work.empty() && steps++ < 512) {
      Value v = work.pop_back_val();
      if (!seen.insert(v.getAsOpaquePointer()).second)
        continue;
      for (mlir::OpOperand &use : v.getUses()) {
        mlir::Operation *op = use.getOwner();
        bool onPath = false;                  // this use must sit on the gather's address path
        for (Value r : op->getResults())
          if (r == gatherPtr || derivesFromValue(gatherPtr, r, 32)) { onPath = true; break; }
        if (!onPath)
          continue;
        bool isScale = false;
        if (auto mul = mlir::dyn_cast<arith::MulIOp>(op)) {
          Value other = (mul.getLhs() == v) ? mul.getRhs() : mul.getLhs();
          isScale = !derivesFromValue(other, idxRes, 32);
        } else if (mlir::isa<tt::AddPtrOp>(op) && use.getOperandNumber() == 1) {
          isScale = true;                     // direct pointer offset by the id
        }
        if (isScale) {
          Value cand = stripLocal(v);
          int r = rankOf(cand);
          if (r < bestRank) { bestRank = r; best = cand; }
        }
        if (mlir::isa<arith::MulIOp, arith::AddIOp, arith::SubIOp, arith::ExtSIOp, arith::ExtUIOp,
                      arith::TruncIOp, arith::IndexCastOp, arith::SelectOp, tt::SplatOp,
                      tt::BroadcastOp, tt::BitcastOp, tt::ExpandDimsOp, tt::ReduceOp>(op))
          for (Value r : op->getResults())
            work.push_back(r);
      }
    }
    return best ? best : idxRes;
  }

  // Normalize an integer scalar/tensor to 32-bit lanes (mask-table indices are i32).
  Value normI32(OpBuilder &b, Location loc, Value v) {
    Type t = v.getType();
    Type elem = t;
    if (auto rt = mlir::dyn_cast<mlir::RankedTensorType>(t))
      elem = rt.getElementType();
    if (elem.isInteger(32))
      return v;
    Type dst = b.getI32Type();
    if (auto rt = mlir::dyn_cast<mlir::RankedTensorType>(t))
      dst = mlir::RankedTensorType::get(rt.getShape(), b.getI32Type());
    if (mlir::isa<mlir::IndexType>(elem))
      return arith::IndexCastOp::create(b, loc, dst, v).getResult();
    if (elem.getIntOrFloatBitWidth() > 32)
      return arith::TruncIOp::create(b, loc, dst, v).getResult();
    return arith::ExtSIOp::create(b, loc, dst, v).getResult();
  }

  // keep(id, slot) = importance[id] >= tau[slot], built AT the given insertion point (each enforcement site
  // site rebuilds its own chain -- a few L2-cached loads -- so dominance holds by construction).
  // Out-of-table ids and out-of-row slots KEEP (conservative bounds).
  // === In-kernel minimum-keep invariant ======================================================
  // The GLOBAL position of each id within its request's row, recovered from the id load's own
  // pointer arithmetic: every additive term of the offset that does NOT derive from the request
  // index is position (the seq-derived part is the row base). On tiled kernels this correctly
  // yields tile_start + arange -- the global position -- not the per-tile arange. Position 0 is
  // the row's SINK block, present in EVERY causal window; OR-ing (pos == 0) into keep makes the
  // all-dropped -> NaN edge impossible in the instrumented kernel, regardless of host table state or
  // daemon race (the host-side floor/pins remain, but correctness no longer depends on them).
  Value findPosValue(Value id, Value seqIdx) {
    mlir::Operation *def = id.getDefiningOp();
    int depth = 0;
    while (def && !mlir::isa<tt::LoadOp>(def) && depth++ < 8) {
      if (def->getNumOperands() < 1)
        return {};
      def = def->getOperand(0).getDefiningOp();
    }
    if (!def || !mlir::isa<tt::LoadOp>(def))
      return {};
    Value ptr = def->getOperand(0);
    llvm::SmallVector<Value, 4> offs;
    int hops = 0;
    while (hops++ < 6) {
      mlir::Operation *pd = ptr.getDefiningOp();
      if (!pd)
        break;
      if (auto ap = mlir::dyn_cast<tt::AddPtrOp>(pd)) {
        offs.push_back(ap.getOffset());
        ptr = ap.getPtr();
        continue;
      }
      if (mlir::isa<tt::SplatOp, tt::BroadcastOp, tt::ExpandDimsOp>(pd)) {
        ptr = pd->getOperand(0);
        continue;
      }
      break;
    }
    llvm::SmallVector<Value, 4> parts, work(offs.begin(), offs.end());
    int steps = 0;
    while (!work.empty() && steps++ < 24) {
      Value v = work.pop_back_val();
      if (auto add = v.getDefiningOp<arith::AddIOp>()) {
        work.push_back(add.getLhs());
        work.push_back(add.getRhs());
        continue;
      }
      if (auto sp = v.getDefiningOp<tt::SplatOp>()) {      // decompose through splats
        work.push_back(sp.getOperand());
        continue;
      }
      if (seqIdx && derivesFromValue(v, seqIdx, 32))
        continue;                                          // the row base
      parts.push_back(v);
    }
    if (parts.empty())
      return {};
    // rebuild pos = sum(parts) at the ID's shape; scalars splat up
    auto idRt = mlir::dyn_cast<mlir::RankedTensorType>(id.getType());
    if (!idRt || idRt.getRank() != 1)
      return {};
    OpBuilder b(id.getContext());
    b.setInsertionPointAfterValue(id);
    Location loc = id.getLoc();
    auto posT = mlir::RankedTensorType::get(idRt.getShape(), b.getI32Type());
    Value pos;
    for (Value p : parts) {
      Value p32 = normI32(b, loc, p);
      if (!mlir::isa<mlir::RankedTensorType>(p32.getType()))
        p32 = tt::SplatOp::create(b, loc, posT, p32);
      if (p32.getType() != posT)
        return {};                                         // shape mismatch -> no floor
      pos = pos ? arith::AddIOp::create(b, loc, pos, p32).getResult() : p32;
    }
    return pos;
  }

  // A softmax numerator is consumed by the normalization reduction across the KV axis.
  // Online-softmax kernels also contain exp(old_max-new_max), which rescales running state but is
  // not indexed by KV position.  Merely asking whether an exponential derives from the index map
  // admits both forms on some unified-attention shapes.  Keep only exponentials whose result flows
  // through elementwise/view operations into a reduction.  Requiring only the output contraction
  // is insufficient because the running-state rescale eventually reaches the same output path.
  bool flowsToAttentionAggregate(Value v) {
    llvm::SmallVector<Value, 8> work{v};
    llvm::SmallPtrSet<void *, 32> seen;
    int steps = 0;
    while (!work.empty() && steps++ < 256) {
      Value x = work.pop_back_val();
      if (!seen.insert(x.getAsOpaquePointer()).second)
        continue;
      for (mlir::Operation *op : x.getUsers()) {
        if (mlir::isa<tt::ReduceOp>(op))
          return true;
        if (mlir::isa<tt::DotOp, tt::StoreOp, tt::LoadOp>(op))
          continue;
        llvm::StringRef dialect = op->getName().getDialectNamespace();
        if (dialect == "arith" || dialect == "math" ||
            op->getName().getStringRef().contains("extern_elementwise") ||
            mlir::isa<tt::SplatOp, tt::BroadcastOp, tt::ExpandDimsOp, tt::BitcastOp,
                      tt::ReshapeOp, tt::TransOp>(op))
          for (Value result : op->getResults())
            work.push_back(result);
      }
    }
    return false;
  }

  // === The MECHANIZED ADMISSION RULE (the effect-algebra verifier) ===========================
  // MODEL.md Sec 5 states the rule; this ENFORCES it: after each emission, the ops it inserted
  // are diffed and their memory effects classified against the emission's DECLARED class:
  //   PURE_READ              no stores, no atomics;
  //   MONOID                 atomic add/fadd only (order-free, CUDA-graph-replay-safe);
  //   IDEMPOTENT             plain stores / xchg only (last-writer-wins, replay-safe).
  // A violation is loud always and fatal under EKV_VERIFY_STRICT -- the same posture as an eBPF
  // verifier rejection: an emission whose effects exceed its declared algebra does not ship.
  // Every emission observes: the kernel's own memory and outputs are never written.
  enum class Effect { PureRead, Monoid, Idempotent };
  llvm::DenseSet<mlir::Operation *> opSnapshot(tt::FuncOp f) {
    llvm::DenseSet<mlir::Operation *> s;
    f.walk([&](mlir::Operation *op) { s.insert(op); });
    return s;
  }
  void verifyEmission(tt::FuncOp f, const llvm::DenseSet<mlir::Operation *> &before,
                      const char *name, Effect declared) {
    int stores = 0, adds = 0, xchgs = 0, otherAtomics = 0, loads = 0, total = 0;
    f.walk([&](mlir::Operation *op) {
      if (before.contains(op))
        return;
      ++total;
      if (mlir::isa<tt::StoreOp>(op)) {
        ++stores;
      } else if (auto rmw = mlir::dyn_cast<tt::AtomicRMWOp>(op)) {
        auto k = rmw.getAtomicRmwOp();
        if (k == tt::RMWOp::ADD || k == tt::RMWOp::FADD)
          ++adds;
        else if (k == tt::RMWOp::XCHG)
          ++xchgs;
        else
          ++otherAtomics;
      } else if (mlir::isa<tt::LoadOp>(op)) {
        ++loads;
      }
    });
    bool ok = false;
    const char *cls = "?";
    switch (declared) {
    case Effect::PureRead:
      cls = "pure-read"; ok = !stores && !adds && !xchgs && !otherAtomics; break;
    case Effect::Monoid:
      cls = "monoid"; ok = !stores && !xchgs && !otherAtomics; break;
    case Effect::Idempotent:
      cls = "idempotent"; ok = !adds && !otherAtomics; break;
    }
    if (!ok) {
      std::fprintf(stderr,
                   "[eKV-verify] VIOLATION: emission '%s' declared %s but inserted "
                   "stores=%d adds=%d xchgs=%d other_atomics=%d -- NOT ADMITTED\n",
                   name, cls, stores, adds, xchgs, otherAtomics);
      if (std::getenv("EKV_VERIFY_STRICT"))
        std::abort();
    } else if (std::getenv("EKV_DEBUG")) {
      std::fprintf(stderr,
                   "[eKV-verify] %s: %d ops (loads=%d stores=%d adds=%d xchgs=%d) "
                   "ADMITTED as %s\n", name, total, loads, stores, adds, xchgs, cls);
    }
  }

  // === Tile log-sum-exp statistic (EKV_ATTN_SHARE) ============================================
  // The kernel's online softmax already computes, for every tile of KV positions, the running max
  // M that it subtracts in exp(S - M) and the tile's sum l of those exponentials. The tile's
  // log attention weight is u = M + log(l), with no new reduction, and it does not depend on how
  // the kernel orders or splits its tiles. Normalizing u over all the tiles of a request gives the
  // exact attention share of each block, so a reader needs no knowledge of the running state, of
  // the number of KV segments, or of the head weighting.
  //
  // The kernel computes nothing for the statistic: it stores the two values it already holds, M in
  // plane 0 and l in plane 1 of the block, each [layer][request row][block][query row], and the
  // reader computes u (observe/tile_shares.py). No arithmetic in the kernel's loop also means no
  // effect on how the backend compiles the kernel's own arithmetic. Every block of a tile gets the
  // tile's pair; the reader splits the tile's weight over its blocks. The stores are idempotent:
  // one program writes each position once per launch, so a repeated write carries the same value.
  // Collection is sampled: the serving process sets a device flag on audit steps only, the flag
  // is the stores' mask, and layer L records KV group (L mod groups). The layer index comes from
  // the position of the kernel's KV pointer in a table of per-layer cache addresses that the
  // serving process fills.
  // ``newestRows`` is set when only the program that holds a request's newest query tokens records, so that a
  // prefill step, in which many programs serve one request, can be an audit step (see newestQueryGate).
  struct ShareResult { int sites = 0; int tileBlocks = 0; bool newestRows = false; };

  // A prefill launch gives one request to several programs, one per block of query tokens, and each would
  // store its own query rows into the same cells. The program that holds the request's newest query token is
  // the one whose positions reach the end of the request: the kernel bounds the keys a program reads with
  // min(candidate, sequence length), where the candidate is the position after its last query token. That
  // program is the one whose candidate is at least the sequence length. A decode launch has one program per
  // request, and its candidate always reaches the end. Returns that condition, built after the bound, or a
  // null value when the bound of the KV positions is not such a minimum of a computed position and a loaded
  // length.
  static Value newestQueryGate(Value positions) {
    for (mlir::Operation *user : positions.getUsers()) {
      auto cmp = mlir::dyn_cast<arith::CmpIOp>(user);
      if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt || cmp.getLhs() != positions)
        continue;
      auto bound = peelReplication(cmp.getRhs()).getDefiningOp<arith::MinSIOp>();
      if (!bound)
        continue;
      auto isLoaded = [](Value v) { return bool(peelReplication(v).getDefiningOp<tt::LoadOp>()); };
      Value a = bound.getLhs(), b = bound.getRhs();
      if (isLoaded(a) == isLoaded(b))
        continue;
      Value length = isLoaded(a) ? a : b, candidate = isLoaded(a) ? b : a;
      OpBuilder builder(bound->getContext());
      builder.setInsertionPointAfter(bound);
      return arith::CmpIOp::create(builder, bound.getLoc(), arith::CmpIPredicate::sge, candidate, length)
          .getResult();
    }
    return {};
  }

  // Peel broadcasts, dimension insertions, splats, and casts to the value they replicate.
  static Value peelReplication(Value v) {
    while (v) {
      mlir::Operation *op = v.getDefiningOp();
      if (op && mlir::isa<tt::BroadcastOp, tt::ExpandDimsOp, tt::SplatOp, arith::ExtSIOp,
                          arith::ExtUIOp, arith::TruncIOp, arith::IndexCastOp, tt::BitcastOp>(op))
        v = op->getOperand(0);
      else
        break;
    }
    return v;
  }

  // The integer value of a constant, whether it is a scalar, a splatted scalar, or a dense splat.
  static std::optional<int64_t> splatConstant(Value v) {
    auto constant = peelReplication(v).getDefiningOp<arith::ConstantOp>();
    if (!constant)
      return std::nullopt;
    if (auto scalar = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      return scalar.getValue().getSExtValue();
    auto dense = mlir::dyn_cast<mlir::DenseIntElementsAttr>(constant.getValue());
    if (dense && dense.isSplat())
      return dense.getSplatValue<llvm::APInt>().getSExtValue();
    return std::nullopt;
  }

  // The scalar result of reducing a vector over its only axis with `Combine` (min, max, or add).
  template <typename Combine>
  static Value reduceVector(OpBuilder &b, Location loc, Value vec) {
    auto rt = mlir::cast<mlir::RankedTensorType>(vec.getType());
    auto reduce = tt::ReduceOp::create(b, loc, mlir::ValueRange{vec}, /*axis=*/0);
    mlir::Block &body = reduce.getCombineOp().emplaceBlock();
    body.addArguments({rt.getElementType(), rt.getElementType()}, {loc, loc});
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&body);
    Value joined = Combine::create(b, loc, body.getArgument(0), body.getArgument(1));
    tt::ReduceReturnOp::create(b, loc, mlir::ValueRange{joined});
    return reduce.getResult().front();
  }

  ShareResult emitTileLse(tt::FuncOp func, int idxArg, Value idxRes, Value slotSem, Value kvArg,
                          int headAxis, uint64_t lseDev, uint64_t flagDev, uint64_t layerTableDev,
                          int layers, int rows, int width, int groupRows) {
    ShareResult result;
    // Every reason applies before the first operation that reads the kernel's values or stores, so a
    // declined kernel keeps only unused integer index arithmetic, which dead-code elimination removes.
    auto decline = [&](const char *why) {
      std::fprintf(stderr, "[eKV-share] statistic declined for %s: %s\n", func.getName().str().c_str(), why);
      return result;
    };
    if (!lseDev || !flagDev || !layerTableDev || !kvArg)
      return result;
    mlir::Operation *gather = kvGatherOp(func, idxArg);
    if (!gather)
      return decline("no KV gather");
    Value id = findIdValue(func, gather, idxRes);
    Value posGlobal = findPosValue(id, slotSem);
    auto posType = posGlobal ? mlir::dyn_cast<mlir::RankedTensorType>(posGlobal.getType())
                             : mlir::RankedTensorType();
    if (!posType || posType.getRank() != 1)
      return decline("logical position not recovered");
    // Positions per block: the position vector is offset / blockSize when blocks are larger than
    // one position, and the offset itself when it counts positions (token-granular pools).
    int64_t tile = posType.getShape()[0], blockSize = 1;
    // A kernel that indexes the cache by token slot has no block of its own. EKV_TOKENS_PER_BLOCK groups that many
    // consecutive positions into one recorded block, the block that the serving process's view keeps or drops.
    bool tokenBlocks = false;
    if (auto div = posGlobal.getDefiningOp<arith::DivSIOp>()) {
      std::optional<int64_t> divisor = splatConstant(div.getRhs());
      if (!divisor) {
        if (std::getenv("EKV_DEBUG")) {
          mlir::Operation *def = peelReplication(div.getRhs()).getDefiningOp();
          llvm::errs() << "[eKV-share] divisor: ";
          if (def)
            def->print(llvm::errs());
          llvm::errs() << "\n[eKV-share] division: ";
          div->print(llvm::errs());
          llvm::errs() << "\n";
        }
        return decline("block size is not a constant");
      }
      blockSize = *divisor;
    } else if (int tokens = getEnvInt("EKV_TOKENS_PER_BLOCK", 1); tokens > 1) {
      blockSize = tokens;
      tokenBlocks = true;
    }
    if (blockSize <= 0 || blockSize > tile || tile % blockSize != 0 ||
        !llvm::isPowerOf2_64((uint64_t)(tile / blockSize)))
      return decline("a tile does not cover a power-of-two number of whole blocks");
    const int64_t perTile = tile / blockSize;
    // The KV positions of the tile, before they are divided into blocks.
    Value positions = posGlobal;
    if (auto div = posGlobal.getDefiningOp<arith::DivSIOp>())
      positions = div.getLhs();
    // Exponentials of the form exp(S - M) whose operand derives from the KV-index lookup.
    llvm::SmallVector<mlir::Operation *, 2> exps;
    func.walk([&](mlir::Operation *op) {
      if (!mlir::isa<mlir::math::ExpOp>(op) || op->getNumResults() != 1)
        return;
      if (!flowsToAttentionAggregate(op->getResult(0)))
        return;                                        // running-state rescale, not KV numerator
      for (Value o : op->getOperands())
        if (derivesFromValue(o, idxRes, 48)) { exps.push_back(op); return; }
    });
    mlir::DominanceInfo dom(func);
    auto domOk = [&](mlir::Operation *site, Value v) {
      if (!v)
        return false;
      if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(v))
        return dom.dominates(ba.getParentBlock(), site->getBlock());
      return dom.properlyDominates(v.getDefiningOp(), site);
    };
    // Check every site before inserting anything, so that a declined kernel gets no partial statistic.
    struct Site { tt::ReduceOp tileSum; Value m, l; };
    llvm::SmallVector<Site, 2> found;
    for (mlir::Operation *ex : exps) {
      auto sub = ex->getOperand(0).getDefiningOp<arith::SubFOp>();   // exp(S - M)
      if (!sub)
        return decline("an exponential is not of the form exp(S - M)");
      Value p = ex->getResult(0);
      tt::ReduceOp tileSum;                            // the kernel's own sum of p over the tile
      for (mlir::Operation *user : p.getUsers())
        if (auto red = mlir::dyn_cast<tt::ReduceOp>(user)) {
          mlir::Block &body = red.getCombineOp().front();
          if (body.getOperations().size() == 2 && mlir::isa<arith::AddFOp>(body.front())) {
            tileSum = red;
            break;
          }
        }
      if (!tileSum)
        return decline("no tile sum of the exponentials");
      Value l = tileSum->getResult(0);
      Value m = peelReplication(sub.getRhs());           // the running max, before broadcast
      if (m.getType() != l.getType())
        return decline("running max and tile sum differ in shape");
      auto lt = mlir::dyn_cast<mlir::RankedTensorType>(l.getType());
      Type elem = lt ? lt.getElementType() : l.getType();
      if (!elem.isF32() || (lt && lt.getRank() != 1))
        return decline("tile sum is not a f32 scalar or vector");
      if (!domOk(tileSum, posGlobal) || !domOk(tileSum, slotSem) || !domOk(tileSum, m))
        return decline("operands do not dominate the tile sum");
      // A new use of a product can stop the backend from fusing that product into a later add (one
      // rounding instead of two), which changes the kernel's result. The statistic reads only the
      // results of the kernel's reductions, never a product.
      if (m.getDefiningOp<arith::MulFOp>() || l.getDefiningOp<arith::MulFOp>())
        return decline("running max or tile sum is a product: a new use could change its fusion");
      found.push_back({tileSum, m, l});
    }
    if (found.empty()) {
      std::fprintf(stderr, "[eKV-share] statistic not recorded for %s: no softmax exponential matched\n",
                   func.getName().str().c_str());
      return result;
    }
    Value newest = newestQueryGate(positions);
    for (const Site &site : found)
      if (newest && !domOk(site.tileSum, newest)) {
        newest.getDefiningOp()->erase();
        newest = {};
      }
    // Per-launch scalars, built once at entry: the layer of this launch and the collection gate.
    mlir::Block &entry = func.front();
    OpBuilder eb(&entry, entry.begin());
    Location eloc = func.getLoc();
    Type i32 = eb.getI32Type(), i64 = eb.getI64Type(), f32 = eb.getF32Type();
    const uint32_t tableLen = llvm::PowerOf2Ceil((uint32_t)layers);
    auto rangeT = mlir::RankedTensorType::get({(int64_t)tableLen}, i32);
    Value slots = tt::MakeRangeOp::create(eb, eloc, rangeT, 0u, tableLen);
    auto tablePtrT = mlir::RankedTensorType::get({(int64_t)tableLen}, tt::getPointerType(i64));
    Value tableBase = tt::SplatOp::create(eb, eloc, tablePtrT,
        tt::IntToPtrOp::create(eb, eloc, tt::getPointerType(i64),
            arith::ConstantIntOp::create(eb, eloc, (int64_t)layerTableDev, 64)));
    Value tablePtrs = tt::AddPtrOp::create(eb, eloc, tablePtrT, tableBase, slots);
    Value inTable = arith::CmpIOp::create(eb, eloc, arith::CmpIPredicate::slt, slots,
        tt::SplatOp::create(eb, eloc, rangeT,
            arith::ConstantIntOp::create(eb, eloc, layers, 32))).getResult();
    Value cached = tt::LoadOp::create(eb, eloc, tablePtrs, inTable, tt::CacheModifier::NONE,
                                      tt::EvictionPolicy::NORMAL, false).getResult();
    Value isThis = arith::CmpIOp::create(eb, eloc, arith::CmpIPredicate::eq, cached,
        tt::SplatOp::create(eb, eloc,
            mlir::RankedTensorType::get({(int64_t)tableLen}, i64),
            tt::PtrToIntOp::create(eb, eloc, i64, kvArg))).getResult();
    Value one = arith::ConstantIntOp::create(eb, eloc, 1, 32);
    Value plusOne = arith::AddIOp::create(eb, eloc, slots, tt::SplatOp::create(eb, eloc, rangeT, one));
    Value zeroV = tt::SplatOp::create(eb, eloc, rangeT, arith::ConstantIntOp::create(eb, eloc, 0, 32));
    Value layer = arith::SubIOp::create(eb, eloc,
        reduceVector<arith::MaxSIOp>(eb, eloc,
            arith::SelectOp::create(eb, eloc, isThis, plusOne, zeroV).getResult()), one).getResult();
    Value zero = arith::ConstantIntOp::create(eb, eloc, 0, 32);
    Value flagPtr = tt::IntToPtrOp::create(eb, eloc, tt::getPointerType(i32),
        arith::ConstantIntOp::create(eb, eloc, (int64_t)flagDev, 64));
    Value flag = tt::LoadOp::create(eb, eloc, flagPtr, tt::CacheModifier::NONE,
                                    tt::EvictionPolicy::NORMAL, false).getResult();
    Value gate = arith::AndIOp::create(eb, eloc,
        arith::CmpIOp::create(eb, eloc, arith::CmpIPredicate::ne, flag, zero),
        arith::CmpIOp::create(eb, eloc, arith::CmpIPredicate::sge, layer, zero)).getResult();
    if (headAxis >= 0) {                               // layer L records KV group L mod groups
      Value group = arith::RemSIOp::create(eb, eloc, layer, tt::GetNumProgramsOp::create(eb, eloc, headAxis));
      gate = arith::AndIOp::create(eb, eloc, gate,
          arith::CmpIOp::create(eb, eloc, arith::CmpIPredicate::eq,
              tt::GetProgramIdOp::create(eb, eloc, headAxis), group)).getResult();
    }
    int sites = 0;
    for (const Site &site : found) {
      tt::ReduceOp tileSum = site.tileSum;
      Value m = site.m, l = site.l;
      auto lt = mlir::dyn_cast<mlir::RankedTensorType>(l.getType());
      OpBuilder b(tileSum->getContext());
      b.setInsertionPointAfter(tileSum);
      Location loc = tileSum->getLoc();
      Value pos0 = reduceVector<arith::MinSIOp>(b, loc, normI32(b, loc, posGlobal));
      if (tokenBlocks)                                 // the first position of the tile, as a block index
        pos0 = arith::DivSIOp::create(b, loc, pos0, arith::ConstantIntOp::create(b, loc, blockSize, 32)).getResult();
      Value slot32 = normI32(b, loc, slotSem);
      auto cst = [&](int64_t v) { return arith::ConstantIntOp::create(b, loc, v, 32).getResult(); };
      Value rowIndex = arith::AddIOp::create(b, loc,
          arith::MulIOp::create(b, loc, layer, cst(rows)), slot32).getResult();
      Value base = arith::MulIOp::create(b, loc,
          arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, rowIndex, cst(width)), pos0),
          cst(groupRows)).getResult();
      Value scalarOk = arith::AndIOp::create(b, loc, gate,
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, slot32, cst(rows))).getResult();
      if (newest)                                        // only the program with the newest query tokens
        scalarOk = arith::AndIOp::create(b, loc, scalarOk, newest).getResult();
      Type f32Ptr = tt::getPointerType(f32);
      auto planeBase = [&](int plane) {                  // plane 0 holds M, plane 1 holds l
        uint64_t cells = (uint64_t)layers * rows * width * groupRows;
        return tt::IntToPtrOp::create(b, loc, f32Ptr,
            arith::ConstantIntOp::create(b, loc, (int64_t)(lseDev + plane * cells * 4u), 64)).getResult();
      };
      auto blockRange = [&](int64_t n) {
        return tt::MakeRangeOp::create(b, loc, mlir::RankedTensorType::get({n}, i32), 0u, (uint32_t)n)
            .getResult();
      };
      Value blocks = blockRange(perTile);
      Value blockIdx = arith::AddIOp::create(b, loc, blocks,
          tt::SplatOp::create(b, loc, mlir::cast<mlir::RankedTensorType>(blocks.getType()), pos0));
      Value inWidth = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, blockIdx,
          tt::SplatOp::create(b, loc, mlir::cast<mlir::RankedTensorType>(blocks.getType()),
                              cst(width))).getResult();
      Value offsets = arith::MulIOp::create(b, loc, blocks,
          tt::SplatOp::create(b, loc, mlir::cast<mlir::RankedTensorType>(blocks.getType()),
                              cst(groupRows))).getResult();
      Value index, mask;
      mlir::RankedTensorType cellsT, ptrsT;              // the cells one tile fills, and their pointers
      if (lt) {                                          // [blocks in tile, query rows]
        int64_t bm = lt.getShape()[0];
        auto gridI32 = mlir::RankedTensorType::get({perTile, bm}, i32);
        auto gridI1 = mlir::RankedTensorType::get({perTile, bm}, b.getI1Type());
        cellsT = mlir::RankedTensorType::get({perTile, bm}, f32);
        ptrsT = mlir::RankedTensorType::get({perTile, bm}, f32Ptr);
        Value rowsV = blockRange(bm);
        Value rowOk = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, rowsV,
            tt::SplatOp::create(b, loc, mlir::cast<mlir::RankedTensorType>(rowsV.getType()),
                                cst(groupRows))).getResult();
        index = arith::AddIOp::create(b, loc,
            tt::BroadcastOp::create(b, loc, gridI32, tt::ExpandDimsOp::create(b, loc, offsets, 1)),
            tt::BroadcastOp::create(b, loc, gridI32, tt::ExpandDimsOp::create(b, loc, rowsV, 0))).getResult();
        index = arith::AddIOp::create(b, loc, index, tt::SplatOp::create(b, loc, gridI32, base)).getResult();
        mask = arith::AndIOp::create(b, loc,
            tt::BroadcastOp::create(b, loc, gridI1, tt::ExpandDimsOp::create(b, loc, inWidth, 1)),
            tt::BroadcastOp::create(b, loc, gridI1, tt::ExpandDimsOp::create(b, loc, rowOk, 0))).getResult();
        mask = arith::AndIOp::create(b, loc, mask, tt::SplatOp::create(b, loc, gridI1, scalarOk)).getResult();
      } else {                                           // one query row per program
        auto lineI32 = mlir::RankedTensorType::get({perTile}, i32);
        auto lineI1 = mlir::RankedTensorType::get({perTile}, b.getI1Type());
        cellsT = mlir::RankedTensorType::get({perTile}, f32);
        ptrsT = mlir::RankedTensorType::get({perTile}, f32Ptr);
        index = arith::AddIOp::create(b, loc, offsets, tt::SplatOp::create(b, loc, lineI32, base)).getResult();
        mask = arith::AndIOp::create(b, loc, inWidth, tt::SplatOp::create(b, loc, lineI1, scalarOk)).getResult();
      }
      for (auto [plane, v] : {std::pair<int, Value>{0, m}, std::pair<int, Value>{1, l}}) {
        Value cells = lt ? tt::BroadcastOp::create(b, loc, cellsT, tt::ExpandDimsOp::create(b, loc, v, 0)).getResult()
                         : tt::SplatOp::create(b, loc, cellsT, v).getResult();
        Value ptrs = tt::AddPtrOp::create(b, loc, ptrsT, tt::SplatOp::create(b, loc, ptrsT, planeBase(plane)), index);
        tt::StoreOp::create(b, loc, ptrs, cells, mask, tt::CacheModifier::NONE,
                            tt::EvictionPolicy::NORMAL, /*ignore_cta=*/true);
      }
      ++sites;
    }
    if (std::getenv("EKV_DEBUG"))
      std::fprintf(stderr, "[eKV-share] %d exponential site(s) record tile log-sum-exp\n", sites);
    result.sites = sites;
    result.tileBlocks = (int)perTile;
    result.newestRows = bool(newest);
    return result;
  }

  // Publish the kernel's KV-cache base POINTER: store ptrtoint(kvArg) into the
  // self-allocated kv_ptr buffer at entry. Every CTA writes the same value (idempotent),
  // via an atomic exchange -- the proven device-write path (mirrors emitTimer's atomic).
  // The external reader reads this u64 and identifies the KV pool as the captured
  // allocation that CONTAINS it: exact pool identification, no size heuristic, any
  // workload. (The attention kernel must be handed a pointer to its KV cache to gather
  // KV -- the same class of kernel input as the block table.)
  // `gate` (optional i1, first-CTA predicate) restricts the idempotent XCHG to ONE writer: the
  // value is identical from every CTA, and each ungated write is a serialized non-posted PCIe
  // atomic to the same host cacheline -- pure cost, no information.
  void emitKvPtr(OpBuilder &b, Location loc, Value kvArg, uint64_t addr, Value gate = Value()) {
    if (addr == 0 || !kvArg) return;
    Type i64 = b.getI64Type();
    Type i64Ptr = tt::getPointerType(i64);
    Value kvInt = tt::PtrToIntOp::create(b, loc, i64, kvArg);
    Value baseInt = arith::ConstantIntOp::create(b, loc, static_cast<int64_t>(addr), 64);
    Value base = tt::IntToPtrOp::create(b, loc, i64Ptr, baseInt);
    tt::AtomicRMWOp::create(b, loc, i64, tt::RMWOp::XCHG, base, kvInt,
                            /*mask=*/gate, tt::MemSemantic::RELAXED, tt::MemSyncScope::GPU);
  }

  // Publish a runtime SCALAR (the arithmetic stride) to a device buffer at entry -- the same proven
  // device-write as emitKvPtr (atomic xchg, every CTA writes the same value). The reader reads this
  // u64 alongside kv_ptr and computes region(seq) = kv_base + seq*stride. Integer strides of any
  // width are sign-extended (or index-cast) to i64.
  void emitScalarPub(OpBuilder &b, Location loc, Value val, uint64_t addr, Value gate = Value()) {
    if (addr == 0 || !val) return;
    Type i64 = b.getI64Type();
    Type i64Ptr = tt::getPointerType(i64);
    Value v64 = val;
    if (val.getType() != i64) {
      if (mlir::isa<mlir::IndexType>(val.getType()))
        v64 = arith::IndexCastOp::create(b, loc, i64, val).getResult();
      else
        v64 = arith::ExtSIOp::create(b, loc, i64, val).getResult();
    }
    Value baseInt = arith::ConstantIntOp::create(b, loc, static_cast<int64_t>(addr), 64);
    Value base = tt::IntToPtrOp::create(b, loc, i64Ptr, baseInt);
    tt::AtomicRMWOp::create(b, loc, i64, tt::RMWOp::XCHG, base, v64,
                            /*mask=*/gate, tt::MemSemantic::RELAXED, tt::MemSyncScope::GPU);
  }

  void emitCoverageLaunch(OpBuilder &b, Location loc, uint64_t addr, Value firstCta) {
    if (!addr)
      return;
    Type i64 = b.getI64Type();
    Value ptr = tt::IntToPtrOp::create(b, loc, tt::getPointerType(i64),
        arith::ConstantIntOp::create(b, loc, (int64_t)addr, 64));
    tt::AtomicRMWOp::create(b, loc, i64, tt::RMWOp::ADD, ptr,
        arith::ConstantIntOp::create(b, loc, 1, 64), firstCta,
        tt::MemSemantic::RELAXED, tt::MemSyncScope::GPU);
  }

  // Arithmetic-index fallback (closes the L1 index-only boundary): a non-paged/contiguous attention
  // whose KV address is pure arithmetic (no index array). The request->region map is the closed form
  // kv_base + program_id(slotAxis)*stride; we recover + publish those two scalars (kv_ptr + stride)
  // and the reader applies the formula -- no per-request capture array (there is no table to read).
  // Called from instrumentAuto only when the index lookup is absent, so it never mis-binds an
  // index-based kernel. Skips loudly (EKV_DEBUG) when neither supported address form is present.
  void instrumentArith(tt::FuncOp func) {
    int slotAxis = getEnvInt("EKV_SLOT_AXIS", 0);
    ArithmeticIndex ak = findArithmeticIndex(func, slotAxis);
    if (!ak.kvBase || !ak.stride) {
      if (std::getenv("EKV_DEBUG"))
        std::fprintf(stderr, ak.sawIndexGather
            ? "[eKV-auto] %s: an index-based KV gather is present but its index lookup is incomplete "
              "-> skip LOUDLY (a detection gap; NOT silently re-bound as arithmetic)\n"
            : "[eKV-auto] %s: no index lookup and no arithmetic KV-address form -> skip\n",
                     func.getName().str().c_str());
      publishCoverage(func.getName().str().c_str(), "untouched",
                      ak.sawIndexGather ? "incomplete_index_lookup" : "no_attention_address_form");
      return;
    }
    if (std::getenv("EKV_DETECT_ONLY")) {
      int kvIdx = -1;
      for (unsigned i = 0; i < func.getNumArguments(); ++i)
        if (func.getArgument(i) == ak.kvBase) { kvIdx = (int)i; break; }
      std::fprintf(stderr, "[eKV-detect] kernel=%s idxArg=-1(arith) kvArg=%d(%s) row=arith\n",
                   func.getName().str().c_str(), kvIdx, argName(ak.kvBase).c_str());
      publishCoverage(func.getName().str().c_str(), "observation_only", "arithmetic_detect_only");
      return;
    }
    int maxRows = getEnvInt("EKV_MAX_ROWS", 1024);
    // The arith timer is OPT-IN (EKV_TIMER_ARITH): in a torch.compile'd engine, foreign inductor
    // kernels (layer norms, top-k, reductions) can match the arithmetic address form, and a
    // per-CTA timer on their large grids is a PCIe atomic storm with no attribution value. The
    // paged/attention timer (EKV_TIMER) is unaffected; enable this only for a genuinely
    // arithmetic-index ATTENTION kernel.
    bool wantTimer = std::getenv("EKV_TIMER_ARITH") != nullptr;
    ArithPub ap2 = selfAllocArith(wantTimer ? (size_t)maxRows * 8u : 0, maxRows,
                                  func.getName().str().c_str());
    if (ap2.kvPtrDev == 0) {
      publishCoverage(func.getName().str().c_str(), "recognized_declined",
                      "arithmetic_state_allocation_failed");
      return;                                        // CUDA alloc failed -> don't touch the kernel
    }
    if (std::getenv("EKV_DEBUG"))
      std::fprintf(stderr, "[eKV-auto] kernel=%s ARITHMETIC INDEX: kv=%s row=arith "
                   "(region=kv_base+seq*stride)\n", func.getName().str().c_str(),
                   argName(ak.kvBase).c_str());
    mlir::Block &entry = func.front();
    OpBuilder b(&entry, entry.begin());
    Location loc = func.getLoc();
    // One writer for the idempotent publications (each ungated write is a serialized non-posted
    // PCIe atomic): CTA (0[,0]) on the slot (and head, if any) axes. Built at entry, so it
    // dominates the anchored emission points below.
    int headAxis = getEnvInt("EKV_HEAD_AXIS", 1);
    Value zero32 = arith::ConstantIntOp::create(b, loc, 0, 32);
    Value slot0 = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                      tt::GetProgramIdOp::create(b, loc, slotAxis), zero32).getResult();
    Value headGate;
    if (headAxis >= 0)
      headGate = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                     tt::GetProgramIdOp::create(b, loc, headAxis), zero32).getResult();
    Value firstCta = headGate ? arith::AndIOp::create(b, loc, slot0, headGate).getResult() : slot0;
    // Publish each scalar AFTER its own def: a matching kernel (e.g. an inductor GEMV) can compute
    // kvBase or the stride mid-body, and publishing at entry would use the value before it is
    // defined -- a verifier error that breaks the FOREIGN kernel's compile. The pass must never
    // fail a compile it merely observes. An ARG value (no def) anchors after the gate ops instead
    // (never at the block start, which would precede the gate's own definition).
    {
      OpBuilder pb(func.getContext());
      mlir::Operation *d = ak.kvBase.getDefiningOp();
      pb.setInsertionPointAfter(d ? d : firstCta.getDefiningOp());
      emitKvPtr(pb, loc, ak.kvBase, ap2.kvPtrDev, firstCta);   // the base K
    }
    {
      OpBuilder pb(func.getContext());
      mlir::Operation *d = ak.stride.getDefiningOp();
      pb.setInsertionPointAfter(d ? d : firstCta.getDefiningOp());
      emitScalarPub(pb, loc, ak.stride, ap2.strideDev, firstCta);   // the per-request stride
    }
    if (wantTimer) {
      Value slot = tt::GetProgramIdOp::create(b, loc, slotAxis);
      emitTimer(func, slot, b, loc, ap2.timerDev, maxRows, Value(), headGate);
    }
    func->setAttr(attrs::kInstrumented, mlir::UnitAttr::get(func.getContext()));
    publishCoverage(func.getName().str().c_str(), "observation_only", "arithmetic_mapping_observed");
  }

  // Zero-touch path (the DEFAULT; EKV_LEGACY selects the oracle instead): ONE structural
  // characterization of the address map index lookup over the
  // TTIR dataflow -- no Python attrs, no kernel/arg NAMES, no guessed constants. The GATHER TRIAD is
  // necessary and sufficient for gather-based attention (GENERALIZATION.md):
  //   (1) INDEX MAP I -- an integer ptr arg whose VECTOR load's values DIRECTLY address ...
  //   (2) KV BASE  K  -- ... a VALUE gather through a SEPARATE value ptr, and
  //   (3) CONTRACTION -- that gathered value feeds tt.dot (GEMM) or tt.reduce (GEMV): attention's <q,k>.
  // Everything else is DERIVED from the triad: K is the gather's value ptr; the row base is I's load
  // offset in one of two affine row-address forms -- seqIdx*stride (paged: vLLM block_table, B=16) or
  // kv_indptr[seq] (CSR: SGLang kv_indices, B=1); the row bound is the structural num_seqs, else the
  // safe buffer size. If the triad or the row-base form is absent we return (skip LOUDLY via EKV_DEBUG)
  // -- never a name or a guessed stride, which can silently mis-bind. The capture host address is
  // published to /tmp/ekv_pass_<pid>.json for the external reader.
  void instrumentAuto(tt::FuncOp func) {
    if (func->hasAttr(attrs::kInstrumented)) return;
    if (func.getBody().empty()) return;

    // --- index lookup = the data-dependent reindexing (the one IR-level principle; see findIndexLookup) ---
    IndexLookup ks = findIndexLookup(func);
    // TEST-ONLY (EKV_FORCE_ARITH): run the arithmetic path on ANY kernel to PROVE its no-index-load
    // structural validation rejects an index-based kernel (loud skip) rather than mis-binding it. Off by default.
    if (std::getenv("EKV_FORCE_ARITH")) { instrumentArith(func); return; }
    if (ks.idxArg < 0) {
      // A pure arithmetic address form is common in layer normalization, top-k, and other
      // non-attention kernels.  It is therefore not a safe default attention classifier.  A
      // backend-specific coverage experiment may opt in after independently identifying an
      // arithmetic-attention kernel; paged/CSR serving uses the loaded index map below.
      const char *arith = std::getenv("EKV_ARITH_ATTENTION");
      bool enabled = arith && *arith && std::strcmp(arith, "0") != 0 &&
                     std::strcmp(arith, "false") != 0;
      if (enabled)
        instrumentArith(func);
      else
        publishCoverage(func.getName().str().c_str(), "untouched",
                        "no_loaded_kv_index_map");
      return;
    }
    int idxArg = ks.idxArg;
    Value kvArg = ks.kvBase;
    tt::LoadOp idxLoad = ks.idxLoad;
    if (std::getenv("EKV_DUMP_IR")) {
      std::fprintf(stderr, "=== EKV_TTIR_BEGIN %s ===\n", func.getName().str().c_str());
      func.print(llvm::errs());
      std::fprintf(stderr, "\n=== EKV_IDXLOAD (the index-map load) ===\n");
      idxLoad.print(llvm::errs());
      std::fprintf(stderr, "\n=== EKV_TTIR_END ===\n");
    }

    // --- row base: the index-load offset in either paged or CSR form ---
    auto csr   = csrRow(idxLoad, func.getArgument(idxArg), func);          // additive: kv_indptr[seq]  (B=1)
    auto paged = pagedRowFactors(idxLoad, func.getArgument(idxArg));       // multiplicative: seqIdx*stride (B>=1)
    Value seqIdx = paged.first, strideVal = paged.second;
    if (!strideVal) strideVal = strideFromChain(idxLoad.getPtr(), 24);     // stride from the same addressing
    bool isCsr   = csr.first && csr.second && csr.second.getDefiningOp();
    bool isPaged = seqIdx && seqIdx.getDefiningOp() && strideVal;
    if (!isCsr && !isPaged) {                         // addressing not recognized -> skip (never guess)
      if (std::getenv("EKV_DEBUG"))
        std::fprintf(stderr, "[eKV-auto] %s: index map found but row base unrecognized -> skip\n",
                     func.getName().str().c_str());
      publishCoverage(func.getName().str().c_str(), "recognized_declined",
                      "index_map_row_base_unrecognized");
      return;
    }

    // EKV_DETECT_ONLY: report the detected index lookup and stop before CUDA allocation and
    // emission. Lets the detection be observed on parsed/compiled TTIR with no GPU context -- used to
    // validate the descriptor (TMA) op-set on hardware that cannot lower it (the box is an A6000, not Hopper).
    if (std::getenv("EKV_DETECT_ONLY")) {
      int kvIdx = -1;
      for (unsigned i = 0; i < func.getNumArguments(); ++i)
        if (func.getArgument(i) == kvArg) { kvIdx = (int)i; break; }
      mlir::Operation *g = kvGatherOp(func, idxArg);
      std::fprintf(stderr, "[eKV-detect] kernel=%s idxArg=%d(%s) kvArg=%d(%s) row=%s gather=%s\n",
                   func.getName().str().c_str(), idxArg, argName(func.getArgument(idxArg)).c_str(),
                   kvIdx, argName(kvArg).c_str(), isCsr ? "CSR" : "paged",
                   g ? g->getName().getStringRef().str().c_str() : "?");
      publishCoverage(func.getName().str().c_str(), "observation_only", "detect_only");
      return;
    }

    // structural signals (no name fallback): num_seqs (the scf.while search bound) tightens the row
    // bound, else the safe default is the buffer size; cu_seqlens_q is the query-length signal (EKV_QLEN).
    Value qslArg = findQueryStartStructural(func);
    Value boundVal = findNumSeqsStructural(func, qslArg);

    int slotAxis = getEnvInt("EKV_SLOT_AXIS", 0);
    int headAxis = getEnvInt("EKV_HEAD_AXIS", 1);   // -1 disables head dedup
    int ncap = getEnvInt("EKV_NCAP", 128);          // capture width: units captured per request
    int maxRows = getEnvInt("EKV_MAX_ROWS", 1024);  // buffer rows = safe row bound
    auto envFlag = [](const char *k) {                 // value-aware: "0"/"false" disables.
      const char *v = std::getenv(k);                  // presence-only semantics let a harness
      return v && *v && std::strcmp(v, "0") != 0 &&    // exporting FLAG=0 silently ENABLE the
             std::strcmp(v, "false") != 0;             // feature it meant to turn off.
    };
    bool wantTimer = envFlag("EKV_TIMER");
    bool wantQlen = std::getenv("EKV_QLEN") != nullptr;
    bool wantShare = envFlag("EKV_ATTN_SHARE");
    int shareLayers = wantShare ? getEnvInt("EKV_LSE_LAYERS", 64) : 0;  // layers the table can hold
    int shareGroupRows = getEnvInt("EKV_LSE_ROWS", 8);                   // query rows kept per launch
    int qslCount = maxRows + 1;
    PubSet ps = selfAllocCapture((size_t)maxRows * (size_t)ncap * 4u, ncap, maxRows,
                                 wantTimer ? (size_t)maxRows * 8u : 0,
                                 wantQlen ? (size_t)qslCount * 4u : 0, qslCount,
                                 shareLayers, shareGroupRows, func.getName().str().c_str());
    uint64_t dev = ps.capDev;
    if (dev == 0) {
      publishCoverage(func.getName().str().c_str(), "recognized_declined",
                      "request_state_allocation_failed");
      return;                                       // CUDA alloc failed -> don't touch the kernel
    }
    if (std::getenv("EKV_DEBUG"))
      std::fprintf(stderr, "[eKV-auto] kernel=%s idxArg=%d kv=%s row=%s num_seqs=%s ncap=%d maxRows=%d\n",
                   func.getName().str().c_str(), idxArg, argName(kvArg).c_str(),
                   isCsr ? "CSR(kv_indptr[seq])" : "paged(seqIdx*stride)",
                   boundVal ? "STRUCT" : "buf", ncap, maxRows);
    mlir::Block &entry = func.front();
    OpBuilder b(&entry, entry.begin());
    Location loc = func.getLoc();
    Value slot = tt::GetProgramIdOp::create(b, loc, slotAxis);   // per-CTA, for the timer bracket
    // Gates against PCIe atomic storms (the buffers are host-mapped; each ungated atomic is a
    // serialized non-posted round trip): the timer counts ONE representative head group (heads are
    // symmetric; shares invariant to the head-group factor), the idempotent kv_ptr XCHG gets ONE writer.
    Value zero32 = arith::ConstantIntOp::create(b, loc, 0, 32);
    Value headGate;
    if (headAxis >= 0)
      headGate = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                     tt::GetProgramIdOp::create(b, loc, headAxis), zero32).getResult();
    Value slot0 = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, slot, zero32).getResult();
    Value firstCta = headGate ? arith::AndIOp::create(b, loc, slot0, headGate).getResult() : slot0;
    // Coverage/lease/idempotent publications need exactly one writer even for a three-axis grid.
    for (int axis = 0; axis < 3; ++axis) {
      if (axis == slotAxis || axis == headAxis)
        continue;
      Value axis0 = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                        tt::GetProgramIdOp::create(b, loc, axis), zero32).getResult();
      firstCta = arith::AndIOp::create(b, loc, firstCta, axis0).getResult();
    }
    if (ps.coverageLaunchDev) {
      auto pre = opSnapshot(func);
      emitCoverageLaunch(b, loc, ps.coverageLaunchDev, firstCta);
      verifyEmission(func, pre, "coverage-launch-count", Effect::Monoid);
    }
    // Capture request<->units into buffer[request] using the kernel's OWN request index, so every query
    // block of a sequence writes that sequence's row (idempotent, sequence-aligned at prefill AND decode).
    // Both affine row-address forms write the same buffer[seq] format:
    if (isCsr) {
      // CSR/SGLang: buffer[cur_seq] = kv_indices[kv_indptr[seq] : +ncap] (additive), emitted after the
      // kv_indptr start load so the offset is in scope.
      OpBuilder cb(csr.second.getContext());
      cb.setInsertionPointAfter(csr.second.getDefiningOp());
      Location cl = csr.second.getLoc();
      // request KV length = kv_indptr[seq+1]-kv_indptr[seq]; the start load's ptr is &kv_indptr[seq], so
      // the next element (+1) is the end. Their difference bounds the ragged read (no OOB over-read).
      Value csrLen;
      if (auto startLd = mlir::dyn_cast<tt::LoadOp>(csr.second.getDefiningOp())) {
        Value sp = startLd.getPtr();
        Value one = arith::ConstantIntOp::create(cb, cl, 1, 32);
        Value endPtr = tt::AddPtrOp::create(cb, cl, sp.getType(), sp, one);
        Value endV = tt::LoadOp::create(cb, cl, endPtr, tt::CacheModifier::NONE,
                                        tt::EvictionPolicy::NORMAL, /*isVolatile=*/false).getResult();
        csrLen = arith::SubIOp::create(cb, cl, endV, csr.second).getResult();
      }
      auto pre = opSnapshot(func);
      emitCapture(func, csr.first, cb, cl, idxArg, dev, /*rowStride=*/0, ncap, maxRows,
                  headAxis, /*strideVal=*/Value(), boundVal, /*csrRowOffset=*/csr.second, csrLen);
      verifyEmission(func, pre, "capture", Effect::Idempotent);
    } else {
      // paged/vLLM: buffer[seqIdx] = block_table[seqIdx*stride : +ncap] (multiplicative). Anchor the
      // capture AFTER the row-offset multiply (seqIdx*stride), not merely after seqIdx: the multiply
      // dominates BOTH seqIdx and the stride operand, so the recomputed capture address can never
      // reference a value defined later than the insertion point (a dominance error if the stride is
      // defined after seqIdx).
      mlir::Value rowOff = addptrOffsetOnBase(idxLoad.getPtr(), func.getArgument(idxArg), 24);
      mlir::Operation *anchor = (rowOff && rowOff.getDefiningOp()) ? rowOff.getDefiningOp()
                                                                   : seqIdx.getDefiningOp();
      OpBuilder cb(anchor->getContext());
      cb.setInsertionPointAfter(anchor);
      auto pre = opSnapshot(func);
      emitCapture(func, seqIdx, cb, anchor->getLoc(), idxArg, dev, /*rowStride=*/0, ncap, maxRows,
                  headAxis, strideVal, boundVal);
      verifyEmission(func, pre, "capture", Effect::Idempotent);
    }
    // The timer brackets the WHOLE kernel: t0 here is at entry (builder b), t1 at every return. The
    // capture is emitted mid-body (it needs seqIdx, computed by find_seq_idx), so it executes INSIDE
    // [t0,t1] and IS counted -- one masked ncap-wide store (~0.5KB), negligible vs the attention
    // contraction. (The legacy/oracle path CAN exclude it -- its capture is at entry on the same
    // builder -- but the auto path's capture is inherently mid-body, so we count it honestly.)
    // The timer row is the kernel's OWN request index (the same value that grades the capture), so a
    // sequence split over many q-block CTAs (chunked prefill) accumulates into ONE row: cycles are
    // per request, not per CTA; emitTimer falls back to program_id where it does not dominate.
    if (wantTimer) {
      auto pre = opSnapshot(func);
      emitTimer(func, slot, b, loc, ps.timerDev, maxRows, isCsr ? csr.first : seqIdx, headGate);
      verifyEmission(func, pre, "timer", Effect::Monoid);
    }
    // Record the log of each tile's total attention weight, for the audit steps the serving process flags.
    ShareResult shareResult;
    if (wantShare && ps.share.lseDev) {
      auto pre = opSnapshot(func);
      shareResult = emitTileLse(func, idxArg, idxLoad.getResult(), isCsr ? csr.first : seqIdx,
                                kvArg, headAxis, ps.share.lseDev, ps.share.flagDev,
                                ps.share.layerTableDev, ps.share.layers, ps.share.rows,
                                ps.share.width, ps.share.groupRows);
      verifyEmission(func, pre, "tile-lse", Effect::Idempotent);
    }
    // publish the KV-cache base pointer -> exact KV-pool identification, no size heuristic.
    if (kvArg) {
      auto pre = opSnapshot(func);
      emitKvPtr(b, loc, kvArg, ps.kvPtrDev, firstCta);
      verifyEmission(func, pre, "kv_ptr", Effect::Idempotent);
    }
    // snapshot cu_seqlens_q -> per-seq query length (exact recompute vs cache-hit signal).
    if (wantQlen && ps.qslDev) {
      if (qslArg) {                                                    // found above (structural-primary)
        auto pre = opSnapshot(func);
        emitQslCopy(b, loc, qslArg, ps.qslDev, qslCount, slotAxis, headAxis, boundVal);
        verifyEmission(func, pre, "qlen", Effect::Idempotent);
      }
      else if (std::getenv("EKV_DEBUG"))
        std::fprintf(stderr, "[eKV-auto] EKV_QLEN set but no cu_seqlens_q arg found in %s\n",
                     func.getName().str().c_str());
    }
    func->setAttr(attrs::kInstrumented, mlir::UnitAttr::get(func.getContext()));
    // Every instrumented kernel only observes. A kernel whose statistic was requested and not
    // recorded is reported as recognized and declined, so a gap is visible and never silent.
    const char *classification = "observation_only";
    const char *reason = wantShare ? "request_mapping_and_attention_share_observed"
                                   : "request_mapping_observed";
    if (wantShare && shareResult.sites == 0) {
      classification = "recognized_declined";
      reason = "attention_share_not_recorded";
    }
    publishCoverage(func.getName().str().c_str(), classification, reason, ps.capHost,
                    shareResult.sites, shareResult.tileBlocks, shareResult.newestRows);
  }

  // The explicit-config ORACLE / validation path (opt-in via EKV_LEGACY). It takes Python-stamped
  // attrs (exact arg index / row stride / buffer addresses) as GROUND TRUTH and instruments exactly
  // that -- NO detection. Run it on a kernel whose mapping is known, then run the default auto path
  // on the same kernel and check that structural detection recovers the same index lookup. It is not a
  // fallback: the two paths are mutually exclusive (selected by EKV_LEGACY).
  void instrumentLegacy(tt::FuncOp func) {
    if (func->hasAttr(attrs::kInstrumented)) return;
    auto ctrlAttr = func->getAttrOfType<mlir::StringAttr>(attrs::kCtrlPtr);
    auto btAttr = func->getAttrOfType<mlir::StringAttr>(attrs::kBtCtrlPtr);
    if (!ctrlAttr && !btAttr) return;  // not selected -> no-op
    if (func.getBody().empty()) return;

    int axis = getIntAttr(func, attrs::kSlotAxis, 0);
    mlir::Block &entry = func.front();
    OpBuilder b(&entry, entry.begin());
    Location loc = func.getLoc();
    Value slot = tt::GetProgramIdOp::create(b, loc, axis);  // i32, shared

    // capture FIRST, on the SAME builder, so the timer's t0 (read next, on b) lands AFTER it and the
    // capture cost is excluded from the per-CTA compute timing. (This path CAN exclude it because its
    // capture is at entry; the auto path's capture is inherently mid-body -- see instrumentAuto.)
    captureBlockTable(func, slot, b, loc);

    if (ctrlAttr)
      emitTimer(func, slot, b, loc,
                std::strtoull(ctrlAttr.getValue().str().c_str(), nullptr, 10),
                getIntAttr(func, attrs::kBtMaxRows, 0));   // row bound from the configured max_rows (0 = off)
    func->setAttr(attrs::kInstrumented, mlir::UnitAttr::get(func.getContext()));
  }

  void instrument(tt::FuncOp func) {
    // DEFAULT: zero-touch structural detection (the contribution). EKV_LEGACY switches to the
    // explicit-config oracle above (used to validate the detection). EKV_PASS_AUTO is no longer
    // needed -- auto IS the default; setting it remains harmless (auto runs either way).
    if (std::getenv("EKV_LEGACY")) {
      instrumentLegacy(func);
      return;
    }
    instrumentAuto(func);
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createEKVTimerPass() {
  return std::make_unique<EKVTimerPass>();
}

}  // namespace ekv
