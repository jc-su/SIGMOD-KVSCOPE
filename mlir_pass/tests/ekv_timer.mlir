// Unit test for the eKV timer pass (no GPU). Run with the plugin loaded:
//   TRITON_EXT_ENABLED=1 \
//   TRITON_PLUGIN_PATHS=<triton-src>/python/triton/plugins/libeKVPlugin.so \
//   <triton-src>/build/.../triton-opt -ekv-timer ekv_timer.mlir | FileCheck ekv_timer.mlir
//
// The pass must, for the SELECTED function @k (it carries ekv.ctrl_ptr), insert
// two clock64 reads, a get_program_id, and a scalar atomic_rmw add; and must
// leave @unselected (no ekv.ctrl_ptr) untouched.

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:86"} {
  // CHECK-LABEL: tt.func @k
  // CHECK: tt.elementwise_inline_asm
  // CHECK: tt.get_program_id
  // CHECK: tt.int_to_ptr
  // CHECK: tt.addptr
  // CHECK: tt.atomic_rmw
  // CHECK: tt.return
  tt.func @k(%arg0: !tt.ptr<f32>) attributes {
      ekv.ctrl_ptr = "140000000000", ekv.slot_axis = 0 : i32} {
    tt.return
  }

  // CHECK-LABEL: tt.func @unselected
  // CHECK-NOT: tt.atomic_rmw
  tt.func @unselected(%arg0: !tt.ptr<f32>) {
    tt.return
  }
}
