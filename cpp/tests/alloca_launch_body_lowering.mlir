// RUN: mlir-opt --pass-pipeline="builtin.module(convert-to-llvm)" %s | FileCheck %s

// ---- A sum op lowered inside a gpu.launch allocates in the LAUNCH body, not
// the host function. The launch body is outlined into the kernel, and its
// region is not isolated from above, so a host-entry alloca would be captured
// into the kernel as a stale host stack pointer. The alloca must appear after
// gpu.launch and before the branch inside it, with none in the host entry.
// CHECK-LABEL: llvm.func @make_in_launch
// CHECK-NOT: llvm.alloca
// CHECK: gpu.launch
// CHECK: llvm.alloca
// CHECK: llvm.cond_br
// CHECK-NOT: llvm.alloca
func.func @make_in_launch(%arg0: i64, %cond: i1) {
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%bxs = %c1, %bys = %c1, %bzs = %c1) {
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    %s = sum.make 0 %arg0 : !sum.sum<(i64, i32)>
    gpu.terminator
  ^bb2:
    gpu.terminator
  }
  return
}
