// RUN: mlir-opt --pass-pipeline="builtin.module(convert-to-llvm)" %s | FileCheck %s

// ---- Test 1: get from 2-variant sum (variant 0 -> i64) ----
// CHECK-LABEL: llvm.func @get_v0_i64
// CHECK-NOT: sum.get
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @get_v0_i64(%arg0: !sum.sum<(i64, i1)>) -> i64 {
  %v = sum.get %arg0, 0 : !sum.sum<(i64, i1)> -> i64
  return %v : i64
}

// ---- Test 2: get from 2-variant sum (variant 1 -> i1) ----
// CHECK-LABEL: llvm.func @get_v1_i1
// CHECK-NOT: sum.get
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @get_v1_i1(%arg0: !sum.sum<(i64, i1)>) -> i1 {
  %v = sum.get %arg0, 1 : !sum.sum<(i64, i1)> -> i1
  return %v : i1
}

// ---- Test 3: get nested sum payload (variant 0 -> !sum.sum<(i64,i1)>) ----
// CHECK-LABEL: llvm.func @get_nested_payload
// CHECK-NOT: sum.get
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @get_nested_payload(%arg0: !sum.sum<(!sum.sum<(i64, i1)>, i1)>) -> !sum.sum<(i64, i1)> {
  %v = sum.get %arg0, 0 : !sum.sum<(!sum.sum<(i64, i1)>, i1)> -> !sum.sum<(i64, i1)>
  return %v : !sum.sum<(i64, i1)>
}
