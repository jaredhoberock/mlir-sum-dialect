// RUN: mlir-opt %s | FileCheck %s

// ---- Test 1: 2-variant sum, get variant 0 payload ----
// CHECK-LABEL: func @get_v0_i64
// CHECK: %[[V:.+]] = sum.get %arg0, 0 : !sum.sum<(i64, i1)> -> i64
func.func @get_v0_i64(%arg0: !sum.sum<(i64, i1)>) -> i64 {
  %v = sum.get %arg0, 0 : !sum.sum<(i64, i1)> -> i64
  return %v : i64
}

// ---- Test 2: 2-variant sum, get variant 1 payload ----
// CHECK-LABEL: func @get_v1_i1
// CHECK: %[[V:.+]] = sum.get %arg0, 1 : !sum.sum<(i64, i1)> -> i1
func.func @get_v1_i1(%arg0: !sum.sum<(i64, i1)>) -> i1 {
  %v = sum.get %arg0, 1 : !sum.sum<(i64, i1)> -> i1
  return %v : i1
}

// ---- Test 3: nested sum payload ----
// CHECK-LABEL: func @get_nested_payload
// CHECK: %[[V:.+]] = sum.get %arg0, 0 : !sum.sum<(!sum.sum<(i64, i1)>, i1)> -> !sum.sum<(i64, i1)>
func.func @get_nested_payload(%arg0: !sum.sum<(!sum.sum<(i64, i1)>, i1)>) -> !sum.sum<(i64, i1)> {
  %v = sum.get %arg0, 0 : !sum.sum<(!sum.sum<(i64, i1)>, i1)> -> !sum.sum<(i64, i1)>
  return %v : !sum.sum<(i64, i1)>
}
