// RUN: mlir-opt %s | FileCheck %s

// ---- Test: make with nullary variant (no payload) ----
// CHECK-LABEL: func @make_nullary
// CHECK: sum.make 0 : !sum.sum<(none, i64)>
func.func @make_nullary() -> !sum.sum<(none, i64)> {
  %c = sum.make 0 : !sum.sum<(none, i64)>
  return %c : !sum.sum<(none, i64)>
}

// ---- Test: make with non-nullary variant alongside nullary ----
// CHECK-LABEL: func @make_non_nullary
// CHECK: sum.make 1 %arg0 : !sum.sum<(none, i64)>
func.func @make_non_nullary(%arg0: i64) -> !sum.sum<(none, i64)> {
  %c = sum.make 1 %arg0 : !sum.sum<(none, i64)>
  return %c : !sum.sum<(none, i64)>
}

// ---- Test: match with nullary case (no block arg) ----
// CHECK-LABEL: func @match_nullary
// CHECK: sum.match %arg0 : !sum.sum<(none, i64)> -> i64
// CHECK: case 0
// CHECK: yield
// CHECK: case 1 (%arg1: i64)
// CHECK: yield %arg1
func.func @match_nullary(%arg0: !sum.sum<(none, i64)>) -> i64 {
  %result = sum.match %arg0 : !sum.sum<(none, i64)> -> i64
  case 0 {
    %c = arith.constant 42 : i64
    yield %c : i64
  }
  case 1 (%inner: i64) {
    yield %inner : i64
  }
  return %result : i64
}

// ---- Test: match with all nullary variants ----
// CHECK-LABEL: func @match_all_nullary
// CHECK: sum.match %arg0 : !sum.sum<(none, none)> -> i64
// CHECK: case 0
// CHECK: case 1
func.func @match_all_nullary(%arg0: !sum.sum<(none, none)>) -> i64 {
  %result = sum.match %arg0 : !sum.sum<(none, none)> -> i64
  case 0 {
    %c = arith.constant 1 : i64
    yield %c : i64
  }
  case 1 {
    %c = arith.constant 2 : i64
    yield %c : i64
  }
  return %result : i64
}
