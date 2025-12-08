// RUN: mlir-opt --pass-pipeline="builtin.module(convert-to-llvm)" %s | FileCheck %s

// ---- Test 1: match with result ----
// CHECK-LABEL: llvm.func @match_with_result
// CHECK-NOT: sum.match
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @match_with_result(%arg0: !sum.sum<(i64, i32)>) -> i64 {
  %result = sum.match %arg0 : !sum.sum<(i64, i32)> -> i64
  case 0 (%inner: i64) {
    yield %inner : i64
  }
  case 1 (%inner: i32) {
    %c = arith.extsi %inner : i32 to i64
    yield %c : i64
  }
  return %result : i64
}

// ---- Test 2: match without result ----
// CHECK-LABEL: llvm.func @match_no_result
// CHECK-NOT: sum.match
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @match_no_result(%arg0: !sum.sum<(i64, i32)>) {
  sum.match %arg0 : !sum.sum<(i64, i32)>
  case 0 (%inner: i64) {
    yield
  }
  case 1 (%inner: i32) {
    yield
  }
  return
}

// ---- Test 3: three variants ----
// CHECK-LABEL: llvm.func @match_three_variants
// CHECK-NOT: sum.match
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @match_three_variants(%arg0: !sum.sum<(i64, i32, i16)>) -> i64 {
  %result = sum.match %arg0 : !sum.sum<(i64, i32, i16)> -> i64
  case 0 (%inner: i64) {
    yield %inner : i64
  }
  case 1 (%inner: i32) {
    %c = arith.extsi %inner : i32 to i64
    yield %c : i64
  }
  case 2 (%inner: i16) {
    %c = arith.extsi %inner : i16 to i64
    yield %c : i64
  }
  return %result : i64
}
