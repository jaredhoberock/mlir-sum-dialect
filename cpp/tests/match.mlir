// RUN: mlir-opt %s | FileCheck %s

// ---- Test 1: match with result ----
// CHECK-LABEL: func @match_with_result
// CHECK: sum.match %arg0 : !sum.sum<(i64, i32)> -> i64
// CHECK: case 0 (%arg1: i64)
// CHECK: yield %arg1 : i64
// CHECK: case 1 (%arg1: i32)
// CHECK: yield %{{.*}} : i64
func.func @match_with_result(%arg0: !sum.sum<(i64, i32)>) -> i64 {
  %result = sum.match %arg0 : !sum.sum<(i64, i32)> -> i64
  case 0 (%inner: i64) {
    yield %inner : i64
  }
  case 1 (%inner: i32) {
    %c = arith.constant 0 : i64
    yield %c : i64
  }
  return %result : i64
}

// ---- Test 2: match without result ----
// CHECK-LABEL: func @match_no_result
// CHECK: sum.match %arg0 : !sum.sum<(i64, i32)>
// CHECK: case 0 (%arg1: i64)
// CHECK: yield
// CHECK: case 1 (%arg1: i32)
// CHECK: yield
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
// CHECK-LABEL: func @match_three_variants
// CHECK: sum.match %arg0 : !sum.sum<(i64, i32, i16)> -> i64
// CHECK: case 0
// CHECK: case 1
// CHECK: case 2
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

// ---- Test 4: nested match ----
// CHECK-LABEL: func @match_nested_sum
// CHECK: sum.match %arg0 : !sum.sum<(!sum.sum<(i64, i32)>, i16)> -> i64
func.func @match_nested_sum(%arg0: !sum.sum<(!sum.sum<(i64, i32)>, i16)>) -> i64 {
  %result = sum.match %arg0 : !sum.sum<(!sum.sum<(i64, i32)>, i16)> -> i64
  case 0 (%inner: !sum.sum<(i64, i32)>) {
    %r = sum.match %inner : !sum.sum<(i64, i32)> -> i64
    case 0 (%x: i64) {
      yield %x : i64
    }
    case 1 (%x: i32) {
      %c = arith.extsi %x : i32 to i64
      yield %c : i64
    }
    yield %r : i64
  }
  case 1 (%inner: i16) {
    %c = arith.extsi %inner : i16 to i64
    yield %c : i64
  }
  return %result : i64
}
