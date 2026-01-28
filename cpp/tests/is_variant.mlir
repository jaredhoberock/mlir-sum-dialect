// RUN: mlir-opt %s | FileCheck %s

// ---- Test 1: Option<i64> - is_variant 0 (Some) ----
// CHECK-LABEL: func @is_variant_some
// CHECK: %[[R:.+]] = sum.is_variant %arg0, 0 : !sum.sum<(i64, tuple<>)>
// CHECK: return %[[R]] : i1
func.func @is_variant_some(%arg0: !sum.sum<(i64, tuple<>)>) -> i1 {
  %r = sum.is_variant %arg0, 0 : !sum.sum<(i64, tuple<>)>
  return %r : i1
}

// ---- Test 2: Option<i64> - is_variant 1 (None) ----
// CHECK-LABEL: func @is_variant_none
// CHECK: %[[R:.+]] = sum.is_variant %arg0, 1 : !sum.sum<(i64, tuple<>)>
// CHECK: return %[[R]] : i1
func.func @is_variant_none(%arg0: !sum.sum<(i64, tuple<>)>) -> i1 {
  %r = sum.is_variant %arg0, 1 : !sum.sum<(i64, tuple<>)>
  return %r : i1
}

// ---- Test 3: Result<i64, i32> - is_variant ----
// CHECK-LABEL: func @is_variant_result
// CHECK: %[[R:.+]] = sum.is_variant %arg0, 1 : !sum.sum<(i64, i32)>
// CHECK: return %[[R]] : i1
func.func @is_variant_result(%arg0: !sum.sum<(i64, i32)>) -> i1 {
  %r = sum.is_variant %arg0, 1 : !sum.sum<(i64, i32)>
  return %r : i1
}

// ---- Test 4: Nested sum - is_variant ----
// CHECK-LABEL: func @is_variant_nested_sum
// CHECK: %[[R:.+]] = sum.is_variant %arg0, 0 : !sum.sum<(!sum.sum<(i64, tuple<>)>, tuple<>)>
// CHECK: return %[[R]] : i1
func.func @is_variant_nested_sum(%arg0: !sum.sum<(!sum.sum<(i64, tuple<>)>, tuple<>)>) -> i1 {
  %r = sum.is_variant %arg0, 0 : !sum.sum<(!sum.sum<(i64, tuple<>)>, tuple<>)>
  return %r : i1
}
