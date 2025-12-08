// RUN: mlir-opt %s | FileCheck %s

// ---- Test 1: Option<i64> - Some variant ----
// CHECK-LABEL: func @make_some
// CHECK: %[[C:.+]] = sum.make 0 %arg0 : !sum.sum<(i64, tuple<>)>
func.func @make_some(%arg0: i64) -> !sum.sum<(i64, tuple<>)> {
  %c = sum.make 0 %arg0 : !sum.sum<(i64, tuple<>)>
  return %c : !sum.sum<(i64, tuple<>)>
}

// ---- Test 2: Option<i64> - None variant ----
// CHECK-LABEL: func @make_none
// CHECK: %[[C:.+]] = sum.make 1 %arg0 : !sum.sum<(i64, tuple<>)>
func.func @make_none(%arg0: tuple<>) -> !sum.sum<(i64, tuple<>)> {
  %c = sum.make 1 %arg0 : !sum.sum<(i64, tuple<>)>
  return %c : !sum.sum<(i64, tuple<>)>
}

// ---- Test 3: Result<i64, i32> - Ok variant ----
// CHECK-LABEL: func @make_ok
// CHECK: %[[C:.+]] = sum.make 0 %arg0 : !sum.sum<(i64, i32)>
func.func @make_ok(%arg0: i64) -> !sum.sum<(i64, i32)> {
  %c = sum.make 0 %arg0 : !sum.sum<(i64, i32)>
  return %c : !sum.sum<(i64, i32)>
}

// ---- Test 4: Result<i64, i32> - Err variant ----
// CHECK-LABEL: func @make_err
// CHECK: %[[C:.+]] = sum.make 1 %arg0 : !sum.sum<(i64, i32)>
func.func @make_err(%arg0: i32) -> !sum.sum<(i64, i32)> {
  %c = sum.make 1 %arg0 : !sum.sum<(i64, i32)>
  return %c : !sum.sum<(i64, i32)>
}

// ---- Test 5: Three variants ----
// CHECK-LABEL: func @make_three_variant
// CHECK: %[[C:.+]] = sum.make 2 %arg0 : !sum.sum<(i64, i32, i16)>
func.func @make_three_variant(%arg0: i16) -> !sum.sum<(i64, i32, i16)> {
  %c = sum.make 2 %arg0 : !sum.sum<(i64, i32, i16)>
  return %c : !sum.sum<(i64, i32, i16)>
}

// ---- Test 6: Nested sum (sum containing sum) ----
// CHECK-LABEL: func @make_nested_sum
// CHECK: %[[C:.+]] = sum.make 0 %arg0 : !sum.sum<(!sum.sum<(i64, tuple<>)>, tuple<>)>
func.func @make_nested_sum(%arg0: !sum.sum<(i64, tuple<>)>) -> !sum.sum<(!sum.sum<(i64, tuple<>)>, tuple<>)> {
  %c = sum.make 0 %arg0 : !sum.sum<(!sum.sum<(i64, tuple<>)>, tuple<>)>
  return %c : !sum.sum<(!sum.sum<(i64, tuple<>)>, tuple<>)>
}

// ---- Test 7: Sum containing tuple ----
// CHECK-LABEL: func @make_sum_of_tuple
// CHECK: %[[C:.+]] = sum.make 0 %arg0 : !sum.sum<(tuple<i64, i32>, tuple<>)>
func.func @make_sum_of_tuple(%arg0: tuple<i64, i32>) -> !sum.sum<(tuple<i64, i32>, tuple<>)> {
  %c = sum.make 0 %arg0 : !sum.sum<(tuple<i64, i32>, tuple<>)>
  return %c : !sum.sum<(tuple<i64, i32>, tuple<>)>
}
