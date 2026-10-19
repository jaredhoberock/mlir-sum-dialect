// RUN: mlir-opt %s | FileCheck %s

// ---- Test 1: Option<i64> - Some variant ----
// CHECK-LABEL: func @make_some
// CHECK: %[[T:.+]] = arith.constant 0 : index
// CHECK: %[[A:.+]] = ub.poison : tuple<>
// CHECK: sum.make %[[T]] (%arg0, %[[A]] : i64, tuple<>) : !sum.sum<(i64, tuple<>)>
func.func @make_some(%arg0: i64) -> !sum.sum<(i64, tuple<>)> {
  %c_tag = arith.constant 0 : index
  %c_absent = ub.poison : tuple<>
  %c = sum.make %c_tag (%arg0, %c_absent : i64, tuple<>) : !sum.sum<(i64, tuple<>)>
  return %c : !sum.sum<(i64, tuple<>)>
}

// ---- Test 2: Option<i64> - None variant ----
// CHECK-LABEL: func @make_none
// CHECK: %[[T:.+]] = arith.constant 1 : index
// CHECK: %[[A:.+]] = ub.poison : i64
// CHECK: sum.make %[[T]] (%[[A]], %arg0 : i64, tuple<>) : !sum.sum<(i64, tuple<>)>
func.func @make_none(%arg0: tuple<>) -> !sum.sum<(i64, tuple<>)> {
  %c_tag = arith.constant 1 : index
  %c_absent = ub.poison : i64
  %c = sum.make %c_tag (%c_absent, %arg0 : i64, tuple<>) : !sum.sum<(i64, tuple<>)>
  return %c : !sum.sum<(i64, tuple<>)>
}

// ---- Test 3: Result<i64, i32> - Ok variant ----
// CHECK-LABEL: func @make_ok
// CHECK: %[[T:.+]] = arith.constant 0 : index
// CHECK: %[[A:.+]] = ub.poison : i32
// CHECK: sum.make %[[T]] (%arg0, %[[A]] : i64, i32) : !sum.sum<(i64, i32)>
func.func @make_ok(%arg0: i64) -> !sum.sum<(i64, i32)> {
  %c_tag = arith.constant 0 : index
  %c_absent = ub.poison : i32
  %c = sum.make %c_tag (%arg0, %c_absent : i64, i32) : !sum.sum<(i64, i32)>
  return %c : !sum.sum<(i64, i32)>
}

// ---- Test 4: Result<i64, i32> - Err variant ----
// CHECK-LABEL: func @make_err
// CHECK: %[[T:.+]] = arith.constant 1 : index
// CHECK: %[[A:.+]] = ub.poison : i64
// CHECK: sum.make %[[T]] (%[[A]], %arg0 : i64, i32) : !sum.sum<(i64, i32)>
func.func @make_err(%arg0: i32) -> !sum.sum<(i64, i32)> {
  %c_tag = arith.constant 1 : index
  %c_absent = ub.poison : i64
  %c = sum.make %c_tag (%c_absent, %arg0 : i64, i32) : !sum.sum<(i64, i32)>
  return %c : !sum.sum<(i64, i32)>
}

// ---- Test 5: Three variants ----
// CHECK-LABEL: func @make_three_variant
// CHECK: %[[T:.+]] = arith.constant 2 : index
// CHECK: %[[A:.+]] = ub.poison : i64
// CHECK: %[[B:.+]] = ub.poison : i32
// CHECK: sum.make %[[T]] (%[[A]], %[[B]], %arg0 : i64, i32, i16) : !sum.sum<(i64, i32, i16)>
func.func @make_three_variant(%arg0: i16) -> !sum.sum<(i64, i32, i16)> {
  %c_tag = arith.constant 2 : index
  %c_absent0 = ub.poison : i64
  %c_absent1 = ub.poison : i32
  %c = sum.make %c_tag (%c_absent0, %c_absent1, %arg0 : i64, i32, i16) : !sum.sum<(i64, i32, i16)>
  return %c : !sum.sum<(i64, i32, i16)>
}

// ---- Test 6: Nested sum (sum containing sum) ----
// CHECK-LABEL: func @make_nested_sum
// CHECK: %[[T:.+]] = arith.constant 0 : index
// CHECK: %[[A:.+]] = ub.poison : tuple<>
// CHECK: sum.make %[[T]] (%arg0, %[[A]] : !sum.sum<(i64, tuple<>)>, tuple<>) : !sum.sum<(!sum.sum<(i64, tuple<>)>, tuple<>)>
func.func @make_nested_sum(%arg0: !sum.sum<(i64, tuple<>)>) -> !sum.sum<(!sum.sum<(i64, tuple<>)>, tuple<>)> {
  %c_tag = arith.constant 0 : index
  %c_absent = ub.poison : tuple<>
  %c = sum.make %c_tag (%arg0, %c_absent : !sum.sum<(i64, tuple<>)>, tuple<>) : !sum.sum<(!sum.sum<(i64, tuple<>)>, tuple<>)>
  return %c : !sum.sum<(!sum.sum<(i64, tuple<>)>, tuple<>)>
}

// ---- Test 7: Sum containing tuple ----
// CHECK-LABEL: func @make_sum_of_tuple
// CHECK: %[[T:.+]] = arith.constant 0 : index
// CHECK: %[[A:.+]] = ub.poison : tuple<>
// CHECK: sum.make %[[T]] (%arg0, %[[A]] : tuple<i64, i32>, tuple<>) : !sum.sum<(tuple<i64, i32>, tuple<>)>
func.func @make_sum_of_tuple(%arg0: tuple<i64, i32>) -> !sum.sum<(tuple<i64, i32>, tuple<>)> {
  %c_tag = arith.constant 0 : index
  %c_absent = ub.poison : tuple<>
  %c = sum.make %c_tag (%arg0, %c_absent : tuple<i64, i32>, tuple<>) : !sum.sum<(tuple<i64, i32>, tuple<>)>
  return %c : !sum.sum<(tuple<i64, i32>, tuple<>)>
}
