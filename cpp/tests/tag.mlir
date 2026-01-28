// RUN: mlir-opt %s | FileCheck %s

// ---- Test 1: Option<i64> - tag(Some) ----
// CHECK-LABEL: func @tag_some
// CHECK: %[[T:.+]] = sum.tag %arg0 : !sum.sum<(i64, tuple<>)>
// CHECK: return %[[T]] : index
func.func @tag_some(%arg0: !sum.sum<(i64, tuple<>)>) -> index {
  %t = sum.tag %arg0 : !sum.sum<(i64, tuple<>)>
  return %t : index
}

// ---- Test 2: Result<i64, i32> - tag ----
// CHECK-LABEL: func @tag_result
// CHECK: %[[T:.+]] = sum.tag %arg0 : !sum.sum<(i64, i32)>
// CHECK: return %[[T]] : index
func.func @tag_result(%arg0: !sum.sum<(i64, i32)>) -> index {
  %t = sum.tag %arg0 : !sum.sum<(i64, i32)>
  return %t : index
}

// ---- Test 3: Nested sum - tag ----
// CHECK-LABEL: func @tag_nested_sum
// CHECK: %[[T:.+]] = sum.tag %arg0 : !sum.sum<(!sum.sum<(i64, tuple<>)>, tuple<>)>
// CHECK: return %[[T]] : index
func.func @tag_nested_sum(%arg0: !sum.sum<(!sum.sum<(i64, tuple<>)>, tuple<>)>) -> index {
  %t = sum.tag %arg0 : !sum.sum<(!sum.sum<(i64, tuple<>)>, tuple<>)>
  return %t : index
}

// ---- Test 4: Sum containing tuple - tag ----
// CHECK-LABEL: func @tag_sum_of_tuple
// CHECK: %[[T:.+]] = sum.tag %arg0 : !sum.sum<(tuple<i64, i32>, tuple<>)>
// CHECK: return %[[T]] : index
func.func @tag_sum_of_tuple(%arg0: !sum.sum<(tuple<i64, i32>, tuple<>)>) -> index {
  %t = sum.tag %arg0 : !sum.sum<(tuple<i64, i32>, tuple<>)>
  return %t : index
}
