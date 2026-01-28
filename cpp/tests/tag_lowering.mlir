// RUN: mlir-opt --pass-pipeline="builtin.module(convert-to-llvm)" %s | FileCheck %s

// ---- Test 1: tag on 2-variant sum ----
// CHECK-LABEL: llvm.func @tag_two_variants
// CHECK-NOT: sum.tag
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @tag_two_variants(%arg0: !sum.sum<(i64, i1)>) -> index {
  %t = sum.tag %arg0 : !sum.sum<(i64, i1)>
  return %t : index
}

// ---- Test 2: tag on nested sum ----
// CHECK-LABEL: llvm.func @tag_nested_sum
// CHECK-NOT: sum.tag
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @tag_nested_sum(%arg0: !sum.sum<(!sum.sum<(i64, i1)>, i1)>) -> index {
  %t = sum.tag %arg0 : !sum.sum<(!sum.sum<(i64, i1)>, i1)>
  return %t : index
}
