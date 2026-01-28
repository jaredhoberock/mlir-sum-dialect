// RUN: mlir-opt --pass-pipeline="builtin.module(convert-to-llvm)" %s | FileCheck %s

// ---- Test 1: is_variant on 2-variant sum ----
// CHECK-LABEL: llvm.func @is_variant_two_variants
// CHECK-NOT: sum.is_variant
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @is_variant_two_variants(%arg0: !sum.sum<(i64, i1)>) -> i1 {
  %r = sum.is_variant %arg0, 0 : !sum.sum<(i64, i1)>
  return %r : i1
}

// ---- Test 2: is_variant on nested sum ----
// CHECK-LABEL: llvm.func @is_variant_nested_sum
// CHECK-NOT: sum.is_variant
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @is_variant_nested_sum(%arg0: !sum.sum<(!sum.sum<(i64, i1)>, i1)>) -> i1 {
  %r = sum.is_variant %arg0, 1 : !sum.sum<(!sum.sum<(i64, i1)>, i1)>
  return %r : i1
}

// ---- Test 3: is_variant second variant ----
// CHECK-LABEL: llvm.func @is_variant_second
// CHECK-NOT: sum.is_variant
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @is_variant_second(%arg0: !sum.sum<(i64, i32, i1)>) -> i1 {
  %r = sum.is_variant %arg0, 1 : !sum.sum<(i64, i32, i1)>
  return %r : i1
}
