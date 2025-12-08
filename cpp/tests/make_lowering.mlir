// RUN: mlir-opt --pass-pipeline="builtin.module(convert-to-llvm)" %s | FileCheck %s

// -----
// CHECK-LABEL: llvm.func @make_first
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @make_first(%arg0: i64) -> !sum.sum<(i64, i32)> {
  %c = sum.make 0 %arg0 : !sum.sum<(i64, i32)>
  return %c : !sum.sum<(i64, i32)>
}

// -----
// CHECK-LABEL: llvm.func @make_second
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @make_second(%arg0: i32) -> !sum.sum<(i64, i32)> {
  %c = sum.make 1 %arg0 : !sum.sum<(i64, i32)>
  return %c : !sum.sum<(i64, i32)>
}

// -----
// CHECK-LABEL: llvm.func @make_three_variants
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @make_three_variants(%arg0: i16) -> !sum.sum<(i64, i32, i16)> {
  %c = sum.make 2 %arg0 : !sum.sum<(i64, i32, i16)>
  return %c : !sum.sum<(i64, i32, i16)>
}

// -----
// CHECK-LABEL: llvm.func @make_floats
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @make_floats(%arg0: f32) -> !sum.sum<(f64, f32)> {
  %c = sum.make 1 %arg0 : !sum.sum<(f64, f32)>
  return %c : !sum.sum<(f64, f32)>
}

// -----
// CHECK-LABEL: llvm.func @make_nested_sum
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @make_nested_sum(%arg0: !sum.sum<(i64, i32)>) -> !sum.sum<(!sum.sum<(i64, i32)>, i16)> {
  %c = sum.make 0 %arg0 : !sum.sum<(!sum.sum<(i64, i32)>, i16)>
  return %c : !sum.sum<(!sum.sum<(i64, i32)>, i16)>
}

// -----
// CHECK-LABEL: llvm.func @make_mixed_sizes
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @make_mixed_sizes(%arg0: i8) -> !sum.sum<(i64, i32, i16, i8)> {
  %c = sum.make 3 %arg0 : !sum.sum<(i64, i32, i16, i8)>
  return %c : !sum.sum<(i64, i32, i16, i8)>
}
