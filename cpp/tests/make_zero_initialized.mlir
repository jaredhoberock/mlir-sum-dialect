// RUN: mlir-opt --pass-pipeline="builtin.module(convert-to-llvm)" %s | FileCheck %s

// A sum value carries no undefined field: the make lowering builds the struct
// from zero, so the payload fields of the variants a value does not hold are
// defined rather than poison. The tag is inserted into the zero struct.

// CHECK-LABEL: llvm.func @make_nullary
// CHECK: %[[Z:.*]] = llvm.mlir.zero : !llvm.struct<
// CHECK: llvm.insertvalue %{{.*}}, %[[Z]][0]
// CHECK: llvm.return
func.func @make_nullary() -> !sum.sum<(none, i64)> {
  %c = sum.make 0 : !sum.sum<(none, i64)>
  return %c : !sum.sum<(none, i64)>
}
