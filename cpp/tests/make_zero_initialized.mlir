// RUN: mlir-opt --pass-pipeline="builtin.module(convert-to-llvm)" %s | FileCheck %s

// A sum value carries no undefined bytes: the make lowering zeroes the whole
// struct before it writes the tag, so a nullary variant's unused payload slot
// and any bytes a narrower payload leaves in the shared union are defined rather
// than poison. The zero store lands on the alloca ahead of the tag store.

// CHECK-LABEL: llvm.func @make_nullary
// CHECK: %[[ALLOCA:.*]] = llvm.alloca
// CHECK: %[[Z:.*]] = llvm.mlir.zero : !llvm.struct<
// CHECK: llvm.store %[[Z]], %[[ALLOCA]]
// CHECK: llvm.return
func.func @make_nullary() -> !sum.sum<(none, i64)> {
  %c = sum.make 0 : !sum.sum<(none, i64)>
  return %c : !sum.sum<(none, i64)>
}
