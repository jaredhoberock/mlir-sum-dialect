// RUN: mlir-opt --pass-pipeline="builtin.module(convert-to-llvm)" %s | FileCheck %s

// A sum value defines its tag and the payload of the variant it holds, and
// nothing else: the make lowering builds the struct over poison, so the payload
// field of every variant the value does not hold is poison, which no read under
// sum.get's precondition observes. The tag is inserted over the poison base, so
// every value's tag is defined; a payload is inserted into its variant's field.

// CHECK-LABEL: llvm.func @make_nullary
// CHECK: %[[BASE:.*]] = llvm.mlir.poison : !llvm.struct<(i8, struct<()>, i64)>
// CHECK: %[[TAG:.*]] = llvm.mlir.constant(0 : i8) : i8
// CHECK: %[[SUM:.*]] = llvm.insertvalue %[[TAG]], %[[BASE]][0]
// CHECK: llvm.return %[[SUM]]
func.func @make_nullary() -> !sum.sum<(none, i64)> {
  %c = sum.make 0 : !sum.sum<(none, i64)>
  return %c : !sum.sum<(none, i64)>
}

// CHECK-LABEL: llvm.func @make_payload
// CHECK-SAME: (%[[X:.*]]: i64)
// CHECK: %[[BASE:.*]] = llvm.mlir.poison : !llvm.struct<(i8, struct<()>, i64)>
// CHECK: %[[TAG:.*]] = llvm.mlir.constant(1 : i8) : i8
// CHECK: %[[TAGGED:.*]] = llvm.insertvalue %[[TAG]], %[[BASE]][0]
// CHECK: %[[SUM:.*]] = llvm.insertvalue %[[X]], %[[TAGGED]][2]
// CHECK: llvm.return %[[SUM]]
func.func @make_payload(%x: i64) -> !sum.sum<(none, i64)> {
  %c = sum.make 1 %x : !sum.sum<(none, i64)>
  return %c : !sum.sum<(none, i64)>
}
