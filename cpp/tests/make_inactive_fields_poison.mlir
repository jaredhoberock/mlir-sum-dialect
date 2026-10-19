// RUN: mlir-opt --pass-pipeline="builtin.module(convert-to-llvm)" %s | FileCheck %s

// A sum value defines its tag and the payload of the variant it holds, and
// nothing else: the make lowering builds the struct over poison and inserts the
// tag, narrowed to its field's width, and each payload operand, so the payload
// field of a variant the value does not hold is the poison its make supplies,
// which no read under sum.get's precondition observes. The tag is inserted over
// the poison base, so every value's tag is defined; a payload is inserted into
// its variant's field.

// CHECK-LABEL: llvm.func @make_nullary
// CHECK: %[[INDEX:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK: %[[ABSENT:.*]] = llvm.mlir.poison : i64
// CHECK: %[[TAG:.*]] = llvm.trunc %[[INDEX]] : i64 to i8
// CHECK: %[[BASE:.*]] = llvm.mlir.poison : !llvm.struct<(i8, struct<()>, i64)>
// CHECK: %[[TAGGED:.*]] = llvm.insertvalue %[[TAG]], %[[BASE]][0]
// CHECK: %[[SUM:.*]] = llvm.insertvalue %[[ABSENT]], %[[TAGGED]][2]
// CHECK: llvm.return %[[SUM]]
func.func @make_nullary() -> !sum.sum<(none, i64)> {
  %c_tag = arith.constant 0 : index
  %c_absent = ub.poison : i64
  %c = sum.make %c_tag (%c_absent : i64) : !sum.sum<(none, i64)>
  return %c : !sum.sum<(none, i64)>
}

// CHECK-LABEL: llvm.func @make_payload
// CHECK-SAME: (%[[X:.*]]: i64)
// CHECK: %[[INDEX:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK: %[[TAG:.*]] = llvm.trunc %[[INDEX]] : i64 to i8
// CHECK: %[[BASE:.*]] = llvm.mlir.poison : !llvm.struct<(i8, struct<()>, i64)>
// CHECK: %[[TAGGED:.*]] = llvm.insertvalue %[[TAG]], %[[BASE]][0]
// CHECK: %[[SUM:.*]] = llvm.insertvalue %[[X]], %[[TAGGED]][2]
// CHECK: llvm.return %[[SUM]]
func.func @make_payload(%x: i64) -> !sum.sum<(none, i64)> {
  %c_tag = arith.constant 1 : index
  %c = sum.make %c_tag (%x : i64) : !sum.sum<(none, i64)>
  return %c : !sum.sum<(none, i64)>
}
