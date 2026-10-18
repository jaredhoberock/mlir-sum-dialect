// RUN: mlir-opt --canonicalize %s | FileCheck %s

// An scf.if whose then arm asserts false and yields poison is the assertion
// that its condition is false, followed by its else arm; an scf.if with no else
// arm whose then arm aborts is that assertion alone.

!O = !sum.sum<(i64, none)>

// CHECK-LABEL: func.func @then_arm_aborts
// CHECK-SAME: (%[[O:.*]]: !sum.sum<(i64, none)>, %[[A:.*]]: i64, %[[B:.*]]: i64)
// CHECK-DAG: %[[TRUE:.*]] = arith.constant true
// CHECK-NOT: scf.if
// CHECK: %[[NONE:.*]] = sum.is_variant %[[O]], 1
// CHECK-NEXT: %[[TAKEN:.*]] = arith.xori %[[NONE]], %[[TRUE]]
// CHECK-NEXT: cf.assert %[[TAKEN]], "index out of bounds"
// CHECK-NEXT: %[[SUM:.*]] = arith.addi %[[A]], %[[B]]
// CHECK-NEXT: return %[[SUM]]
func.func @then_arm_aborts(%o: !O, %a: i64, %b: i64) -> i64 {
  %none = sum.is_variant %o, 1 : !O
  %r = scf.if %none -> i64 {
    %false = arith.constant false
    cf.assert %false, "index out of bounds"
    %p = ub.poison : i64
    scf.yield %p : i64
  } else {
    %s = arith.addi %a, %b : i64
    scf.yield %s : i64
  }
  return %r : i64
}

// CHECK-LABEL: func.func @then_arm_aborts_without_else
// CHECK-SAME: (%[[O:.*]]: !sum.sum<(i64, none)>)
// CHECK-DAG: %[[TRUE:.*]] = arith.constant true
// CHECK-NOT: scf.if
// CHECK: %[[NONE:.*]] = sum.is_variant %[[O]], 1
// CHECK-NEXT: %[[TAKEN:.*]] = arith.xori %[[NONE]], %[[TRUE]]
// CHECK-NEXT: cf.assert %[[TAKEN]], "assert failed"
// CHECK-NEXT: return
func.func @then_arm_aborts_without_else(%o: !O) {
  %none = sum.is_variant %o, 1 : !O
  scf.if %none {
    %false = arith.constant false
    cf.assert %false, "assert failed"
  }
  return
}
