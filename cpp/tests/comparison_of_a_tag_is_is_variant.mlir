// RUN: mlir-opt --canonicalize %s | FileCheck %s

// A comparison of a sum's tag for equality with a constant that names a variant
// is the question whether the sum holds that variant: `arith.cmpi eq` becomes
// `sum.is_variant` and `ne` its negation, so the question folds through the
// branch that builds the sum to that branch's condition. A constant that names
// no variant, and any other predicate, is left to arith.

!O = !sum.sum<(i64, none)>
!T = !sum.sum<(i64, none, f32)>

// CHECK-LABEL: func.func @tag_of_a_sum_an_if_selects
// CHECK-SAME: (%[[C:.*]]: i1, %[[X:.*]]: i64)
// CHECK-DAG: %[[TRUE:.*]] = arith.constant true
// CHECK-NOT: scf.if
// CHECK-NOT: sum.
// CHECK: %[[OTHER:.*]] = arith.xori %[[C]], %[[TRUE]]
// CHECK-NEXT: %[[NONE:.*]] = arith.xori %[[C]], %[[TRUE]]
// CHECK-NEXT: return %[[C]], %[[OTHER]], %[[NONE]]
func.func @tag_of_a_sum_an_if_selects(%c: i1, %x: i64) -> (i1, i1, i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %s = scf.if %c -> !O {
    %m = sum.make 0 %x : !O
    scf.yield %m : !O
  } else {
    %m = sum.make 1 : !O
    scf.yield %m : !O
  }
  %t = sum.tag %s : !O
  %some = arith.cmpi eq, %t, %c0 : index
  %other = arith.cmpi ne, %t, %c0 : index
  %none = arith.cmpi eq, %c1, %t : index
  return %some, %other, %none : i1, i1, i1
}

// CHECK-LABEL: func.func @tag_of_an_argument
// CHECK-SAME: (%[[S:.*]]: !sum.sum<(i64, none, f32)>)
// CHECK-NOT: sum.tag
// CHECK: %[[LAST:.*]] = sum.is_variant %[[S]], 2
// CHECK-NEXT: return %[[LAST]]
func.func @tag_of_an_argument(%s: !T) -> i1 {
  %c2 = arith.constant 2 : index
  %t = sum.tag %s : !T
  %last = arith.cmpi eq, %t, %c2 : index
  return %last : i1
}

// CHECK-LABEL: func.func @comparisons_left_to_arith
// CHECK-SAME: (%[[S:.*]]: !sum.sum<(i64, none, f32)>)
// CHECK-NOT: sum.is_variant
// CHECK: %[[T:.*]] = sum.tag %[[S]]
// CHECK-DAG: arith.cmpi eq, %[[T]], %{{.*}}
// CHECK-DAG: arith.cmpi ult, %[[T]], %{{.*}}
// CHECK: return
func.func @comparisons_left_to_arith(%s: !T) -> (i1, i1) {
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %t = sum.tag %s : !T
  %beyond = arith.cmpi eq, %t, %c5 : index
  %first = arith.cmpi ult, %t, %c1 : index
  return %beyond, %first : i1, i1
}
