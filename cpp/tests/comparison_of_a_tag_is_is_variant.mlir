// RUN: mlir-opt --canonicalize %s | FileCheck %s
// RUN: mlir-opt --pass-pipeline="builtin.module(leaf-conversion)" %s | FileCheck %s --check-prefix=FLAT

// A comparison of a sum's tag for equality with a constant that names a variant
// is the question whether the sum holds that variant: `arith.cmpi eq` becomes
// `sum.is_variant` and `ne` its negation. Of a sum a branch builds, once the
// flattening takes the sum apart, the question is of a make whose tag is a
// choice between two constants on the branch's condition, and it folds to that
// condition. A constant that names no variant, and any other predicate, is
// left to arith.

!O = !sum.sum<(i64, none)>
!T = !sum.sum<(i64, none, f32)>

// FLAT-LABEL: func.func @tag_of_a_sum_an_if_selects
// FLAT-SAME: (%[[C:.*]]: i1, %[[X:.*]]: i64)
// FLAT-DAG: %[[TRUE:.*]] = arith.constant true
// FLAT-NOT: scf.if
// FLAT-NOT: sum.
// FLAT: %[[OTHER:.*]] = arith.xori %[[C]], %[[TRUE]]
// FLAT-NEXT: %[[NONE:.*]] = arith.xori %[[C]], %[[TRUE]]
// FLAT-NEXT: return %[[C]], %[[OTHER]], %[[NONE]]
func.func @tag_of_a_sum_an_if_selects(%c: i1, %x: i64) -> (i1, i1, i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %s = scf.if %c -> !O {
    %m_tag = arith.constant 0 : index
    %m = sum.make %m_tag (%x : i64) : !O
    scf.yield %m : !O
  } else {
    %m_tag = arith.constant 1 : index
    %m_absent = ub.poison : i64
    %m = sum.make %m_tag (%m_absent : i64) : !O
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
