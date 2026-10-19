// RUN: mlir-opt --canonicalize %s | FileCheck %s

// A comparison against a constant of a select between two constants is the
// select of the two comparisons' truths: the select's condition where they are
// true and false, its negation where they are false and true, and that truth
// where they agree, for any predicate. A tag a branch between two constant tags
// selects is so compared as the branch's condition. A select of a value no
// constant spells is left to arith. The fold is among the sum dialect's
// canonicalizations, so the module's attribute names a sum type, which loads
// the dialect.

module attributes {test.loads_the_sum_dialect = !sum.sum<(i64, none)>} {

// CHECK-LABEL: func.func @equal_to_the_true_value
// CHECK-SAME: (%[[C:.*]]: i1)
// CHECK-NEXT: return %[[C]]
func.func @equal_to_the_true_value(%c: i1) -> i1 {
  %k0 = arith.constant 0 : index
  %k1 = arith.constant 1 : index
  %t = arith.select %c, %k0, %k1 : index
  %b = arith.cmpi eq, %t, %k0 : index
  return %b : i1
}

// CHECK-LABEL: func.func @equal_to_the_false_value
// CHECK-SAME: (%[[C:.*]]: i1)
// CHECK-DAG: %[[TRUE:.*]] = arith.constant true
// CHECK: %[[N:.*]] = arith.xori %[[C]], %[[TRUE]]
// CHECK-NEXT: return %[[N]]
func.func @equal_to_the_false_value(%c: i1) -> i1 {
  %k0 = arith.constant 0 : index
  %k1 = arith.constant 1 : index
  %t = arith.select %c, %k0, %k1 : index
  %b = arith.cmpi eq, %t, %k1 : index
  return %b : i1
}

// CHECK-LABEL: func.func @unequal_to_the_true_value
// CHECK-SAME: (%[[C:.*]]: i1)
// CHECK-DAG: %[[TRUE:.*]] = arith.constant true
// CHECK: %[[N:.*]] = arith.xori %[[C]], %[[TRUE]]
// CHECK-NEXT: return %[[N]]
func.func @unequal_to_the_true_value(%c: i1) -> i1 {
  %k0 = arith.constant 0 : index
  %k1 = arith.constant 1 : index
  %t = arith.select %c, %k0, %k1 : index
  %b = arith.cmpi ne, %t, %k0 : index
  return %b : i1
}

// CHECK-LABEL: func.func @unequal_to_the_false_value
// CHECK-SAME: (%[[C:.*]]: i1)
// CHECK-NEXT: return %[[C]]
func.func @unequal_to_the_false_value(%c: i1) -> i1 {
  %k0 = arith.constant 0 : index
  %k1 = arith.constant 1 : index
  %t = arith.select %c, %k0, %k1 : index
  %b = arith.cmpi ne, %t, %k1 : index
  return %b : i1
}

// CHECK-LABEL: func.func @equal_to_neither
// CHECK-NEXT: %[[FALSE:.*]] = arith.constant false
// CHECK-NEXT: return %[[FALSE]]
func.func @equal_to_neither(%c: i1) -> i1 {
  %k0 = arith.constant 0 : index
  %k1 = arith.constant 1 : index
  %k2 = arith.constant 2 : index
  %t = arith.select %c, %k0, %k1 : index
  %b = arith.cmpi eq, %t, %k2 : index
  return %b : i1
}

// CHECK-LABEL: func.func @below_one_of_them
// CHECK-SAME: (%[[C:.*]]: i1)
// CHECK-NEXT: return %[[C]]
func.func @below_one_of_them(%c: i1) -> i1 {
  %k1 = arith.constant 1 : i64
  %k3 = arith.constant 3 : i64
  %k2 = arith.constant 2 : i64
  %t = arith.select %c, %k1, %k3 : i64
  %b = arith.cmpi ult, %t, %k2 : i64
  return %b : i1
}

// CHECK-LABEL: func.func @constant_on_the_left
// CHECK-SAME: (%[[C:.*]]: i1)
// CHECK-NEXT: return %[[C]]
func.func @constant_on_the_left(%c: i1) -> i1 {
  %k0 = arith.constant 0 : index
  %k1 = arith.constant 1 : index
  %t = arith.select %c, %k0, %k1 : index
  %b = arith.cmpi eq, %k0, %t : index
  return %b : i1
}

// CHECK-LABEL: func.func @tag_a_branch_selects
// CHECK-SAME: (%[[C:.*]]: i1)
// CHECK-NOT: scf.if
// CHECK-NOT: arith.select
// CHECK: cf.assert %[[C]], "unwrap on None"
func.func @tag_a_branch_selects(%c: i1) {
  %t = scf.if %c -> index {
    %k0 = arith.constant 0 : index
    scf.yield %k0 : index
  } else {
    %k1 = arith.constant 1 : index
    scf.yield %k1 : index
  }
  %zero = arith.constant 0 : index
  %some = arith.cmpi eq, %t, %zero : index
  cf.assert %some, "unwrap on None"
  return
}

// CHECK-LABEL: func.func @select_of_a_value
// CHECK: arith.select
// CHECK: arith.cmpi eq
func.func @select_of_a_value(%c: i1, %x: index) -> i1 {
  %k0 = arith.constant 0 : index
  %t = arith.select %c, %k0, %x : index
  %b = arith.cmpi eq, %t, %k0 : index
  return %b : i1
}
}
