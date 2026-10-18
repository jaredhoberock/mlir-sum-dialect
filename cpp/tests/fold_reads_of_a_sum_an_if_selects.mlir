// RUN: mlir-opt --canonicalize %s | FileCheck %s

// A sum an scf.if selects -- each of its two arms yielding the make of a
// different variant -- holds the then arm's variant where the condition holds
// and the else arm's where it does not. A read of it folds to what selects the
// arm: is_variant is the condition, its negation, or false; the tag chooses
// between the two variants on the condition; and get reads the payload of the
// make of its variant when that payload is defined outside the make's arm, the
// only value the read can observe under its precondition. Once every read
// folds, the if is dead and goes. A payload built inside its arm is carried
// out of it: the if yields it from that arm and poison from the other, the
// value the read has where the sum holds the other variant; an if whose other
// arm then yields only poison is its pure arm. An if whose two arms make one
// variant selects between two payloads, so its reads stay.

!S = !sum.sum<(i64, none, f32)>
!I = !sum.sum<(i64, none)>
!N = !sum.sum<(!sum.sum<(i64, none)>, none)>

// CHECK-LABEL: func.func @reads_of_a_sum_an_if_selects
// CHECK-SAME: (%[[C:.*]]: i1, %[[X:.*]]: i64)
// CHECK-DAG: %[[TRUE:.*]] = arith.constant true
// CHECK-DAG: %[[SOME:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[NONE:.*]] = arith.constant 1 : index
// CHECK-NOT: scf.if
// CHECK: %[[TAG:.*]] = arith.select %[[C]], %[[SOME]], %[[NONE]] : index
// CHECK: %[[ABSENT:.*]] = arith.xori %[[C]], %[[TRUE]] : i1
// CHECK-NOT: sum.
// CHECK: return %[[TAG]], %[[C]], %[[ABSENT]], %[[X]]
func.func @reads_of_a_sum_an_if_selects(%c: i1, %x: i64) -> (index, i1, i1, i64) {
  %s = scf.if %c -> !I {
    %m = sum.make 0 %x : !I
    scf.yield %m : !I
  } else {
    %m = sum.make 1 : !I
    scf.yield %m : !I
  }
  %t = sum.tag %s : !I
  %a = sum.is_variant %s, 0 : !I
  %b = sum.is_variant %s, 1 : !I
  %p = sum.get %s, 0 : !I -> i64
  return %t, %a, %b, %p : index, i1, i1, i64
}

// CHECK-LABEL: func.func @reads_of_a_variant_neither_arm_makes
// CHECK-SAME: (%[[C:.*]]: i1, %[[X:.*]]: i64, %[[Y:.*]]: f32)
// CHECK-DAG: %[[FALSE:.*]] = arith.constant false
// CHECK-DAG: %[[FIRST:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[LAST:.*]] = arith.constant 2 : index
// CHECK-NOT: scf.if
// CHECK: %[[TAG:.*]] = arith.select %[[C]], %[[LAST]], %[[FIRST]] : index
// CHECK-NOT: sum.
// CHECK: return %[[TAG]], %[[FALSE]], %[[Y]], %[[X]]
func.func @reads_of_a_variant_neither_arm_makes(%c: i1, %x: i64, %y: f32) -> (index, i1, f32, i64) {
  %s = scf.if %c -> !S {
    %m = sum.make 2 %y : !S
    scf.yield %m : !S
  } else {
    %m = sum.make 0 %x : !S
    scf.yield %m : !S
  }
  %t = sum.tag %s : !S
  %a = sum.is_variant %s, 1 : !S
  %q = sum.get %s, 2 : !S -> f32
  %p = sum.get %s, 0 : !S -> i64
  return %t, %a, %q, %p : index, i1, f32, i64
}

// CHECK-LABEL: func.func @reads_of_a_sum_beside_other_results
// CHECK-SAME: (%[[C:.*]]: i1, %[[X:.*]]: i64, %[[Y:.*]]: i64)
// CHECK-NOT: sum.
// CHECK: return %{{.*}}, %[[C]], %[[X]]
func.func @reads_of_a_sum_beside_other_results(%c: i1, %x: i64, %y: i64) -> (i64, i1, i64) {
  %o, %s = scf.if %c -> (i64, !I) {
    %m = sum.make 0 %x : !I
    scf.yield %y, %m : i64, !I
  } else {
    %m = sum.make 1 : !I
    scf.yield %x, %m : i64, !I
  }
  %a = sum.is_variant %s, 0 : !I
  %p = sum.get %s, 0 : !I -> i64
  return %o, %a, %p : i64, i1, i64
}

// CHECK-LABEL: func.func @read_of_a_payload_built_in_its_arm
// CHECK-SAME: (%[[C:.*]]: i1, %[[X:.*]]: i64)
// CHECK-NOT: scf.if
// CHECK: %[[P:.*]] = sum.make 0 %[[X]]
// CHECK-NOT: sum.get
// CHECK: return %[[C]], %[[P]]
func.func @read_of_a_payload_built_in_its_arm(%c: i1, %x: i64) -> (i1, !I) {
  %s = scf.if %c -> !N {
    %inner = sum.make 0 %x : !I
    %m = sum.make 0 %inner : !N
    scf.yield %m : !N
  } else {
    %m = sum.make 1 : !N
    scf.yield %m : !N
  }
  %a = sum.is_variant %s, 0 : !N
  %p = sum.get %s, 0 : !N -> !I
  return %a, %p : i1, !I
}

// CHECK-LABEL: func.func @reads_of_an_if_whose_arms_make_one_variant
// CHECK: scf.if
// CHECK: sum.tag
// CHECK: sum.is_variant
// CHECK: sum.get
// CHECK: return
func.func @reads_of_an_if_whose_arms_make_one_variant(%c: i1, %x: i64, %y: i64) -> (index, i1, i64) {
  %s = scf.if %c -> !I {
    %m = sum.make 0 %x : !I
    scf.yield %m : !I
  } else {
    %m = sum.make 0 %y : !I
    scf.yield %m : !I
  }
  %t = sum.tag %s : !I
  %a = sum.is_variant %s, 0 : !I
  %p = sum.get %s, 0 : !I -> i64
  return %t, %a, %p : index, i1, i64
}
