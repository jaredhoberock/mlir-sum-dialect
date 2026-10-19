// RUN: mlir-opt --canonicalize %s | FileCheck %s

// A read of a sum built in view folds to what it reads. Of a make: the tag is
// its tag operand, is_variant compares that tag with the variant (a constant
// tag answers it), and get of every variant reads that variant's payload
// operand, whatever the tag: where the make holds another variant the read is
// poison, and the operand, the poison its make site supplies, refines it. A
// make of a sum's own tag and its own payloads is the sum. Of a sum an
// scf.index_switch rebuilds from its tag -- arm j yielding the make of variant
// j, the default the last variant's -- the tag is the switch's tag clamped to
// the last variant, is_variant compares that tag, and get reads the payload arm
// k makes when it is defined outside the arm, whichever of the switch's results
// the sum is. A payload built inside its arm does not dominate the read, so the
// read stays. Every read of a switch whose arms do not make their own variants
// stays.

!S = !sum.sum<(i64, none, f32)>
!I = !sum.sum<(i64, none)>
!N = !sum.sum<(!sum.sum<(i64, none)>, none)>
!M = memref<?xf32>
!B = !sum.sum<(!M, none)>

// CHECK-LABEL: func.func @reads_of_a_make
// CHECK-SAME: (%[[X:.*]]: i64)
// CHECK-DAG: %[[TAG:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[ABSENT:.*]] = ub.poison : f32
// CHECK-DAG: %[[TRUE:.*]] = arith.constant true
// CHECK-DAG: %[[FALSE:.*]] = arith.constant false
// CHECK-NOT: sum.
// CHECK: return %[[TAG]], %[[TRUE]], %[[FALSE]], %[[X]], %[[ABSENT]]
func.func @reads_of_a_make(%x: i64) -> (index, i1, i1, i64, f32) {
  %s_tag = arith.constant 0 : index
  %s_absent = ub.poison : f32
  %s = sum.make %s_tag (%x, %s_absent : i64, f32) : !S
  %t = sum.tag %s : !S
  %a = sum.is_variant %s, 0 : !S
  %b = sum.is_variant %s, 1 : !S
  %p = sum.get %s, 0 : !S -> i64
  %q = sum.get %s, 2 : !S -> f32
  return %t, %a, %b, %p, %q : index, i1, i1, i64, f32
}

// CHECK-LABEL: func.func @reads_of_a_make_of_a_dynamic_tag
// CHECK-SAME: (%[[TAG:.*]]: index, %[[X:.*]]: i64, %[[Y:.*]]: f32)
// CHECK-DAG: %[[FIRST:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[LAST:.*]] = arith.constant 2 : index
// CHECK: %[[A:.*]] = arith.cmpi eq, %[[TAG]], %[[FIRST]]
// CHECK: %[[B:.*]] = arith.cmpi eq, %[[TAG]], %[[LAST]]
// CHECK-NOT: sum.
// CHECK: return %[[TAG]], %[[A]], %[[B]], %[[X]], %[[Y]]
func.func @reads_of_a_make_of_a_dynamic_tag(%tag: index, %x: i64, %y: f32) -> (index, i1, i1, i64, f32) {
  %s = sum.make %tag (%x, %y : i64, f32) : !S
  %t = sum.tag %s : !S
  %a = sum.is_variant %s, 0 : !S
  %b = sum.is_variant %s, 2 : !S
  %p = sum.get %s, 0 : !S -> i64
  %q = sum.get %s, 2 : !S -> f32
  return %t, %a, %b, %p, %q : index, i1, i1, i64, f32
}

// The payloads taken apart as the leaf conversion takes them apart: each the
// read of the sum's own payload.
// CHECK-LABEL: func.func @make_of_its_own_parts
// CHECK-SAME: (%[[S:.*]]: !sum.sum<(memref<?xf32>, none)>)
// CHECK-NOT: sum.make
// CHECK: return %[[S]]
func.func @make_of_its_own_parts(%s: !B) -> !B {
  %t = sum.tag %s : !B
  %p = sum.get %s, 0 : !B -> !M
  %read = sum.make %t (%p : !M) : !B
  return %read : !B
}

// The tag of one sum with the payload of another is not the sum.
// CHECK-LABEL: func.func @make_of_other_parts
// CHECK: sum.make
// CHECK: return
func.func @make_of_other_parts(%s: !B, %r: !B) -> !B {
  %t = sum.tag %s : !B
  %p = sum.get %r, 0 : !B -> !M
  %mixed = sum.make %t (%p : !M) : !B
  return %mixed : !B
}

// CHECK-LABEL: func.func @reads_of_a_rebuilt_sum
// CHECK-NOT: scf.index_switch
// CHECK: arith.minui
// CHECK: arith.cmpi eq
// CHECK: arith.cmpi uge
// CHECK-NOT: sum.
// CHECK: return
func.func @reads_of_a_rebuilt_sum(%tag: index, %x: i64, %y: f32) -> (index, i1, i1, i64, f32) {
  %s = scf.index_switch %tag -> !S
  case 0 {
    %m_tag = arith.constant 0 : index
    %m_absent = ub.poison : f32
    %m = sum.make %m_tag (%x, %m_absent : i64, f32) : !S
    scf.yield %m : !S
  }
  case 1 {
    %m_tag = arith.constant 1 : index
    %m_absent0 = ub.poison : i64
    %m_absent2 = ub.poison : f32
    %m = sum.make %m_tag (%m_absent0, %m_absent2 : i64, f32) : !S
    scf.yield %m : !S
  }
  default {
    %m_tag = arith.constant 2 : index
    %m_absent = ub.poison : i64
    %m = sum.make %m_tag (%m_absent, %y : i64, f32) : !S
    scf.yield %m : !S
  }
  %t = sum.tag %s : !S
  %a = sum.is_variant %s, 0 : !S
  %b = sum.is_variant %s, 2 : !S
  %p = sum.get %s, 0 : !S -> i64
  %q = sum.get %s, 2 : !S -> f32
  return %t, %a, %b, %p, %q : index, i1, i1, i64, f32
}

// CHECK-LABEL: func.func @read_of_a_payload_built_in_its_arm
// CHECK-SAME: (%[[TAG:.*]]: index, %[[X:.*]]: i64)
// CHECK: %[[FIRST:.*]] = arith.constant 0 : index
// CHECK: %[[S:.*]] = scf.index_switch %[[TAG]]
// CHECK-NEXT: case 0 {
// CHECK-NEXT: %[[INNER:.*]] = sum.make %[[FIRST]] (%[[X]] : i64)
// CHECK-NEXT: %[[OUTER:.*]] = sum.make %[[FIRST]] (%[[INNER]] :
// CHECK-NEXT: scf.yield %[[OUTER]]
// CHECK: %[[P:.*]] = sum.get %[[S]], 0
// CHECK-NEXT: return %[[P]]
func.func @read_of_a_payload_built_in_its_arm(%tag: index, %x: i64) -> !I {
  %s = scf.index_switch %tag -> !N
  case 0 {
    %inner_tag = arith.constant 0 : index
    %inner = sum.make %inner_tag (%x : i64) : !I
    %m_tag = arith.constant 0 : index
    %m = sum.make %m_tag (%inner : !sum.sum<(i64, none)>) : !N
    scf.yield %m : !N
  }
  default {
    %m_tag = arith.constant 1 : index
    %m_absent = ub.poison : !sum.sum<(i64, none)>
    %m = sum.make %m_tag (%m_absent : !sum.sum<(i64, none)>) : !N
    scf.yield %m : !N
  }
  %p = sum.get %s, 0 : !N -> !I
  return %p : !I
}

// CHECK-LABEL: func.func @reads_of_a_switch_whose_arms_make_other_variants
// CHECK: scf.index_switch
// CHECK: sum.tag
// CHECK: sum.is_variant
// CHECK: return
func.func @reads_of_a_switch_whose_arms_make_other_variants(%tag: index, %x: i64) -> (index, i1) {
  %s = scf.index_switch %tag -> !I
  case 0 {
    %m_tag = arith.constant 1 : index
    %m_absent = ub.poison : i64
    %m = sum.make %m_tag (%m_absent : i64) : !I
    scf.yield %m : !I
  }
  default {
    %m_tag = arith.constant 0 : index
    %m = sum.make %m_tag (%x : i64) : !I
    scf.yield %m : !I
  }
  %t = sum.tag %s : !I
  %a = sum.is_variant %s, 0 : !I
  return %t, %a : index, i1
}

// CHECK-LABEL: func.func @reads_of_a_rebuilt_sum_beside_other_results
// CHECK-SAME: (%[[TAG:.*]]: index, %[[X:.*]]: i64, %[[Y:.*]]: i64)
// CHECK: %[[OTHER:.*]] = scf.index_switch %[[TAG]] -> i64
// CHECK: %[[HOLDS:.*]] = arith.cmpi eq, %[[TAG]]
// CHECK-NOT: sum.
// CHECK: return %[[OTHER]], %[[HOLDS]], %[[X]]
func.func @reads_of_a_rebuilt_sum_beside_other_results(%tag: index, %x: i64, %y: i64) -> (i64, i1, i64) {
  %o, %s = scf.index_switch %tag -> i64, !I
  case 0 {
    %m_tag = arith.constant 0 : index
    %m = sum.make %m_tag (%x : i64) : !I
    scf.yield %y, %m : i64, !I
  }
  default {
    %m_tag = arith.constant 1 : index
    %m_absent = ub.poison : i64
    %m = sum.make %m_tag (%m_absent : i64) : !I
    scf.yield %x, %m : i64, !I
  }
  %a = sum.is_variant %s, 0 : !I
  %p = sum.get %s, 0 : !I -> i64
  return %o, %a, %p : i64, i1, i64
}
