// RUN: mlir-opt --canonicalize %s | FileCheck %s

// A read of a sum built in view folds to what it reads. Of a make: the tag is
// the variant made, is_variant compares it, and get reads the payload of the
// variant made; a get of another variant states a precondition that does not
// hold, so no value answers it and it stays. Of a sum an scf.index_switch
// rebuilds from its tag -- arm j yielding the make of variant j, the default
// the last variant's -- the tag is the switch's tag clamped to the last variant,
// is_variant compares that tag, and get reads the payload arm k makes when it is
// defined outside the arm, whichever of the switch's results the sum is. A
// payload built inside its arm is carried out of it: the switch yields it from
// that arm and poison from every other, the value the read has where the sum
// holds another variant. Every read of a switch whose arms do not make their
// own variants stays.

!S = !sum.sum<(i64, none, f32)>
!I = !sum.sum<(i64, none)>
!N = !sum.sum<(!sum.sum<(i64, none)>, none)>

// CHECK-LABEL: func.func @reads_of_a_make
// CHECK-NOT: sum.tag
// CHECK-NOT: sum.is_variant
// CHECK: sum.get {{.*}}, 2
// CHECK-NOT: sum.get
// CHECK: return
func.func @reads_of_a_make(%x: i64) -> (index, i1, i1, i64, f32) {
  %s = sum.make 0 %x : !S
  %t = sum.tag %s : !S
  %a = sum.is_variant %s, 0 : !S
  %b = sum.is_variant %s, 1 : !S
  %p = sum.get %s, 0 : !S -> i64
  %q = sum.get %s, 2 : !S -> f32
  return %t, %a, %b, %p, %q : index, i1, i1, i64, f32
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
    %m = sum.make 0 %x : !S
    scf.yield %m : !S
  }
  case 1 {
    %m = sum.make 1 : !S
    scf.yield %m : !S
  }
  default {
    %m = sum.make 2 %y : !S
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
// CHECK: %[[POISON:.*]] = ub.poison
// CHECK: %[[P:.*]] = scf.index_switch %[[TAG]]
// CHECK-NEXT: case 0 {
// CHECK-NEXT: %[[INNER:.*]] = sum.make 0 %[[X]]
// CHECK-NEXT: scf.yield %[[INNER]]
// CHECK: default {
// CHECK-NEXT: scf.yield %[[POISON]]
// CHECK-NOT: sum.get
// CHECK: return %[[P]]
func.func @read_of_a_payload_built_in_its_arm(%tag: index, %x: i64) -> !I {
  %s = scf.index_switch %tag -> !N
  case 0 {
    %inner = sum.make 0 %x : !I
    %m = sum.make 0 %inner : !N
    scf.yield %m : !N
  }
  default {
    %m = sum.make 1 : !N
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
    %m = sum.make 1 : !I
    scf.yield %m : !I
  }
  default {
    %m = sum.make 0 %x : !I
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
    %m = sum.make 0 %x : !I
    scf.yield %y, %m : i64, !I
  }
  default {
    %m = sum.make 1 : !I
    scf.yield %x, %m : i64, !I
  }
  %a = sum.is_variant %s, 0 : !I
  %p = sum.get %s, 0 : !I -> i64
  return %o, %a, %p : i64, i1, i64
}
