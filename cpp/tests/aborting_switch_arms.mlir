// RUN: mlir-opt --canonicalize %s | FileCheck %s

// An scf.index_switch all of whose arms but one assert false is a sequence of
// assertions followed by its live arm: that the value selects no aborting case,
// each with that case's message, then, when the default arm aborts, that the
// value selects the live case, with the default's message. The first shape reads
// a sum's payload, as a match lowers to: one case live and the default
// aborting.

!O = !sum.sum<(f32, none)>
!T = !sum.sum<(none, f32, none)>

// CHECK-LABEL: func.func @one_live_case
// CHECK-SAME: (%[[O:.*]]: !sum.sum<(f32, none)>)
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NOT: scf.index_switch
// CHECK: %[[TAG:.*]] = sum.tag %[[O]]
// CHECK-NEXT: %[[SELECTED:.*]] = arith.cmpi eq, %[[TAG]], %[[ZERO]]
// CHECK-NEXT: cf.assert %[[SELECTED]], "check failed"
// CHECK-NEXT: %[[X:.*]] = sum.get %[[O]], 0
// CHECK-NEXT: %[[Y:.*]] = arith.mulf %[[X]], %[[X]]
// CHECK-NEXT: return %[[Y]]
func.func @one_live_case(%o: !O) -> f32 {
  %false = arith.constant false
  %tag = sum.tag %o : !O
  %r = scf.index_switch %tag -> f32
  case 0 {
    %x = sum.get %o, 0 : !O -> f32
    %y = arith.mulf %x, %x : f32
    scf.yield %y : f32
  }
  default {
    cf.assert %false, "check failed"
    %p = ub.poison : f32
    scf.yield %p : f32
  }
  return %r : f32
}

// CHECK-LABEL: func.func @aborting_case_and_default
// CHECK-SAME: (%[[T:.*]]: !sum.sum<(none, f32, none)>)
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
// CHECK-NOT: scf.index_switch
// CHECK: %[[TAG:.*]] = sum.tag %[[T]]
// CHECK-NEXT: %[[OTHER:.*]] = arith.cmpi ne, %[[TAG]], %[[ZERO]]
// CHECK-NEXT: cf.assert %[[OTHER]], "case zero"
// CHECK-NEXT: %[[SELECTED:.*]] = arith.cmpi eq, %[[TAG]], %[[ONE]]
// CHECK-NEXT: cf.assert %[[SELECTED]], "no case"
// CHECK-NEXT: %[[X:.*]] = sum.get %[[T]], 1
// CHECK-NEXT: return %[[X]]
func.func @aborting_case_and_default(%t: !T) -> f32 {
  %false = arith.constant false
  %tag = sum.tag %t : !T
  %r = scf.index_switch %tag -> f32
  case 0 {
    cf.assert %false, "case zero"
    %p = ub.poison : f32
    scf.yield %p : f32
  }
  case 1 {
    %x = sum.get %t, 1 : !T -> f32
    scf.yield %x : f32
  }
  default {
    cf.assert %false, "no case"
    %p = ub.poison : f32
    scf.yield %p : f32
  }
  return %r : f32
}
