// RUN: mlir-opt --canonicalize %s | FileCheck %s

// A read of the payload of variant k of a sum a branch selects among makes,
// where the make of k builds its payload inside its arm, becomes a result of
// the branch: the arm that makes k yields the payload, and every other arm
// yields poison, the value the read has where the sum holds another variant.
// The read no longer holds the sum, so a sum no other reader holds is not
// built. Two reads of one payload read one result. The arms here read memory,
// so each branch stays a branch.

!O = !sum.sum<(i64, none)>
!T = !sum.sum<(i64, f32, none)>

// CHECK-LABEL: func.func @payload_of_an_if
// CHECK-SAME: (%[[C:.*]]: i1, %[[M:.*]]: memref<i64>)
// CHECK: %[[POISON:.*]] = ub.poison : i64
// CHECK: %[[P:.*]] = scf.if %[[C]] -> (i64) {
// CHECK-NEXT: %[[V:.*]] = memref.load %[[M]]
// CHECK-NEXT: scf.yield %[[V]]
// CHECK-NEXT: } else {
// CHECK-NEXT: scf.yield %[[POISON]]
// CHECK-NOT: sum.
// CHECK: return %[[P]], %[[P]]
func.func @payload_of_an_if(%c: i1, %m: memref<i64>) -> (i64, i64) {
  %s = scf.if %c -> !O {
    %v = memref.load %m[] : memref<i64>
    %made = sum.make 0 %v : !O
    scf.yield %made : !O
  } else {
    %made = sum.make 1 : !O
    scf.yield %made : !O
  }
  %p = sum.get %s, 0 : !O -> i64
  %q = sum.get %s, 0 : !O -> i64
  return %p, %q : i64, i64
}

// CHECK-LABEL: func.func @payload_of_a_switch
// CHECK-SAME: (%[[TAG:.*]]: index, %[[M:.*]]: memref<f32>, %[[X:.*]]: i64)
// CHECK: %[[POISON:.*]] = ub.poison : f32
// CHECK: %[[P:.*]] = scf.index_switch %[[TAG]] -> f32
// CHECK-NEXT: case 0 {
// CHECK-NEXT: scf.yield %[[POISON]]
// CHECK: case 1 {
// CHECK-NEXT: %[[V:.*]] = memref.load %[[M]]
// CHECK-NEXT: scf.yield %[[V]]
// CHECK: default {
// CHECK-NEXT: scf.yield %[[POISON]]
// CHECK-NOT: sum.
// CHECK: return %[[P]]
func.func @payload_of_a_switch(%tag: index, %m: memref<f32>, %x: i64) -> f32 {
  %s = scf.index_switch %tag -> !T
  case 0 {
    %made = sum.make 0 %x : !T
    scf.yield %made : !T
  }
  case 1 {
    %v = memref.load %m[] : memref<f32>
    %made = sum.make 1 %v : !T
    scf.yield %made : !T
  }
  default {
    %made = sum.make 2 : !T
    scf.yield %made : !T
  }
  %p = sum.get %s, 1 : !T -> f32
  return %p : f32
}
