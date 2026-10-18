// RUN: mlir-opt --canonicalize %s | FileCheck %s

// An scf.if one of whose arms yields only poison and whose other arm is pure is
// that other arm: where the if takes the poison arm its results are poison,
// which the pure arm's values refine, and the pure arm, free of memory effects
// and speculatable, may run there. A value the pure arm computes then reaches
// the if's readers as itself. An arm that reads memory is not pure, and a poison
// arm that writes memory does something, so either keeps the if.

!View = memref<?x?xf32, strided<[?, 1], offset: ?>>
!O = !sum.sum<(i64, none)>

// The read of an Option's payload, poison where it holds None.
// CHECK-LABEL: func.func @payload_where_some
// CHECK-SAME: (%[[O:.*]]: !sum.sum<(i64, none)>)
// CHECK-NOT: scf.if
// CHECK: %[[P:.*]] = sum.get %[[O]], 0
// CHECK-NEXT: %[[Q:.*]] = arith.muli %[[P]], %[[P]]
// CHECK-NEXT: return %[[Q]]
func.func @payload_where_some(%o: !O) -> i64 {
  %some = sum.is_variant %o, 0 : !O
  %r = scf.if %some -> i64 {
    %p = sum.get %o, 0 : !O -> i64
    %q = arith.muli %p, %p : i64
    scf.yield %q : i64
  } else {
    %u = ub.poison : i64
    scf.yield %u : i64
  }
  return %r : i64
}

// An Option of a view taken apart into its tag and its payload, the payload
// built in the arm of Some and poison in the arm of None: the tag is a select,
// and the view, with its static extent, is the function's own value.
// CHECK-LABEL: func.func @view_where_some
// CHECK-SAME: (%[[OK:.*]]: i1, %[[M:.*]]: memref<?xf32>, %[[N:.*]]: index)
// CHECK-NOT: scf.if
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast %[[M]] {{.*}} to memref<?x8xf32, strided<[8, 1]>>
// CHECK-NEXT: %[[CAST:.*]] = memref.cast %[[VIEW]]
// CHECK: cf.assert %{{.*}}, "unwrap on None"
// CHECK-NEXT: return %[[CAST]]
func.func @view_where_some(%ok: i1, %m: memref<?xf32>, %n: index) -> !View {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r:2 = scf.if %ok -> (index, !View) {
    %v = memref.reinterpret_cast %m to offset: [0], sizes: [%n, 8], strides: [8, 1] : memref<?xf32> to memref<?x8xf32, strided<[8, 1]>>
    %w = memref.cast %v : memref<?x8xf32, strided<[8, 1]>> to !View
    scf.yield %c0, %w : index, !View
  } else {
    %p = ub.poison : !View
    scf.yield %c1, %p : index, !View
  }
  %isok = arith.cmpi eq, %r#0, %c0 : index
  cf.assert %isok, "unwrap on None"
  return %r#1 : !View
}

// The poison arm first: the else arm is the if.
// CHECK-LABEL: func.func @poison_then_arm
// CHECK-SAME: (%[[C:.*]]: i1, %[[A:.*]]: i64, %[[B:.*]]: i64)
// CHECK-NOT: scf.if
// CHECK: %[[S:.*]] = arith.addi %[[A]], %[[B]]
// CHECK-NEXT: return %[[S]]
func.func @poison_then_arm(%c: i1, %a: i64, %b: i64) -> i64 {
  %r = scf.if %c -> i64 {
    %u = ub.poison : i64
    scf.yield %u : i64
  } else {
    %s = arith.addi %a, %b : i64
    scf.yield %s : i64
  }
  return %r : i64
}

// CHECK-LABEL: func.func @live_arm_reads_memory
// CHECK: scf.if
// CHECK-NEXT: memref.load
// CHECK: } else {
func.func @live_arm_reads_memory(%c: i1, %m: memref<i64>) -> i64 {
  %r = scf.if %c -> i64 {
    %v = memref.load %m[] : memref<i64>
    scf.yield %v : i64
  } else {
    %u = ub.poison : i64
    scf.yield %u : i64
  }
  return %r : i64
}

// CHECK-LABEL: func.func @poison_arm_writes_memory
// CHECK: scf.if
// CHECK: } else {
// CHECK-NEXT: memref.store
func.func @poison_arm_writes_memory(%c: i1, %m: memref<i64>, %a: i64, %b: i64) -> i64 {
  %r = scf.if %c -> i64 {
    %s = arith.addi %a, %b : i64
    scf.yield %s : i64
  } else {
    memref.store %a, %m[] : memref<i64>
    %u = ub.poison : i64
    scf.yield %u : i64
  }
  return %r : i64
}
