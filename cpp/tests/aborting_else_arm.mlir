// RUN: mlir-opt --canonicalize %s | FileCheck %s

// An scf.if whose else arm asserts false and yields poison is the assertion of
// its condition followed by its then arm. The view the then arm builds is then
// the function's own value, so the read of its static extent folds to 8, which
// the join of the two arms hid.

!View = memref<?x?xf32, strided<[?, 1], offset: ?>>
!O = !sum.sum<(index, none)>

// CHECK-LABEL: func.func @else_arm_aborts
// CHECK-SAME: (%[[O:.*]]: !sum.sum<(index, none)>, %[[M:.*]]: memref<?xf32>)
// CHECK-DAG: %[[EIGHT:.*]] = arith.constant 8 : index
// CHECK-NOT: scf.if
// CHECK: %[[SOME:.*]] = sum.is_variant %[[O]], 0
// CHECK-NEXT: cf.assert %[[SOME]], "check failed"
// CHECK-NEXT: %[[N:.*]] = sum.get %[[O]], 0
// CHECK-NEXT: %[[VIEW:.*]] = memref.reinterpret_cast %[[M]]
// CHECK-NEXT: %[[CAST:.*]] = memref.cast %[[VIEW]]
// CHECK-NEXT: return %[[CAST]], %[[EIGHT]]
func.func @else_arm_aborts(%o: !O, %m: memref<?xf32>) -> (!View, index) {
  %false = arith.constant false
  %c1 = arith.constant 1 : index
  %some = sum.is_variant %o, 0 : !O
  %r = scf.if %some -> !View {
    %n = sum.get %o, 0 : !O -> index
    %v = memref.reinterpret_cast %m to offset: [0], sizes: [%n, 8], strides: [8, 1] : memref<?xf32> to memref<?x8xf32, strided<[8, 1]>>
    %w = memref.cast %v : memref<?x8xf32, strided<[8, 1]>> to !View
    scf.yield %w : !View
  } else {
    cf.assert %false, "check failed"
    %p = ub.poison : !View
    scf.yield %p : !View
  }
  %d = memref.dim %r, %c1 : !View
  return %r, %d : !View, index
}
