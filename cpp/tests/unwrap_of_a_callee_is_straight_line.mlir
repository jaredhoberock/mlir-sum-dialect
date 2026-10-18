// RUN: mlir-opt --pass-pipeline="builtin.module(canonicalize,inline,canonicalize)" %s | FileCheck %s
// RUN: mlir-opt --pass-pipeline="builtin.module(inline,canonicalize)" %s | FileCheck %s

// A callee builds an Option of a view in one arm of a branch, and its caller
// unwraps it with a switch on the tag whose default aborts. Canonicalized
// before the call is inlined, the switch is the assertion that the Option is
// Some, followed by the read of the payload; after inlining, that assertion is
// of the branch's own condition, which refutes the branch's None arm, so the
// assertion moves ahead of the Some arm and the read folds to the view the arm
// builds. The view is the caller's value as itself, with its static extent,
// built after the assertion that guards it, whichever order the rewrites meet
// it in.

!View = memref<?x?xf32, strided<[?, 1], offset: ?>>
!Opt = !sum.sum<(!View, none)>

// CHECK-LABEL: func.func @unwrapped
// CHECK-SAME: (%[[OK:.*]]: i1, %[[M:.*]]: memref<?xf32>, %[[N:.*]]: index)
// CHECK-NOT: scf.if
// CHECK-NOT: sum.
// CHECK: cf.assert %[[OK]], "unwrap on None"
// CHECK-NEXT: %[[VIEW:.*]] = memref.reinterpret_cast %[[M]] {{.*}} to memref<?x8xf32, strided<[8, 1]>>
// CHECK-NEXT: %[[CAST:.*]] = memref.cast %[[VIEW]]
// CHECK-NEXT: return %[[CAST]]
func.func private @packed(%ok: i1, %m: memref<?xf32>, %n: index) -> !Opt {
  %r = scf.if %ok -> !Opt {
    %v = memref.reinterpret_cast %m to offset: [0], sizes: [%n, 8], strides: [8, 1] : memref<?xf32> to memref<?x8xf32, strided<[8, 1]>>
    %w = memref.cast %v : memref<?x8xf32, strided<[8, 1]>> to !View
    %s = sum.make 0 %w : !Opt
    scf.yield %s : !Opt
  } else {
    %s = sum.make 1 : !Opt
    scf.yield %s : !Opt
  }
  return %r : !Opt
}

func.func @unwrapped(%ok: i1, %m: memref<?xf32>, %n: index) -> !View {
  %false = arith.constant false
  %o = func.call @packed(%ok, %m, %n) : (i1, memref<?xf32>, index) -> !Opt
  %t = sum.tag %o : !Opt
  %p = scf.index_switch %t -> !View
  case 0 {
    %x = sum.get %o, 0 : !Opt -> !View
    scf.yield %x : !View
  }
  default {
    cf.assert %false, "unwrap on None"
    %q = ub.poison : !View
    scf.yield %q : !View
  }
  return %p : !View
}
