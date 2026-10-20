// RUN: mlir-opt --pass-pipeline="builtin.module(leaf-conversion,inline,canonicalize)" %s | FileCheck %s
// RUN: mlir-opt --pass-pipeline="builtin.module(inline,leaf-conversion)" %s | FileCheck %s

// A callee builds an Option of a view in one arm of a branch, and its caller
// unwraps it with a switch on the tag whose default aborts. The flattening
// takes the Option apart, at the call's boundary or, once the call is inlined,
// at the branch: its tag is a choice between two constants on the branch's
// condition, so the switch is the assertion of that condition, and its payload
// is the view the Some arm builds, the None arm's poison folded away. The view
// is the caller's value as itself, with its static extent, built after the
// assertion that guards it, whichever order the flattening and the inlining
// meet it in.

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
    %s_tag = arith.constant 0 : index
    %s = sum.make %s_tag (%w : !View) : !Opt
    scf.yield %s : !Opt
  } else {
    %s_tag = arith.constant 1 : index
    %s_absent = ub.poison : !View
    %s = sum.make %s_tag (%s_absent : !View) : !Opt
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
