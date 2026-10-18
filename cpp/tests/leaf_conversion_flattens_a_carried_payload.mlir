// RUN: mlir-opt --pass-pipeline='builtin.module(leaf-conversion)' %s | FileCheck %s

// A function reads the payload of the payload of an Option of an Option. The
// leaf conversion takes the argument apart into leaves and rebuilds the outer
// sum from them where it is read, the inner sum rebuilt inside the arm of
// Some; the canonicalization after the conversion carries that inner sum out of
// the arm as a result of the rebuild's switch, a boundary carrying a sum, which
// the conversion then flattens in turn. No boundary carries a sum when the
// conversion is done.

!I = !sum.sum<(i64, none)>
!O = !sum.sum<(!I, none)>

// CHECK-LABEL: func.func private @f(
// CHECK-SAME: %{{[^:]+}}: index, %{{[^:]+}}: index, %{{[^:]+}}: i64) -> i64
// CHECK-NOT: -> !sum.sum
// CHECK-NOT: -> (!sum.sum
// CHECK: return
func.func private @f(%p: !O) -> i64 {
  %c7 = arith.constant 7 : i64
  %t = sum.tag %p : !O
  %r = scf.index_switch %t -> i64
  case 0 {
    %inner = sum.get %p, 0 : !O -> !I
    %some = sum.is_variant %inner, 0 : !I
    %x = scf.if %some -> (i64) {
      %v = sum.get %inner, 0 : !I -> i64
      scf.yield %v : i64
    } else {
      scf.yield %c7 : i64
    }
    scf.yield %x : i64
  }
  default {
    scf.yield %c7 : i64
  }
  return %r : i64
}
