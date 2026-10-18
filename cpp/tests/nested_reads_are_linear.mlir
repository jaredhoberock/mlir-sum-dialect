// RUN: mlir-opt --pass-pipeline='builtin.module(leaf-conversion)' %s | FileCheck %s

// A function unwraps four nested Options, each read checked by an assertion.
// The leaf conversion takes the argument apart into its five leaves and
// rebuilds it as nested assembles, every read of which is a projection: each
// check folds to a comparison of one tag leaf, and the innermost payload is the
// last leaf itself. The work is one operation per leaf, however deeply the sums
// nest, and no branch is built.

!T0 = i64
!T1 = !sum.sum<(!T0, none)>
!T2 = !sum.sum<(!T1, none)>
!T3 = !sum.sum<(!T2, none)>
!T4 = !sum.sum<(!T3, none)>

// CHECK-LABEL: func.func @f(
// CHECK-SAME: %{{[^:]+}}: index, %{{[^:]+}}: index, %{{[^:]+}}: index, %{{[^:]+}}: index, %[[X:[^:]+]]: i64) -> i64
// CHECK-NOT: scf.index_switch
// CHECK-NOT: sum.
// CHECK-COUNT-4: cf.assert
// CHECK-NOT: scf.index_switch
// CHECK-NOT: sum.
// CHECK: return %[[X]] : i64
func.func @f(%v4: !T4) -> i64 {
  %c4 = sum.is_variant %v4, 0 : !T4
  cf.assert %c4, "None"
  %v3 = sum.get %v4, 0 : !T4 -> !T3
  %c3 = sum.is_variant %v3, 0 : !T3
  cf.assert %c3, "None"
  %v2 = sum.get %v3, 0 : !T3 -> !T2
  %c2 = sum.is_variant %v2, 0 : !T2
  cf.assert %c2, "None"
  %v1 = sum.get %v2, 0 : !T2 -> !T1
  %c1 = sum.is_variant %v1, 0 : !T1
  cf.assert %c1, "None"
  %v0 = sum.get %v1, 0 : !T1 -> !T0
  return %v0 : i64
}
