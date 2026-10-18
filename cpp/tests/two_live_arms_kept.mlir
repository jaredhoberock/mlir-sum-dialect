// RUN: mlir-opt --canonicalize %s | FileCheck %s

// An scf.if neither of whose arms aborts joins two values and stays as it is,
// as does an scf.if with an absent else arm whose then arm does not abort.

!O = !sum.sum<(i64, none)>

// CHECK-LABEL: func.func @two_live_arms
// CHECK-NOT: cf.assert
// CHECK: scf.if
// CHECK: memref.store
// CHECK: } else {
// CHECK: memref.load
// CHECK: }
// CHECK-NOT: cf.assert
func.func @two_live_arms(%o: !O, %m: memref<i64>, %a: i64) -> i64 {
  %some = sum.is_variant %o, 0 : !O
  %r = scf.if %some -> i64 {
    memref.store %a, %m[] : memref<i64>
    scf.yield %a : i64
  } else {
    %v = memref.load %m[] : memref<i64>
    scf.yield %v : i64
  }
  return %r : i64
}

// CHECK-LABEL: func.func @no_else_arm
// CHECK-NOT: cf.assert
// CHECK: scf.if
// CHECK-NEXT: memref.store
func.func @no_else_arm(%o: !O, %m: memref<i64>, %a: i64) {
  %some = sum.is_variant %o, 0 : !O
  scf.if %some {
    memref.store %a, %m[] : memref<i64>
  }
  return
}
