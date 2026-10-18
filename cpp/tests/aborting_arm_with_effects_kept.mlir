// RUN: mlir-opt --canonicalize %s | FileCheck %s

// An arm that writes memory before it asserts false does not abort for the
// rewrite: the store happens before the program ends and is observable, so the
// scf.if stays as it is, the store and the false assertion inside its arm.

!O = !sum.sum<(i64, none)>

// CHECK-LABEL: func.func @store_before_the_assertion
// CHECK: scf.if
// CHECK: } else {
// CHECK-NEXT: memref.store
// CHECK-NEXT: cf.assert %false, "check failed"
func.func @store_before_the_assertion(%o: !O, %m: memref<i64>, %b: i64) -> i64 {
  %false = arith.constant false
  %some = sum.is_variant %o, 0 : !O
  %r = scf.if %some -> i64 {
    %a = sum.get %o, 0 : !O -> i64
    %s = arith.addi %a, %b : i64
    scf.yield %s : i64
  } else {
    memref.store %b, %m[] : memref<i64>
    cf.assert %false, "check failed"
    %p = ub.poison : i64
    scf.yield %p : i64
  }
  return %r : i64
}
