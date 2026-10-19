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

// An arm whose loop may never end before it asserts false does not abort for
// the rewrite either: a run that takes the arm may never reach the assertion,
// and the rewrite would end it there at once. The scf.if stays as it is, the
// loop and the false assertion inside its arm. (The loop's result is yielded,
// so the loop is not dead.)

// CHECK-LABEL: func.func @endless_loop_before_the_assertion
// CHECK: scf.if
// CHECK: } else {
// CHECK-NEXT: scf.while
// CHECK: cf.assert %false, "check failed"
func.func @endless_loop_before_the_assertion(%o: !O, %b: i64, %go: i1) -> (i64, i64) {
  %false = arith.constant false
  %one = arith.constant 1 : i64
  %some = sum.is_variant %o, 0 : !O
  %r, %q = scf.if %some -> (i64, i64) {
    %a = sum.get %o, 0 : !O -> i64
    %s = arith.addi %a, %b : i64
    scf.yield %s, %b : i64, i64
  } else {
    %n = scf.while (%x = %b) : (i64) -> i64 {
      scf.condition(%go) %x : i64
    } do {
    ^bb0(%x: i64):
      %y = arith.addi %x, %one : i64
      scf.yield %y : i64
    }
    cf.assert %false, "check failed"
    %p = ub.poison : i64
    scf.yield %p, %n : i64, i64
  }
  return %r, %q : i64, i64
}
