// RUN: mlir-opt --canonicalize %s | FileCheck %s

// An arm that asserts a condition before it asserts false does not abort for
// the rewrite: when that condition fails, the program ends at the arm's first
// assertion with its message, which the assertion of the branch's condition
// would not carry, so the scf.if stays as it is, both assertions inside its arm.

!O = !sum.sum<(i64, none)>

// CHECK-LABEL: func.func @assertion_before_the_false_assertion
// CHECK: scf.if
// CHECK: } else {
// CHECK-NEXT: cf.assert %{{.+}}, "first check failed"
// CHECK-NEXT: cf.assert %false, "second check failed"
func.func @assertion_before_the_false_assertion(%o: !O, %d: i1, %b: i64) -> i64 {
  %false = arith.constant false
  %some = sum.is_variant %o, 0 : !O
  %r = scf.if %some -> i64 {
    %a = sum.get %o, 0 : !O -> i64
    %s = arith.addi %a, %b : i64
    scf.yield %s : i64
  } else {
    cf.assert %d, "first check failed"
    cf.assert %false, "second check failed"
    %p = ub.poison : i64
    scf.yield %p : i64
  }
  return %r : i64
}
