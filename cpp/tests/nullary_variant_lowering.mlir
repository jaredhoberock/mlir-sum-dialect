// RUN: mlir-opt --pass-pipeline="builtin.module(convert-sum-to-scf,convert-scf-to-cf,convert-to-llvm)" %s | FileCheck %s

// ---- Test: match with nullary variant lowers all the way to LLVM ----
// CHECK-LABEL: llvm.func @match_nullary_lowering
// CHECK-NOT: sum.match
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @match_nullary_lowering(%arg0: !sum.sum<(none, i64)>) -> i64 {
  %result = sum.match %arg0 : !sum.sum<(none, i64)> -> i64
  case 0 {
    %c = arith.constant 42 : i64
    yield %c : i64
  }
  case 1 (%inner: i64) {
    yield %inner : i64
  }
  return %result : i64
}

// ---- Test: make with nullary variant lowers to LLVM ----
// CHECK-LABEL: llvm.func @make_nullary_lowering
// CHECK-NOT: sum.make
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @make_nullary_lowering() -> !sum.sum<(none, i64)> {
  %c = sum.make 0 : !sum.sum<(none, i64)>
  return %c : !sum.sum<(none, i64)>
}
