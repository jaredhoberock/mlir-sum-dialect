// RUN: mlir-opt %s -verify-diagnostics

// ---- Test: sum.get on nullary variant should fail ----
func.func @get_nullary(%arg0: !sum.sum<(none, i64)>) -> none {
  // expected-error @+1 {{cannot extract payload from nullary variant 0}}
  %v = sum.get %arg0, 0 : !sum.sum<(none, i64)> -> none
  return %v : none
}
