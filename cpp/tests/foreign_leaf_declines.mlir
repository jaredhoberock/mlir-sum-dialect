// A sum whose variant carries a type no loaded dialect lowers to LLVM is not the
// sum converter's to rewrite: it declines rather than aborting, leaving the
// function's signature standing for the step that owns the inner type. Here the
// inner `tuple` has no LLVM conversion without the tuple dialect loaded, so the
// sum stays and only the body's terminator lowers. A converter that reached past
// the sum to force the inner type would abort here instead.

// RUN: mlir-opt --pass-pipeline="builtin.module(convert-to-llvm)" %s | FileCheck %s

module {
  // CHECK: func.func private @sum_of_foreign_leaf
  // CHECK-SAME: !sum.sum<(tuple<i64, i64>, none)>
  func.func private @sum_of_foreign_leaf(%a: !sum.sum<(tuple<i64, i64>, none)>) {
    return
  }
}
