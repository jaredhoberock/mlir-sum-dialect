// The rewrite of a branch with an aborting arm keeps the program's behavior on
// both of the branch's ways. Taken, the live arm stores 7 and yields it, after
// the assertion of its condition passes; not taken, the program aborts at that
// assertion, where it aborted at the arm's own.

// RUN: mlir-opt --pass-pipeline="builtin.module(llvm-target-to-data-layout,canonicalize,convert-scf-to-cf,finalize-memref-to-llvm,convert-to-llvm,reconcile-unrealized-casts)" %s \
// RUN: | mlir-runner -e taken --entry-point-result=void --shared-libs=%mlir_c_runner_utils \
// RUN: | FileCheck %s
// RUN: mlir-opt --pass-pipeline="builtin.module(llvm-target-to-data-layout,canonicalize,convert-scf-to-cf,finalize-memref-to-llvm,convert-to-llvm,reconcile-unrealized-casts)" %s \
// RUN: | not --crash mlir-runner -e aborts --entry-point-result=void --shared-libs=%mlir_c_runner_utils

// CHECK: 7
// CHECK-NEXT: 7

!O = !sum.sum<(i64, none)>

module attributes {llvm.target = #llvm.target<triple = "x86_64-unknown-linux-gnu", chip = "">} {
  func.func private @printI64(i64)
  func.func private @printNewline()

  func.func @pick(%o: !O, %m: memref<i64>) -> i64 {
    %false = arith.constant false
    %some = sum.is_variant %o, 0 : !O
    %r = scf.if %some -> i64 {
      %seven = sum.get %o, 0 : !O -> i64
      memref.store %seven, %m[] : memref<i64>
      scf.yield %seven : i64
    } else {
      cf.assert %false, "check failed"
      %p = ub.poison : i64
      scf.yield %p : i64
    }
    return %r : i64
  }

  func.func @run(%o: !O) {
    %m = memref.alloca() : memref<i64>
    %zero = arith.constant 0 : i64
    memref.store %zero, %m[] : memref<i64>
    %r = func.call @pick(%o, %m) : (!O, memref<i64>) -> i64
    func.call @printI64(%r) : (i64) -> ()
    func.call @printNewline() : () -> ()
    %stored = memref.load %m[] : memref<i64>
    func.call @printI64(%stored) : (i64) -> ()
    func.call @printNewline() : () -> ()
    return
  }

  func.func @taken() {
    %seven = arith.constant 7 : i64
    %some = sum.make 0 %seven : !O
    func.call @run(%some) : (!O) -> ()
    return
  }

  func.func @aborts() {
    %none = sum.make 1 : !O
    func.call @run(%none) : (!O) -> ()
    return
  }
}
