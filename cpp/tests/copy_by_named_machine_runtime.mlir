// RUN: mlir-opt --pass-pipeline="builtin.module(llvm-target-to-data-layout,convert-sum-to-scf,convert-scf-to-cf,convert-to-llvm,reconcile-unrealized-casts)" %s \
// RUN: | mlir-runner --entry-point-result=void --shared-libs=%mlir_c_runner_utils \
// RUN: | FileCheck %s

// Two sums are copied between memrefs and the second's payload is printed.
// Upstream's memref.copy lowering counts the bytes it copies through a null
// pointer, which translation folds to a constant under the layout the module
// states: the module names x86-64 and states the layout derived from it, under
// which a sum is 16 bytes, so the copy carries the second payload, 1. Naming
// no machine, the same program is translated under LLVM's default layout, a
// sum is 12 bytes, and it prints 0; nothing on this pipeline refuses that
// module, so the module's own statement is what this row checks.

// CHECK: {{^}}1{{$}}
!s = !sum.sum<(i64, none)>
module attributes {llvm.target = #llvm.target<triple = "x86_64-unknown-linux-gnu", chip = "">} {
  func.func @main() {
    %first = arith.constant 0 : index
    %second = arith.constant 1 : index
    %one = arith.constant 1 : i64
    %zero = arith.constant 0 : i64
    %v = sum.make 0 %one : !s
    %z = sum.make 0 %zero : !s
    %src = memref.alloca() : memref<2x!s>
    %dst = memref.alloca() : memref<2x!s>
    memref.store %v, %src[%first] : memref<2x!s>
    memref.store %v, %src[%second] : memref<2x!s>
    memref.store %z, %dst[%second] : memref<2x!s>
    memref.copy %src, %dst : memref<2x!s> to memref<2x!s>
    %copied = memref.load %dst[%second] : memref<2x!s>
    %payload = sum.get %copied, 0 : !s -> i64
    func.call @printI64(%payload) : (i64) -> ()
    func.call @printNewline() : () -> ()
    return
  }

  func.func private @printI64(i64)
  func.func private @printNewline()
}
