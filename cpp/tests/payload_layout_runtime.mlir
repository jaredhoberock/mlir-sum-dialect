// RUN: mlir-opt --pass-pipeline="builtin.module(llvm-target-to-data-layout,convert-sum-to-scf,convert-scf-to-cf,convert-to-llvm,reconcile-unrealized-casts)" %s \
// RUN: | mlir-runner --entry-point-result=void --shared-libs=%mlir_c_runner_utils \
// RUN: | FileCheck %s

// A sum whose payload has an i64 between two narrower fields keeps every field
// through construction and a match. The payload is a field of the struct the
// sum lowers to, so LLVM lays it out -- 24 bytes on x86-64, its i64 at offset
// 8 -- and no field falls outside the sum's storage.

// CHECK: 1
// CHECK-NEXT: 7
// CHECK-NEXT: 1
module attributes {llvm.target = #llvm.target<triple = "x86_64-unknown-linux-gnu", chip = "">} {
  func.func @main() {
    %true = arith.constant true
    %seven = arith.constant 7 : i64
    %undef = llvm.mlir.undef : !llvm.struct<(i1, i64, i1)>
    %p0 = llvm.insertvalue %true, %undef[0] : !llvm.struct<(i1, i64, i1)>
    %p1 = llvm.insertvalue %seven, %p0[1] : !llvm.struct<(i1, i64, i1)>
    %p2 = llvm.insertvalue %true, %p1[2] : !llvm.struct<(i1, i64, i1)>
    %some = sum.make 0 %p2 : !sum.sum<(!llvm.struct<(i1, i64, i1)>, none)>
    sum.match %some : !sum.sum<(!llvm.struct<(i1, i64, i1)>, none)>
    case 0 (%p: !llvm.struct<(i1, i64, i1)>) {
      %a = llvm.extractvalue %p[0] : !llvm.struct<(i1, i64, i1)>
      %n = llvm.extractvalue %p[1] : !llvm.struct<(i1, i64, i1)>
      %b = llvm.extractvalue %p[2] : !llvm.struct<(i1, i64, i1)>
      %a64 = arith.extui %a : i1 to i64
      %b64 = arith.extui %b : i1 to i64
      func.call @printI64(%a64) : (i64) -> ()
      func.call @printNewline() : () -> ()
      func.call @printI64(%n) : (i64) -> ()
      func.call @printNewline() : () -> ()
      func.call @printI64(%b64) : (i64) -> ()
      func.call @printNewline() : () -> ()
      yield
    }
    case 1 {
      yield
    }
    return
  }
  func.func private @printI64(i64)
  func.func private @printNewline()
}
