// RUN: mlir-opt --pass-pipeline="builtin.module(convert-to-llvm)" %s | FileCheck %s

// A sum whose variant is a memref lowers all the way to LLVM: the payload is
// sized and aligned from the memref's LLVM descriptor struct, not from a data
// layout query on the memref type itself, which carries no layout of its own and
// would abort the conversion. The 1-D descriptor is five index-sized words, so
// the payload blob is 40 bytes and the tag widens to the descriptor's 8-byte
// alignment.

// CHECK-LABEL: llvm.func @wrap
// CHECK-NOT: sum.make
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: !llvm.struct<(i64, array<40 x i8>)>
// CHECK: llvm.return
module {
  func.func @wrap(%x: memref<?xi64, strided<[1], offset: ?>>)
      -> !sum.sum<(memref<?xi64, strided<[1], offset: ?>>, none)> {
    %s = sum.make 0 %x : !sum.sum<(memref<?xi64, strided<[1], offset: ?>>, none)>
    return %s : !sum.sum<(memref<?xi64, strided<[1], offset: ?>>, none)>
  }
}
