// RUN: mlir-opt --pass-pipeline="builtin.module(convert-to-llvm)" %s | FileCheck %s

// A sum whose variant is a memref lowers all the way to LLVM: the payload field
// holds the memref's LLVM descriptor struct, converted as the lowering converts
// it.

// CHECK-LABEL: llvm.func @wrap
// CHECK-NOT: sum.make
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
module {
  func.func @wrap(%x: memref<?xi64, strided<[1], offset: ?>>)
      -> !sum.sum<(memref<?xi64, strided<[1], offset: ?>>, none)> {
    %s_tag = arith.constant 0 : index
    %s = sum.make %s_tag (%x : memref<?xi64, strided<[1], offset: ?>>) : !sum.sum<(memref<?xi64, strided<[1], offset: ?>>, none)>
    return %s : !sum.sum<(memref<?xi64, strided<[1], offset: ?>>, none)>
  }

  // A memref whose element is itself a sum lowers the same way: its descriptor
  // is converted with every loaded dialect's conversion, so the sum element
  // converts as the lowering converts it.
  // CHECK-LABEL: llvm.func @wrap_memref_of_sums
  // CHECK-NOT: sum.make
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: llvm.return
  func.func @wrap_memref_of_sums(%x: memref<4x!sum.sum<(i64, none)>>)
      -> !sum.sum<(memref<4x!sum.sum<(i64, none)>>, none)> {
    %s_tag = arith.constant 0 : index
    %s = sum.make %s_tag (%x : memref<4x!sum.sum<(i64, none)>>) : !sum.sum<(memref<4x!sum.sum<(i64, none)>>, none)>
    return %s : !sum.sum<(memref<4x!sum.sum<(i64, none)>>, none)>
  }
}
