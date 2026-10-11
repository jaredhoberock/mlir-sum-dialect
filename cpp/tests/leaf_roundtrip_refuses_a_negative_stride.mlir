// RUN: not mlir-opt --pass-pipeline="builtin.module(test-sum-leaf-roundtrip)" %s 2>&1 | FileCheck %s

// A buffer payload with a negative stride has no absent stand-in: the stand-in's
// storage is sized for strides that do not step backward, so a sum holding such a
// buffer does not cross as its leaves, and the conversion says so rather than
// handing out a view that reaches past its storage.

// CHECK: error: a returned '!sum.sum<(memref<4xi64, strided<[-1], offset: 3>>, none)>' does not cross as its leaves
!S = !sum.sum<(memref<4xi64, strided<[-1], offset: 3>>, none)>
module {
  func.func @f(%s: !S) -> !S {
    return %s : !S
  }
}
