// RUN: mlir-opt --pass-pipeline="builtin.module(convert-to-llvm)" %s | FileCheck %s

// ---- A sum op lowered from a non-entry block allocates in the function
// entry block. The alloca must precede the branch that leads out of the entry
// block, and nothing behind that branch may allocate; a non-entry alloca would
// be a dynamic alloca that bars inlining.
// CHECK-LABEL: llvm.func @make_in_second_block
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.alloca
// CHECK: llvm.br
// CHECK-NOT: llvm.alloca
// CHECK: llvm.return
// CHECK-NOT: llvm.alloca
func.func @make_in_second_block(%arg0: i64) -> !sum.sum<(i64, i32)> {
  cf.br ^bb1
^bb1:
  %c = sum.make 0 %arg0 : !sum.sum<(i64, i32)>
  return %c : !sum.sum<(i64, i32)>
}
