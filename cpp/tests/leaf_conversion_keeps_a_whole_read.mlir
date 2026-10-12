// RUN: mlir-opt --allow-unregistered-dialect --pass-pipeline='builtin.module(leaf-conversion,leaf-conversion)' %s | FileCheck %s

// A sum crosses a call as its leaves and is rebuilt where the callee reads it
// whole, here by an operation the conversion knows nothing of. The rebuild, an
// scf.index_switch carrying the sum, is legal under the leaf conversion, so the
// conversion leaves it standing for that reader rather than flattening it into
// another rebuild: the first run converges, and the second finds nothing to do.

!S = !sum.sum<(i64, none)>
module {
  func.func private @consume(%s: !S) {
    "test.read_whole"(%s) : (!S) -> ()
    return
  }
  func.func @produce(%x: i64) {
    %s = sum.make 0 %x : !S
    func.call @consume(%s) : (!S) -> ()
    return
  }
}

// CHECK-LABEL: func.func private @consume(
// CHECK-SAME: index
// CHECK-SAME: i64
// CHECK: scf.index_switch {{.*}} -> !sum.sum<(i64, none)>
// CHECK-NOT: scf.index_switch
// CHECK: "test.read_whole"({{.*}}) : (!sum.sum<(i64, none)>) -> ()
// CHECK-LABEL: func.func @produce(
// CHECK-NOT: sum.make
// CHECK: call @consume({{.*}}) : (index, i64) -> ()
