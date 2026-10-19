// RUN: mlir-opt --pass-pipeline='builtin.module(leaf-conversion)' %s | FileCheck %s

// A switch whose arm makes a sum is the leaf conversion's own rebuild only when
// every arm rebuilds from values defined outside it, by operations free of memory
// effects. Here case 0 allocates and fills the buffer it makes, so the switch
// computes its payload rather than rebuilding it: the conversion flattens it like
// any switch carrying a sum, the buffer crossing as one of its results, rather
// than keeping it whole with the allocation hidden inside the sum.

!S = !sum.sum<(memref<1xi64>, none)>
module {
  func.func @f(%tag: index) -> i64 {
    %c0 = arith.constant 0 : index
    %c42 = arith.constant 42 : i64
    %s = scf.index_switch %tag -> !S
    case 0 {
      %m = memref.alloc() : memref<1xi64>
      memref.store %c42, %m[%c0] : memref<1xi64>
      %v_tag = arith.constant 0 : index
      %v = sum.make %v_tag (%m : memref<1xi64>) : !S
      scf.yield %v : !S
    }
    default {
      %v_tag = arith.constant 1 : index
      %v_absent = ub.poison : memref<1xi64>
      %v = sum.make %v_tag (%v_absent : memref<1xi64>) : !S
      scf.yield %v : !S
    }
    %m = sum.get %s, 0 : !S -> memref<1xi64>
    %v = memref.load %m[%c0] : memref<1xi64>
    return %v : i64
  }
}

// CHECK-LABEL: func.func @f(
// CHECK-NOT: -> !sum.sum
// CHECK: scf.index_switch {{.*}} -> memref<1xi64>
// CHECK: memref.alloc
// CHECK-NOT: -> !sum.sum
// CHECK-NOT: sum.get
// CHECK: memref.load
// CHECK: return
