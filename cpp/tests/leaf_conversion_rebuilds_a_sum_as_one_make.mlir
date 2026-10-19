// RUN: mlir-opt --pass-pipeline='builtin.module(leaf-conversion,leaf-conversion)' %s | FileCheck %s --check-prefix=IR
// RUN: mlir-opt --pass-pipeline="builtin.module(leaf-conversion,llvm-target-to-data-layout,convert-sum-to-scf,convert-scf-to-cf,finalize-memref-to-llvm,convert-to-llvm,reconcile-unrealized-casts)" %s \
// RUN: | mlir-runner --entry-point-result=void --shared-libs=%mlir_c_runner_utils \
// RUN: | FileCheck %s

// A sum crosses a call as its leaves, and the callee stores it whole, by an
// operation the conversion knows nothing of. The sum is rebuilt there as one
// sum.make of its tag and payload leaves, an operation that carries the sum
// and is no boundary, so the first run of the conversion leaves no boundary
// carrying a sum and the second finds nothing to do. The make survives to
// the LLVM lowering, and the caller, loading the sum back, reads the payload
// under the variant it holds: 42 where Some, 7 where None.

// IR-LABEL: func.func private @store(
// IR-SAME: %[[TAG:[^:]+]]: index, %[[PAYLOAD:[^:]+]]: i64, %[[M:[^:]+]]: memref<!sum.sum<(i64, none)>>)
// IR-NEXT: %[[S:.*]] = sum.make %[[TAG]] (%[[PAYLOAD]] : i64) : !sum.sum<(i64, none)>
// IR-NEXT: memref.store %[[S]], %[[M]][]
// IR-NOT: scf.index_switch
// IR-LABEL: func.func @main(
// IR-NOT: scf.index_switch
// IR: call @store(%{{.*}}, %{{.*}}, %{{.*}}) : (index, i64, memref<!sum.sum<(i64, none)>>) -> ()

// CHECK: 42
// CHECK-NEXT: 7

!S = !sum.sum<(i64, none)>
module attributes {llvm.target = #llvm.target<triple = "x86_64-unknown-linux-gnu", chip = "">} {
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @store(%s: !S, %m: memref<!S>) {
    memref.store %s, %m[] : memref<!S>
    return
  }
  func.func private @read(%m: memref<!S>) -> i64 {
    %c7 = arith.constant 7 : i64
    %s = memref.load %m[] : memref<!S>
    %some = sum.is_variant %s, 0 : !S
    %r = scf.if %some -> i64 {
      %p = sum.get %s, 0 : !S -> i64
      scf.yield %p : i64
    } else {
      scf.yield %c7 : i64
    }
    return %r : i64
  }
  func.func @main() {
    %c42 = arith.constant 42 : i64
    %m = memref.alloca() : memref<!S>
    %some_tag = arith.constant 0 : index
    %some = sum.make %some_tag (%c42 : i64) : !S
    func.call @store(%some, %m) : (!S, memref<!S>) -> ()
    %a = func.call @read(%m) : (memref<!S>) -> i64
    func.call @printI64(%a) : (i64) -> ()
    func.call @printNewline() : () -> ()
    %none_tag = arith.constant 1 : index
    %none_absent = ub.poison : i64
    %none = sum.make %none_tag (%none_absent : i64) : !S
    func.call @store(%none, %m) : (!S, memref<!S>) -> ()
    %b = func.call @read(%m) : (memref<!S>) -> i64
    func.call @printI64(%b) : (i64) -> ()
    func.call @printNewline() : () -> ()
    return
  }
}
