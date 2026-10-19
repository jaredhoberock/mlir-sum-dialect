// RUN: mlir-opt --verify-diagnostics %s | FileCheck %s --check-prefix=PARSED
// RUN: mlir-opt --canonicalize %s | FileCheck %s
// RUN: mlir-opt --pass-pipeline="builtin.module(llvm-target-to-data-layout,convert-sum-to-scf,convert-scf-to-cf,convert-to-llvm,reconcile-unrealized-casts)" %s \
// RUN: | mlir-runner --entry-point-result=void --shared-libs=%mlir_c_runner_utils \
// RUN: | FileCheck %s --check-prefix=EXEC

// A make whose tag names no variant of its sum is poison, by the dialect's
// statement of the make, and no diagnostic: the verifier reads no tag value, a
// constant one included, and nothing checks the tag at run time. Its reads fold
// to values that refine poison (a variant test of the constant tag to false).
// A branch one of whose arms makes such a sum selects no variant there, so it
// is no sum a branch selects among makes, and a match of it stays where it is.

module attributes {llvm.target = #llvm.target<triple = "x86_64-unknown-linux-gnu", chip = "">} {
  // PARSED-LABEL: func.func @constant_tag_naming_no_variant
  // CHECK-LABEL: func.func @constant_tag_naming_no_variant
  // CHECK-DAG: %[[FALSE:.*]] = arith.constant false
  // CHECK-DAG: %[[FIVE:.*]] = arith.constant 5 : index
  // CHECK-NOT: sum.
  // CHECK: return %[[FIVE]], %[[FALSE]], %[[FALSE]]
  func.func @constant_tag_naming_no_variant(%x: i64) -> (index, i1, i1) {
    %five = arith.constant 5 : index
    %s = sum.make %five (%x : i64) : !sum.sum<(i64, none)>
    %t = sum.tag %s : !sum.sum<(i64, none)>
    %a = sum.is_variant %s, 0 : !sum.sum<(i64, none)>
    %b = sum.is_variant %s, 1 : !sum.sum<(i64, none)>
    return %t, %a, %b : index, i1, i1
  }

  // CHECK-LABEL: func.func @branch_with_an_arm_naming_no_variant
  // CHECK: scf.if
  // CHECK: sum.match
  // CHECK: return
  func.func @branch_with_an_arm_naming_no_variant(%c: i1, %x: i64) -> i64 {
    %s = scf.if %c -> !sum.sum<(i64, none)> {
      %zero = arith.constant 0 : index
      %m = sum.make %zero (%x : i64) : !sum.sum<(i64, none)>
      scf.yield %m : !sum.sum<(i64, none)>
    } else {
      %five = arith.constant 5 : index
      %absent = ub.poison : i64
      %n = sum.make %five (%absent : i64) : !sum.sum<(i64, none)>
      scf.yield %n : !sum.sum<(i64, none)>
    }
    %r = sum.match %s : !sum.sum<(i64, none)> -> i64
    case 0 (%p: i64) {
      yield %p : i64
    }
    case 1 {
      %seven = arith.constant 7 : i64
      yield %seven : i64
    }
    return %r : i64
  }

  // A tag the program computes at run time, naming no variant, makes a sum the
  // program stores and goes on past.
  // EXEC: 1
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @build(%t: index, %x: i64) -> !sum.sum<(i64, none)> {
    %s = sum.make %t (%x : i64) : !sum.sum<(i64, none)>
    return %s : !sum.sum<(i64, none)>
  }
  func.func @main() {
    %two = arith.constant 2 : index
    %three = arith.constant 3 : index
    %five = arith.addi %two, %three : index
    %x = arith.constant 42 : i64
    %s = func.call @build(%five, %x) : (index, i64) -> !sum.sum<(i64, none)>
    %m = memref.alloca() : memref<!sum.sum<(i64, none)>>
    memref.store %s, %m[] : memref<!sum.sum<(i64, none)>>
    %one = arith.constant 1 : i64
    func.call @printI64(%one) : (i64) -> ()
    func.call @printNewline() : () -> ()
    return
  }
}
