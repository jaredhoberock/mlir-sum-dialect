// RUN: mlir-opt --pass-pipeline="builtin.module(llvm-target-to-data-layout,convert-sum-to-scf,convert-scf-to-cf,convert-to-llvm,reconcile-unrealized-casts)" %s \
// RUN: | mlir-runner --entry-point-result=void --shared-libs=%mlir_c_runner_utils \
// RUN: | FileCheck %s

// A sum of 130 variants has an i8 tag, so variant 128 sets the top bit of its
// tag. The tag reads as the variant ordinal, 128, not as the negative number the
// same bits hold signed, and the match enters the arm of variant 128, which
// receives its payload.

// CHECK: {{^}}128{{$}}
// CHECK-NEXT: {{^}}7{{$}}
// CHECK-NEXT: {{^}}128{{$}}

!s130 = !sum.sum<(none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, i64, none)>

module attributes {llvm.target = #llvm.target<triple = "x86_64-unknown-linux-gnu", chip = "">} {
  func.func @main() {
    %seven = arith.constant 7 : i64
    %s_tag = arith.constant 128 : index
    %s = sum.make %s_tag (%seven : i64) : !s130
    %tag = sum.tag %s : !s130
    %tag64 = arith.index_cast %tag : index to i64
    func.call @printI64(%tag64) : (i64) -> ()
    func.call @printNewline() : () -> ()
    %arm = sum.match %s : !s130 -> i64
    case 0 { %c = arith.constant 0 : i64 yield %c : i64 }
    case 1 { %c = arith.constant 1 : i64 yield %c : i64 }
    case 2 { %c = arith.constant 2 : i64 yield %c : i64 }
    case 3 { %c = arith.constant 3 : i64 yield %c : i64 }
    case 4 { %c = arith.constant 4 : i64 yield %c : i64 }
    case 5 { %c = arith.constant 5 : i64 yield %c : i64 }
    case 6 { %c = arith.constant 6 : i64 yield %c : i64 }
    case 7 { %c = arith.constant 7 : i64 yield %c : i64 }
    case 8 { %c = arith.constant 8 : i64 yield %c : i64 }
    case 9 { %c = arith.constant 9 : i64 yield %c : i64 }
    case 10 { %c = arith.constant 10 : i64 yield %c : i64 }
    case 11 { %c = arith.constant 11 : i64 yield %c : i64 }
    case 12 { %c = arith.constant 12 : i64 yield %c : i64 }
    case 13 { %c = arith.constant 13 : i64 yield %c : i64 }
    case 14 { %c = arith.constant 14 : i64 yield %c : i64 }
    case 15 { %c = arith.constant 15 : i64 yield %c : i64 }
    case 16 { %c = arith.constant 16 : i64 yield %c : i64 }
    case 17 { %c = arith.constant 17 : i64 yield %c : i64 }
    case 18 { %c = arith.constant 18 : i64 yield %c : i64 }
    case 19 { %c = arith.constant 19 : i64 yield %c : i64 }
    case 20 { %c = arith.constant 20 : i64 yield %c : i64 }
    case 21 { %c = arith.constant 21 : i64 yield %c : i64 }
    case 22 { %c = arith.constant 22 : i64 yield %c : i64 }
    case 23 { %c = arith.constant 23 : i64 yield %c : i64 }
    case 24 { %c = arith.constant 24 : i64 yield %c : i64 }
    case 25 { %c = arith.constant 25 : i64 yield %c : i64 }
    case 26 { %c = arith.constant 26 : i64 yield %c : i64 }
    case 27 { %c = arith.constant 27 : i64 yield %c : i64 }
    case 28 { %c = arith.constant 28 : i64 yield %c : i64 }
    case 29 { %c = arith.constant 29 : i64 yield %c : i64 }
    case 30 { %c = arith.constant 30 : i64 yield %c : i64 }
    case 31 { %c = arith.constant 31 : i64 yield %c : i64 }
    case 32 { %c = arith.constant 32 : i64 yield %c : i64 }
    case 33 { %c = arith.constant 33 : i64 yield %c : i64 }
    case 34 { %c = arith.constant 34 : i64 yield %c : i64 }
    case 35 { %c = arith.constant 35 : i64 yield %c : i64 }
    case 36 { %c = arith.constant 36 : i64 yield %c : i64 }
    case 37 { %c = arith.constant 37 : i64 yield %c : i64 }
    case 38 { %c = arith.constant 38 : i64 yield %c : i64 }
    case 39 { %c = arith.constant 39 : i64 yield %c : i64 }
    case 40 { %c = arith.constant 40 : i64 yield %c : i64 }
    case 41 { %c = arith.constant 41 : i64 yield %c : i64 }
    case 42 { %c = arith.constant 42 : i64 yield %c : i64 }
    case 43 { %c = arith.constant 43 : i64 yield %c : i64 }
    case 44 { %c = arith.constant 44 : i64 yield %c : i64 }
    case 45 { %c = arith.constant 45 : i64 yield %c : i64 }
    case 46 { %c = arith.constant 46 : i64 yield %c : i64 }
    case 47 { %c = arith.constant 47 : i64 yield %c : i64 }
    case 48 { %c = arith.constant 48 : i64 yield %c : i64 }
    case 49 { %c = arith.constant 49 : i64 yield %c : i64 }
    case 50 { %c = arith.constant 50 : i64 yield %c : i64 }
    case 51 { %c = arith.constant 51 : i64 yield %c : i64 }
    case 52 { %c = arith.constant 52 : i64 yield %c : i64 }
    case 53 { %c = arith.constant 53 : i64 yield %c : i64 }
    case 54 { %c = arith.constant 54 : i64 yield %c : i64 }
    case 55 { %c = arith.constant 55 : i64 yield %c : i64 }
    case 56 { %c = arith.constant 56 : i64 yield %c : i64 }
    case 57 { %c = arith.constant 57 : i64 yield %c : i64 }
    case 58 { %c = arith.constant 58 : i64 yield %c : i64 }
    case 59 { %c = arith.constant 59 : i64 yield %c : i64 }
    case 60 { %c = arith.constant 60 : i64 yield %c : i64 }
    case 61 { %c = arith.constant 61 : i64 yield %c : i64 }
    case 62 { %c = arith.constant 62 : i64 yield %c : i64 }
    case 63 { %c = arith.constant 63 : i64 yield %c : i64 }
    case 64 { %c = arith.constant 64 : i64 yield %c : i64 }
    case 65 { %c = arith.constant 65 : i64 yield %c : i64 }
    case 66 { %c = arith.constant 66 : i64 yield %c : i64 }
    case 67 { %c = arith.constant 67 : i64 yield %c : i64 }
    case 68 { %c = arith.constant 68 : i64 yield %c : i64 }
    case 69 { %c = arith.constant 69 : i64 yield %c : i64 }
    case 70 { %c = arith.constant 70 : i64 yield %c : i64 }
    case 71 { %c = arith.constant 71 : i64 yield %c : i64 }
    case 72 { %c = arith.constant 72 : i64 yield %c : i64 }
    case 73 { %c = arith.constant 73 : i64 yield %c : i64 }
    case 74 { %c = arith.constant 74 : i64 yield %c : i64 }
    case 75 { %c = arith.constant 75 : i64 yield %c : i64 }
    case 76 { %c = arith.constant 76 : i64 yield %c : i64 }
    case 77 { %c = arith.constant 77 : i64 yield %c : i64 }
    case 78 { %c = arith.constant 78 : i64 yield %c : i64 }
    case 79 { %c = arith.constant 79 : i64 yield %c : i64 }
    case 80 { %c = arith.constant 80 : i64 yield %c : i64 }
    case 81 { %c = arith.constant 81 : i64 yield %c : i64 }
    case 82 { %c = arith.constant 82 : i64 yield %c : i64 }
    case 83 { %c = arith.constant 83 : i64 yield %c : i64 }
    case 84 { %c = arith.constant 84 : i64 yield %c : i64 }
    case 85 { %c = arith.constant 85 : i64 yield %c : i64 }
    case 86 { %c = arith.constant 86 : i64 yield %c : i64 }
    case 87 { %c = arith.constant 87 : i64 yield %c : i64 }
    case 88 { %c = arith.constant 88 : i64 yield %c : i64 }
    case 89 { %c = arith.constant 89 : i64 yield %c : i64 }
    case 90 { %c = arith.constant 90 : i64 yield %c : i64 }
    case 91 { %c = arith.constant 91 : i64 yield %c : i64 }
    case 92 { %c = arith.constant 92 : i64 yield %c : i64 }
    case 93 { %c = arith.constant 93 : i64 yield %c : i64 }
    case 94 { %c = arith.constant 94 : i64 yield %c : i64 }
    case 95 { %c = arith.constant 95 : i64 yield %c : i64 }
    case 96 { %c = arith.constant 96 : i64 yield %c : i64 }
    case 97 { %c = arith.constant 97 : i64 yield %c : i64 }
    case 98 { %c = arith.constant 98 : i64 yield %c : i64 }
    case 99 { %c = arith.constant 99 : i64 yield %c : i64 }
    case 100 { %c = arith.constant 100 : i64 yield %c : i64 }
    case 101 { %c = arith.constant 101 : i64 yield %c : i64 }
    case 102 { %c = arith.constant 102 : i64 yield %c : i64 }
    case 103 { %c = arith.constant 103 : i64 yield %c : i64 }
    case 104 { %c = arith.constant 104 : i64 yield %c : i64 }
    case 105 { %c = arith.constant 105 : i64 yield %c : i64 }
    case 106 { %c = arith.constant 106 : i64 yield %c : i64 }
    case 107 { %c = arith.constant 107 : i64 yield %c : i64 }
    case 108 { %c = arith.constant 108 : i64 yield %c : i64 }
    case 109 { %c = arith.constant 109 : i64 yield %c : i64 }
    case 110 { %c = arith.constant 110 : i64 yield %c : i64 }
    case 111 { %c = arith.constant 111 : i64 yield %c : i64 }
    case 112 { %c = arith.constant 112 : i64 yield %c : i64 }
    case 113 { %c = arith.constant 113 : i64 yield %c : i64 }
    case 114 { %c = arith.constant 114 : i64 yield %c : i64 }
    case 115 { %c = arith.constant 115 : i64 yield %c : i64 }
    case 116 { %c = arith.constant 116 : i64 yield %c : i64 }
    case 117 { %c = arith.constant 117 : i64 yield %c : i64 }
    case 118 { %c = arith.constant 118 : i64 yield %c : i64 }
    case 119 { %c = arith.constant 119 : i64 yield %c : i64 }
    case 120 { %c = arith.constant 120 : i64 yield %c : i64 }
    case 121 { %c = arith.constant 121 : i64 yield %c : i64 }
    case 122 { %c = arith.constant 122 : i64 yield %c : i64 }
    case 123 { %c = arith.constant 123 : i64 yield %c : i64 }
    case 124 { %c = arith.constant 124 : i64 yield %c : i64 }
    case 125 { %c = arith.constant 125 : i64 yield %c : i64 }
    case 126 { %c = arith.constant 126 : i64 yield %c : i64 }
    case 127 { %c = arith.constant 127 : i64 yield %c : i64 }
    case 128 (%p: i64) {
      func.call @printI64(%p) : (i64) -> ()
      func.call @printNewline() : () -> ()
      %c = arith.constant 128 : i64
      yield %c : i64
    }
    case 129 { %c = arith.constant 129 : i64 yield %c : i64 }
    func.call @printI64(%arm) : (i64) -> ()
    func.call @printNewline() : () -> ()
    return
  }
  func.func private @printI64(i64)
  func.func private @printNewline()
}
