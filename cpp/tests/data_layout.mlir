// RUN: mlir-opt --test-data-layout-query %s | FileCheck %s

// A sum answers data-layout queries with the layout of the struct it lowers to
// -- the tag, then each payload in its own field -- measured by the layout the
// module states, here x86-64's. Each sum query is paired with that struct, and
// the two report the same alignment, bit size, preferred alignment, and size:
// the i64 between two i1 fields puts the payload at 24 bytes and 8-byte
// alignment, so the sum is 32 bytes.

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>} {
  func.func @layout() {
    // CHECK: alignment = 8
    // CHECK-SAME: bitsize = 256
    // CHECK-SAME: preferred = 8
    // CHECK-SAME: size = 32
    // CHECK-SAME: sum.sum
    "test.data_layout_query"() : () -> !sum.sum<(!llvm.struct<(i1, i64, i1)>, none)>
    // CHECK: alignment = 8
    // CHECK-SAME: bitsize = 256
    // CHECK-SAME: preferred = 8
    // CHECK-SAME: size = 32
    // CHECK-SAME: llvm.struct<(i8, struct<(i1, i64, i1)>)>
    "test.data_layout_query"() : () -> !llvm.struct<(i8, struct<(i1, i64, i1)>)>
    return
  }
}
