// RUN: mlir-opt --test-data-layout-query %s | FileCheck %s

// A sum's layout is the layout of the LLVM struct it lowers to, so a query
// builds LLVM types, and a dialect first loaded inside a pass aborts. The LLVM
// dialect loads with the sum dialect, so a module that names no LLVM type is
// measured: the i64 payload follows the tag at 8 bytes, and the sum is 16.

// CHECK: alignment = 8
// CHECK-SAME: size = 16
// CHECK-SAME: sum.sum
module attributes {dlti.dl_spec = #dlti.dl_spec<i64 = dense<64> : vector<2xi64>>} {
  func.func @layout() {
    "test.data_layout_query"() : () -> !sum.sum<(i64, none)>
    return
  }
}
