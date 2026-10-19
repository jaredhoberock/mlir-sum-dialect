// RUN: mlir-opt -inline %s | FileCheck %s

// MLIR's inliner clones a callee only when every op in it belongs to a dialect
// registering an inliner interface. A function holding a sum op is inlinable only
// because the sum dialect registers one; without it the call below survives.
// After -inline @caller holds the sum.make directly and no call remains.

// CHECK-LABEL: func @caller
// CHECK-NOT: call
// CHECK: sum.make
// CHECK: return
func.func @caller(%arg0: i64) -> !sum.sum<(i64, tuple<>)> {
  %r = call @callee(%arg0) : (i64) -> !sum.sum<(i64, tuple<>)>
  return %r : !sum.sum<(i64, tuple<>)>
}

func.func private @callee(%arg0: i64) -> !sum.sum<(i64, tuple<>)> {
  %c_tag = arith.constant 0 : index
  %c_absent = ub.poison : tuple<>
  %c = sum.make %c_tag (%arg0, %c_absent : i64, tuple<>) : !sum.sum<(i64, tuple<>)>
  return %c : !sum.sum<(i64, tuple<>)>
}
