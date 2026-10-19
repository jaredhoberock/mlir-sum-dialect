// RUN: mlir-opt --canonicalize %s | FileCheck %s

// A poison sum holds no tag and no payload that is not poison, so its tag, its
// variant test and its payload read each fold to a poison of their own type.

// CHECK-LABEL: func.func @reads_of_poison
// CHECK-DAG:     %[[TAG:.*]] = ub.poison : index
// CHECK-DAG:     %[[HOLDS:.*]] = ub.poison : i1
// CHECK-DAG:     %[[PAYLOAD:.*]] = ub.poison : i64
// CHECK-NOT:     sum.
// CHECK:         return %[[TAG]], %[[HOLDS]], %[[PAYLOAD]] : index, i1, i64
func.func @reads_of_poison() -> (index, i1, i64) {
  %s = ub.poison : !sum.sum<(i64, none)>
  %tag = sum.tag %s : !sum.sum<(i64, none)>
  %holds = sum.is_variant %s, 0 : !sum.sum<(i64, none)>
  %payload = sum.get %s, 0 : !sum.sum<(i64, none)> -> i64
  return %tag, %holds, %payload : index, i1, i64
}
