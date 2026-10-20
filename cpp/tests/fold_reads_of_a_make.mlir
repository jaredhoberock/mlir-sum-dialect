// RUN: mlir-opt --canonicalize %s | FileCheck %s

// A read of a make folds to what it reads: the tag is its tag operand,
// is_variant compares that tag with the variant (a constant tag answers it),
// and get of every variant reads that variant's payload operand, whatever the
// tag: where the make holds another variant the read is poison, and the
// operand, the poison its make site supplies, refines it. A make of a sum's own
// tag and its own payloads is the sum.

!S = !sum.sum<(i64, none, f32)>
!M = memref<?xf32>
!B = !sum.sum<(!M, none)>

// CHECK-LABEL: func.func @reads_of_a_make
// CHECK-SAME: (%[[X:.*]]: i64)
// CHECK-DAG: %[[TAG:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[ABSENT:.*]] = ub.poison : f32
// CHECK-DAG: %[[TRUE:.*]] = arith.constant true
// CHECK-DAG: %[[FALSE:.*]] = arith.constant false
// CHECK-NOT: sum.
// CHECK: return %[[TAG]], %[[TRUE]], %[[FALSE]], %[[X]], %[[ABSENT]]
func.func @reads_of_a_make(%x: i64) -> (index, i1, i1, i64, f32) {
  %s_tag = arith.constant 0 : index
  %s_absent = ub.poison : f32
  %s = sum.make %s_tag (%x, %s_absent : i64, f32) : !S
  %t = sum.tag %s : !S
  %a = sum.is_variant %s, 0 : !S
  %b = sum.is_variant %s, 1 : !S
  %p = sum.get %s, 0 : !S -> i64
  %q = sum.get %s, 2 : !S -> f32
  return %t, %a, %b, %p, %q : index, i1, i1, i64, f32
}

// CHECK-LABEL: func.func @reads_of_a_make_of_a_dynamic_tag
// CHECK-SAME: (%[[TAG:.*]]: index, %[[X:.*]]: i64, %[[Y:.*]]: f32)
// CHECK-DAG: %[[FIRST:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[LAST:.*]] = arith.constant 2 : index
// CHECK: %[[A:.*]] = arith.cmpi eq, %[[TAG]], %[[FIRST]]
// CHECK: %[[B:.*]] = arith.cmpi eq, %[[TAG]], %[[LAST]]
// CHECK-NOT: sum.
// CHECK: return %[[TAG]], %[[A]], %[[B]], %[[X]], %[[Y]]
func.func @reads_of_a_make_of_a_dynamic_tag(%tag: index, %x: i64, %y: f32) -> (index, i1, i1, i64, f32) {
  %s = sum.make %tag (%x, %y : i64, f32) : !S
  %t = sum.tag %s : !S
  %a = sum.is_variant %s, 0 : !S
  %b = sum.is_variant %s, 2 : !S
  %p = sum.get %s, 0 : !S -> i64
  %q = sum.get %s, 2 : !S -> f32
  return %t, %a, %b, %p, %q : index, i1, i1, i64, f32
}

// The payloads taken apart as the leaf conversion takes them apart: each the
// read of the sum's own payload.
// CHECK-LABEL: func.func @make_of_its_own_parts
// CHECK-SAME: (%[[S:.*]]: !sum.sum<(memref<?xf32>, none)>)
// CHECK-NOT: sum.make
// CHECK: return %[[S]]
func.func @make_of_its_own_parts(%s: !B) -> !B {
  %t = sum.tag %s : !B
  %p = sum.get %s, 0 : !B -> !M
  %read = sum.make %t (%p : !M) : !B
  return %read : !B
}

// The tag of one sum with the payload of another is not the sum.
// CHECK-LABEL: func.func @make_of_other_parts
// CHECK: sum.make
// CHECK: return
func.func @make_of_other_parts(%s: !B, %r: !B) -> !B {
  %t = sum.tag %s : !B
  %p = sum.get %r, 0 : !B -> !M
  %mixed = sum.make %t (%p : !M) : !B
  return %mixed : !B
}
