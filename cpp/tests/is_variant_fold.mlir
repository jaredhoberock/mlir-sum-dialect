// RUN: mlir-opt --canonicalize %s | FileCheck %s

// ---- Test 1: is_variant of matching make folds to true ----
// CHECK-LABEL: func @is_variant_make_match
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: return %[[TRUE]]
func.func @is_variant_make_match(%val: i64) -> i1 {
  %x = sum.make 0 %val : !sum.sum<(i64, none)>
  %r = sum.is_variant %x, 0 : !sum.sum<(i64, none)>
  return %r : i1
}

// ---- Test 2: is_variant of non-matching make folds to false ----
// CHECK-LABEL: func @is_variant_make_mismatch
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: return %[[FALSE]]
func.func @is_variant_make_mismatch(%val: i64) -> i1 {
  %x = sum.make 0 %val : !sum.sum<(i64, none)>
  %r = sum.is_variant %x, 1 : !sum.sum<(i64, none)>
  return %r : i1
}

// ---- Test 3: is_variant of nullary make folds to true ----
// CHECK-LABEL: func @is_variant_make_nullary
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: return %[[TRUE]]
func.func @is_variant_make_nullary() -> i1 {
  %x = sum.make 1 : !sum.sum<(i64, none)>
  %r = sum.is_variant %x, 1 : !sum.sum<(i64, none)>
  return %r : i1
}

// ---- Test 4: is_variant of unknown input is not folded ----
// CHECK-LABEL: func @is_variant_unknown
// CHECK: sum.is_variant
func.func @is_variant_unknown(%x: !sum.sum<(i64, none)>) -> i1 {
  %r = sum.is_variant %x, 0 : !sum.sum<(i64, none)>
  return %r : i1
}
