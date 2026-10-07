// RUN: mlir-opt --pass-pipeline="builtin.module(remove-dead-values)" %s | FileCheck %s --check-prefix=DEAD
// RUN: mlir-opt --pass-pipeline="builtin.module(sccp)" %s | FileCheck %s --check-prefix=SCCP

// sum.match states its region control flow as scf.index_switch does, so the
// upstream analyses that read RegionBranchOpInterface treat a match as they
// treat the switch the sum lowering rewrites it into.

// A dead match is erased; a match whose result is dead keeps its effects and
// yields poison in place of the dead values.
// DEAD-LABEL: func.func @dead_match
// DEAD-NOT: sum.match
// DEAD: return
func.func @dead_match(%s: !sum.sum<(i64, none)>, %x: i64) -> i64 {
  %r = sum.match %s : !sum.sum<(i64, none)> -> i64
  case 0 (%n: i64) {
    %m = arith.addi %n, %x : i64
    yield %m : i64
  }
  case 1 {
    yield %x : i64
  }
  return %x : i64
}

// DEAD-LABEL: func.func @dead_result
// DEAD: sum.match
// DEAD: case 0
// DEAD: memref.store
// DEAD: %[[P0:.*]] = ub.poison : i64
// DEAD-NEXT: yield %[[P0]] : i64
// DEAD: case 1
// DEAD: %[[P1:.*]] = ub.poison : i64
// DEAD-NEXT: yield %[[P1]] : i64
func.func @dead_result(%s: !sum.sum<(i64, none)>, %out: memref<i64>) {
  %r = sum.match %s : !sum.sum<(i64, none)> -> i64
  case 0 (%n: i64) {
    memref.store %n, %out[] : memref<i64>
    yield %n : i64
  }
  case 1 {
    %c = arith.constant 7 : i64
    yield %c : i64
  }
  return
}

// Every case yielding the same constant makes the result that constant; a
// case argument comes from the sum value, not from a forwarded operand, so
// nothing is known of it.
// SCCP-LABEL: func.func @same_constant
// SCCP-NEXT: %[[C:.*]] = arith.constant 3 : i64
// SCCP-NEXT: return %[[C]] : i64
func.func @same_constant(%s: !sum.sum<(i64, none)>) -> i64 {
  %c3 = arith.constant 3 : i64
  %r = sum.match %s : !sum.sum<(i64, none)> -> i64
  case 0 (%n: i64) {
    yield %c3 : i64
  }
  case 1 {
    yield %c3 : i64
  }
  return %r : i64
}

// SCCP-LABEL: func.func @case_argument
// SCCP: %[[R:.*]] = sum.match
// SCCP: case 0 (%[[N:.*]]: i64)
// SCCP-NEXT: yield %[[N]] : i64
// SCCP: return %[[R]] : i64
func.func @case_argument(%s: !sum.sum<(i64, none)>) -> i64 {
  %c3 = arith.constant 3 : i64
  %r = sum.match %s : !sum.sum<(i64, none)> -> i64
  case 0 (%n: i64) {
    yield %n : i64
  }
  case 1 {
    yield %c3 : i64
  }
  return %r : i64
}
