// RUN: mlir-opt --allow-unregistered-dialect --verify-diagnostics --split-input-file --pass-pipeline="builtin.module(test-sum-payload-sources)" %s

// The payload of the variant a make builds is the make's operand, read by
// sum.get and by the match case of that variant.
func.func @make_read(%a: i64) {
  // expected-note @+1 {{payload source}}
  %v = arith.addi %a, %a : i64
  %s = sum.make 0 %v : !sum.sum<(i64, none)>
  %p = sum.get %s, 0 : !sum.sum<(i64, none)> -> i64
  // expected-remark @+1 {{1 payload sources}}
  "test.payload_sources"(%p) : (i64) -> ()
  // expected-note @+1 {{payload source}}
  %w = arith.muli %a, %a : i64
  %t = sum.make 0 %w : !sum.sum<(i64, none)>
  sum.match %t : !sum.sum<(i64, none)>
  case 0 (%q: i64) {
    // expected-remark @+1 {{1 payload sources}}
    "test.payload_sources"(%q) : (i64) -> ()
    yield
  }
  case 1 {
    yield
  }
  return
}

// -----

// A make of another variant carries no payload of the variant read.
func.func @other_variant(%v: i64, %w: i32) {
  %s = sum.make 1 %w : !sum.sum<(i64, i32)>
  %p = sum.get %s, 0 : !sum.sum<(i64, i32)> -> i64
  // expected-remark @+1 {{0 payload sources}}
  "test.payload_sources"(%p) : (i64) -> ()
  return
}

// -----

// A branch of makes gives the payload from each arm that makes the variant;
// an arm that asserts a constant false and yields poison gives nothing.
func.func @branch_of_makes(%c: i1, %a: i64, %b: i64) {
  %s = scf.if %c -> !sum.sum<(i64, none)> {
    // expected-note @+1 {{payload source}}
    %x = arith.addi %a, %b : i64
    %m = sum.make 0 %x : !sum.sum<(i64, none)>
    scf.yield %m : !sum.sum<(i64, none)>
  } else {
    %n = sum.make 1 : !sum.sum<(i64, none)>
    scf.yield %n : !sum.sum<(i64, none)>
  }
  %t = scf.if %c -> !sum.sum<(i64, none)> {
    scf.yield %s : !sum.sum<(i64, none)>
  } else {
    %false = arith.constant false
    cf.assert %false, "unreachable"
    %m = ub.poison : !sum.sum<(i64, none)>
    scf.yield %m : !sum.sum<(i64, none)>
  }
  %u = scf.if %c -> !sum.sum<(i64, none)> {
    scf.yield %t : !sum.sum<(i64, none)>
  } else {
    %poison = ub.poison : !sum.sum<(i64, none)>
    scf.yield %poison : !sum.sum<(i64, none)>
  }
  sum.match %u : !sum.sum<(i64, none)>
  case 0 (%q: i64) {
    // expected-remark @+1 {{1 payload sources}}
    "test.payload_sources"(%q) : (i64) -> ()
    yield
  }
  case 1 {
    yield
  }
  return
}

// -----

// Two arms making the variant give two payloads.
func.func @two_makes(%c: i1, %a: i64, %b: i64) {
  %s = scf.if %c -> !sum.sum<(i64, none)> {
    // expected-note @+1 {{payload source}}
    %x = arith.addi %a, %b : i64
    %m = sum.make 0 %x : !sum.sum<(i64, none)>
    scf.yield %m : !sum.sum<(i64, none)>
  } else {
    // expected-note @+1 {{payload source}}
    %y = arith.muli %a, %b : i64
    %m = sum.make 0 %y : !sum.sum<(i64, none)>
    scf.yield %m : !sum.sum<(i64, none)>
  }
  %p = sum.get %s, 0 : !sum.sum<(i64, none)> -> i64
  // expected-remark @+1 {{2 payload sources}}
  "test.payload_sources"(%p) : (i64) -> ()
  return
}

// -----

// The spelling the sum lowering leaves: a tag switched on, and sum.get inside
// the case; the sum's producer here is itself an index switch of makes.
func.func @lowered_spelling(%i: index, %a: i64) {
  %s = scf.index_switch %i -> !sum.sum<(i64, none)>
  case 0 {
    // expected-note @+1 {{payload source}}
    %x = arith.addi %a, %a : i64
    %m = sum.make 0 %x : !sum.sum<(i64, none)>
    scf.yield %m : !sum.sum<(i64, none)>
  }
  default {
    %n = sum.make 1 : !sum.sum<(i64, none)>
    scf.yield %n : !sum.sum<(i64, none)>
  }
  %tag = sum.tag %s : !sum.sum<(i64, none)>
  scf.index_switch %tag
  case 0 {
    %p = sum.get %s, 0 : !sum.sum<(i64, none)> -> i64
    // expected-remark @+1 {{1 payload sources}}
    "test.payload_sources"(%p) : (i64) -> ()
    scf.yield
  }
  default {
    scf.yield
  }
  return
}

// -----

// A match whose cases yield sums is read through its declared region control
// flow, and a sum carried as another sum's payload is read through that payload.
func.func @match_of_sums(%o: !sum.sum<(i64, i32)>, %a: i64) {
  %s = sum.match %o : !sum.sum<(i64, i32)> -> !sum.sum<(i64, none)>
  // expected-note @+1 {{payload source}}
  case 0 (%x: i64) {
    %m = sum.make 0 %x : !sum.sum<(i64, none)>
    yield %m : !sum.sum<(i64, none)>
  }
  case 1 (%y: i32) {
    %n = sum.make 1 : !sum.sum<(i64, none)>
    yield %n : !sum.sum<(i64, none)>
  }
  %p = sum.get %s, 0 : !sum.sum<(i64, none)> -> i64
  // expected-remark @+1 {{1 payload sources}}
  "test.payload_sources"(%p) : (i64) -> ()
  // expected-note @+1 {{payload source}}
  %b = arith.addi %a, %a : i64
  %inner = sum.make 0 %b : !sum.sum<(i64, none)>
  %outer = sum.make 0 %inner : !sum.sum<(!sum.sum<(i64, none)>, none)>
  %q = sum.get %outer, 0 : !sum.sum<(!sum.sum<(i64, none)>, none)> -> !sum.sum<(i64, none)>
  %r = sum.get %q, 0 : !sum.sum<(i64, none)> -> i64
  // expected-remark @+1 {{1 payload sources}}
  "test.payload_sources"(%r) : (i64) -> ()
  return
}

// -----

// A sum no rule reads, and a value that reads no payload, have no answer.
func.func @no_answer(%s: !sum.sum<(i64, none)>, %a: i64) {
  %p = sum.get %s, 0 : !sum.sum<(i64, none)> -> i64
  // expected-remark @+1 {{no payload sources}}
  "test.payload_sources"(%p) : (i64) -> ()
  // expected-remark @+1 {{no payload sources}}
  "test.payload_sources"(%a) : (i64) -> ()
  return
}
