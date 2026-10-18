// RUN: mlir-opt --allow-unregistered-dialect --canonicalize %s | FileCheck %s

// A match of a sum that an scf.if or scf.index_switch selects among makes --
// or the scf.index_switch on its sum.tag that a match lowers to -- moves into
// the branch: each arm runs the match's case of the variant it makes on its own
// make's payload and yields that case's results beside its own, so the sum is
// built only for the readers that remain. A case no arm makes the variant of is
// dropped, and a reader arm two arms would run keeps the reader where it is.
// The reader runs at the branch's place, so an operation between them that is
// not pure -- one with memory effects, or a loop that may not end -- keeps it
// where it is.

!O = !sum.sum<(i64, none)>
!T = !sum.sum<(i64, none, f32)>

// The index of a checked read: Some of a value the arm computes, None
// otherwise; the None case of its unwrap traps. The if yields the read and the
// trap moves into its else arm, after which the if, one of whose arms aborts,
// is the assertion of its condition followed by its live arm (the dialect's
// branch rewrites, BranchArms.hpp).
// CHECK-LABEL: func.func @match_of_some_or_none
// CHECK-SAME: (%[[C:.*]]: i1, %[[X:.*]]: i64)
// CHECK-NOT: scf.if
// CHECK: cf.assert %[[C]], "unwrap on None"
// CHECK-NEXT: %[[P:.*]] = arith.muli %[[X]], %[[X]]
// CHECK-NOT: sum.
// CHECK: return %[[P]]
func.func @match_of_some_or_none(%c: i1, %x: i64) -> i64 {
  %s = scf.if %c -> !O {
    %p = arith.muli %x, %x : i64
    %m = sum.make 0 %p : !O
    scf.yield %m : !O
  } else {
    %m = sum.make 1 : !O
    scf.yield %m : !O
  }
  %r = sum.match %s : !O -> i64
  case 0 (%v: i64) {
    sum.yield %v : i64
  }
  case 1 {
    %false = arith.constant false
    cf.assert %false, "unwrap on None"
    %u = ub.poison : i64
    sum.yield %u : i64
  }
  return %r : i64
}

// The shape a match lowers to before the canonicalizer meets it.
// CHECK-LABEL: func.func @switch_on_the_tag_of_some_or_none
// CHECK-SAME: (%[[C:.*]]: i1, %[[X:.*]]: i64)
// CHECK-NOT: scf.if
// CHECK: cf.assert %[[C]], "unwrap on None"
// CHECK-NEXT: %[[P:.*]] = arith.muli %[[X]], %[[X]]
// CHECK-NOT: sum.
// CHECK: return %[[P]]
func.func @switch_on_the_tag_of_some_or_none(%c: i1, %x: i64) -> i64 {
  %s = scf.if %c -> !O {
    %p = arith.muli %x, %x : i64
    %m = sum.make 0 %p : !O
    scf.yield %m : !O
  } else {
    %m = sum.make 1 : !O
    scf.yield %m : !O
  }
  %t = sum.tag %s : !O
  %r = scf.index_switch %t -> i64
  case 0 {
    %v = sum.get %s, 0 : !O -> i64
    scf.yield %v : i64
  }
  default {
    %false = arith.constant false
    cf.assert %false, "unwrap on None"
    %u = ub.poison : i64
    scf.yield %u : i64
  }
  return %r : i64
}

// A switch rebuilding a three-variant sum from its tag, each payload computed in
// its arm, under a match of all three: each arm runs its own case.
// CHECK-LABEL: func.func @match_of_a_three_way_switch
// CHECK-SAME: (%[[TAG:.*]]: index, %[[X:.*]]: i64, %[[Y:.*]]: f32)
// CHECK: %[[R:.*]] = scf.index_switch %[[TAG]] -> i64
// CHECK: case 0 {
// CHECK:   %[[A:.*]] = arith.addi %[[X]], %[[X]]
// CHECK:   scf.yield %[[A]]
// CHECK: case 1 {
// CHECK:   scf.yield %c7_i64
// CHECK: default {
// CHECK:   %[[F:.*]] = arith.addf %[[Y]], %[[Y]]
// CHECK:   %[[B:.*]] = arith.fptosi %[[F]]
// CHECK:   scf.yield %[[B]]
// CHECK-NOT: sum.
// CHECK: return %[[R]]
func.func @match_of_a_three_way_switch(%tag: index, %x: i64, %y: f32) -> i64 {
  %s = scf.index_switch %tag -> !T
  case 0 {
    %p = arith.addi %x, %x : i64
    %m = sum.make 0 %p : !T
    scf.yield %m : !T
  }
  case 1 {
    %m = sum.make 1 : !T
    scf.yield %m : !T
  }
  default {
    %f = arith.addf %y, %y : f32
    %m = sum.make 2 %f : !T
    scf.yield %m : !T
  }
  %r = sum.match %s : !T -> i64
  case 0 (%v: i64) {
    sum.yield %v : i64
  }
  case 1 {
    %seven = arith.constant 7 : i64
    sum.yield %seven : i64
  }
  case 2 (%w: f32) {
    %i = arith.fptosi %w : f32 to i64
    sum.yield %i : i64
  }
  return %r : i64
}

// An if making two of three variants: the case of the third is dropped.
// CHECK-LABEL: func.func @match_with_a_case_no_arm_reaches
// CHECK: scf.if
// CHECK-NOT: "test.unreached"
// CHECK-NOT: sum.
// CHECK: return
func.func @match_with_a_case_no_arm_reaches(%c: i1, %x: i64, %y: f32) -> i64 {
  %s = scf.if %c -> !T {
    %m = sum.make 0 %x : !T
    scf.yield %m : !T
  } else {
    %m = sum.make 2 %y : !T
    scf.yield %m : !T
  }
  %r = sum.match %s : !T -> i64
  case 0 (%v: i64) {
    sum.yield %v : i64
  }
  case 1 {
    %u = "test.unreached"() : () -> i64
    sum.yield %u : i64
  }
  case 2 (%w: f32) {
    %i = arith.fptosi %w : f32 to i64
    sum.yield %i : i64
  }
  return %r : i64
}

// A sum with a second reader: the match moves, and the if still yields the sum
// for the other reader beside the match's result.
// CHECK-LABEL: func.func @match_of_a_sum_read_again
// CHECK-SAME: (%[[C:.*]]: i1, %[[X:.*]]: i64)
// CHECK: %[[R:.*]]:2 = scf.if %[[C]] -> (!sum.sum<(i64, none)>, i64) {
// CHECK:   %[[P:.*]] = arith.muli %[[X]], %[[X]]
// CHECK:   %[[M:.*]] = sum.make 0 %[[P]]
// CHECK:   scf.yield %[[M]], %[[P]]
// CHECK: } else {
// CHECK:   %[[N:.*]] = sum.make 1
// CHECK:   scf.yield %[[N]], %c0_i64
// CHECK: }
// CHECK-NOT: sum.match
// CHECK: return %[[R]]#1, %[[R]]#0
func.func @match_of_a_sum_read_again(%c: i1, %x: i64) -> (i64, !O) {
  %s = scf.if %c -> !O {
    %p = arith.muli %x, %x : i64
    %m = sum.make 0 %p : !O
    scf.yield %m : !O
  } else {
    %m = sum.make 1 : !O
    scf.yield %m : !O
  }
  %r = sum.match %s : !O -> i64
  case 0 (%v: i64) {
    sum.yield %v : i64
  }
  case 1 {
    %zero = arith.constant 0 : i64
    sum.yield %zero : i64
  }
  return %r, %s : i64, !O
}

// A store between the if and the match: the match stays where it is.
// CHECK-LABEL: func.func @match_after_a_store
// CHECK: scf.if
// CHECK: memref.store
// CHECK: sum.match
// CHECK: return
func.func @match_after_a_store(%c: i1, %x: i64, %buffer: memref<i64>) -> i64 {
  %s = scf.if %c -> !O {
    %p = arith.muli %x, %x : i64
    %m = sum.make 0 %p : !O
    scf.yield %m : !O
  } else {
    %m = sum.make 1 : !O
    scf.yield %m : !O
  }
  memref.store %x, %buffer[] : memref<i64>
  %r = sum.match %s : !O -> i64
  case 0 (%v: i64) {
    sum.yield %v : i64
  }
  case 1 {
    %zero = arith.constant 0 : i64
    sum.yield %zero : i64
  }
  return %r : i64
}

// A switch on the tag whose default stands for variants 1 and 2, each made by
// its own arm: sinking would copy the default into both arms, so the reader
// stays where it is and its default runs once.
// CHECK-LABEL: func.func @switch_whose_default_two_arms_reach
// CHECK: "test.big"
// CHECK-NOT: "test.big"
// CHECK: return
func.func @switch_whose_default_two_arms_reach(%tag: index, %x: i64, %y: f32) -> i64 {
  %s = scf.index_switch %tag -> !T
  case 0 {
    %m = sum.make 0 %x : !T
    scf.yield %m : !T
  }
  case 1 {
    %m = sum.make 1 : !T
    scf.yield %m : !T
  }
  default {
    %m = sum.make 2 %y : !T
    scf.yield %m : !T
  }
  %t = sum.tag %s : !T
  %r = scf.index_switch %t -> i64
  case 0 {
    %v = sum.get %s, 0 : !T -> i64
    scf.yield %v : i64
  }
  default {
    %u = "test.big"() : () -> i64
    scf.yield %u : i64
  }
  return %r : i64
}

// A loop between the if and the match is free of memory effects but may never
// end, so it is not pure: running the match's call ahead of it could make a
// call the program never makes. The match stays after the loop.
// CHECK-LABEL: func.func @match_after_a_loop
// CHECK: scf.if
// CHECK: scf.while
// CHECK: sum.match
// CHECK: func.call @use
// CHECK: return
func.func @match_after_a_loop(%c: i1, %again: i1, %x: i64) -> i64 {
  %one = arith.constant 1 : i64
  %s = scf.if %c -> !O {
    %p = arith.addi %x, %one : i64
    %m = sum.make 0 %p : !O
    scf.yield %m : !O
  } else {
    %m = sum.make 1 : !O
    scf.yield %m : !O
  }
  %loop = scf.while (%v = %x) : (i64) -> i64 {
    scf.condition(%again) %v : i64
  } do {
  ^bb0(%v: i64):
    %next = arith.addi %v, %one : i64
    scf.yield %next : i64
  }
  sum.match %s : !O
  case 0 (%v: i64) {
    func.call @use(%v) : (i64) -> ()
    sum.yield
  }
  case 1 {
    func.call @use(%x) : (i64) -> ()
    sum.yield
  }
  return %loop : i64
}

// The same loop before the switch on the tag that a match lowers to: the
// switch stays after the loop, its tag the if's choice between the variants
// its arms make, and its case reads the payload of the sum the if builds,
// which no rewrite moves out of the arm that builds it.
// CHECK-LABEL: func.func @switch_on_the_tag_after_a_loop
// CHECK-SAME: (%[[C:.*]]: i1, %{{.*}}: i1, %{{.*}}: i64)
// CHECK: %[[S:.*]] = scf.if %[[C]]
// CHECK: arith.addi
// CHECK: scf.while
// CHECK: %[[TAG:.*]] = arith.select %[[C]]
// CHECK: scf.index_switch %[[TAG]]
// CHECK-NEXT: case 0 {
// CHECK-NEXT: %[[P:.*]] = sum.get %[[S]], 0
// CHECK-NEXT: func.call @use(%[[P]])
// CHECK: return
func.func @switch_on_the_tag_after_a_loop(%c: i1, %again: i1, %x: i64) -> i64 {
  %one = arith.constant 1 : i64
  %s = scf.if %c -> !O {
    %p = arith.addi %x, %one : i64
    %m = sum.make 0 %p : !O
    scf.yield %m : !O
  } else {
    %m = sum.make 1 : !O
    scf.yield %m : !O
  }
  %loop = scf.while (%v = %x) : (i64) -> i64 {
    scf.condition(%again) %v : i64
  } do {
  ^bb0(%v: i64):
    %next = arith.addi %v, %one : i64
    scf.yield %next : i64
  }
  %t = sum.tag %s : !O
  scf.index_switch %t
  case 0 {
    %v = sum.get %s, 0 : !O -> i64
    func.call @use(%v) : (i64) -> ()
    scf.yield
  }
  default {
    func.call @use(%x) : (i64) -> ()
    scf.yield
  }
  return %loop : i64
}

func.func private @use(i64)
