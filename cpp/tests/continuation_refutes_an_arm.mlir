// RUN: mlir-opt --canonicalize %s | FileCheck %s

// An scf.if followed, past pure operations, by the assertion of its condition
// is that assertion followed by its then arm: every run that takes the else arm
// ends at the assertion, and the else arm, free of memory effects, does nothing
// observed before it ends. The assertion of the condition's negation takes the
// else arm. The live arm runs after the assertion of its own condition, where
// it ran before, whatever its operations: one that reads memory, a loop, a
// read of a tensor. An scf.index_switch followed by the assertion that its
// value is one case's is that assertion followed by that case. A store between
// the branch and the assertion, or a refuted arm with effects, keeps the
// branch.

!O = !sum.sum<(i64, none)>

// CHECK-LABEL: func.func @live_arm_reads_memory
// CHECK-SAME: (%[[O:.*]]: !sum.sum<(i64, none)>, %[[M:.*]]: memref<i64>)
// CHECK-NOT: scf.if
// CHECK: %[[SOME:.*]] = sum.is_variant %[[O]], 0
// CHECK-NEXT: cf.assert %[[SOME]], "unwrap on None"
// CHECK-NEXT: %[[V:.*]] = memref.load %[[M]]
// CHECK-NEXT: return %[[V]]
func.func @live_arm_reads_memory(%o: !O, %m: memref<i64>) -> i64 {
  %some = sum.is_variant %o, 0 : !O
  %r = scf.if %some -> i64 {
    %v = memref.load %m[] : memref<i64>
    scf.yield %v : i64
  } else {
    %u = ub.poison : i64
    scf.yield %u : i64
  }
  cf.assert %some, "unwrap on None"
  return %r : i64
}

// CHECK-LABEL: func.func @negated_condition_takes_the_else_arm
// CHECK-SAME: (%[[O:.*]]: !sum.sum<(i64, none)>, %[[M:.*]]: memref<i64>)
// CHECK-DAG: %[[TRUE:.*]] = arith.constant true
// CHECK-NOT: scf.if
// CHECK: %[[NONE:.*]] = sum.is_variant %[[O]], 1
// CHECK-NEXT: %[[SOME:.*]] = arith.xori %[[NONE]], %[[TRUE]]
// CHECK-NEXT: cf.assert %[[SOME]], "unwrap on None"
// CHECK-NEXT: %[[V:.*]] = memref.load %[[M]]
// CHECK-NEXT: return %[[V]]
func.func @negated_condition_takes_the_else_arm(%o: !O, %m: memref<i64>) -> i64 {
  %none = sum.is_variant %o, 1 : !O
  %r = scf.if %none -> i64 {
    %u = ub.poison : i64
    scf.yield %u : i64
  } else {
    %v = memref.load %m[] : memref<i64>
    scf.yield %v : i64
  }
  %true = arith.constant true
  %some = arith.xori %none, %true : i1
  cf.assert %some, "unwrap on None"
  return %r : i64
}

// An Option built by a loop only where n is small, unwrapped: the loop runs
// after the assertion that n is small, never where it is not.
// CHECK-LABEL: func.func @loop_in_the_arm
// CHECK-SAME: (%[[N:.*]]: index)
// CHECK-NOT: scf.if
// CHECK: %[[SMALL:.*]] = arith.cmpi ult, %[[N]]
// CHECK-NEXT: cf.assert %[[SMALL]], "unwrap on None"
// CHECK-NEXT: %[[ACC:.*]] = scf.for
// CHECK-NOT: sum.
// CHECK: return %[[ACC]]
func.func @loop_in_the_arm(%n: index) -> i64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %one = arith.constant 1 : i64
  %three = arith.constant 3 : i64
  %small = arith.cmpi ult, %n, %c10 : index
  %o = scf.if %small -> !O {
    %acc = scf.for %i = %c0 to %n step %c1 iter_args(%a = %one) -> i64 {
      %t = arith.muli %a, %three : i64
      scf.yield %t : i64
    }
    %s_tag = arith.constant 0 : index
    %s = sum.make %s_tag (%acc : i64) : !O
    scf.yield %s : !O
  } else {
    %s_tag = arith.constant 1 : index
    %s_absent = ub.poison : i64
    %s = sum.make %s_tag (%s_absent : i64) : !O
    scf.yield %s : !O
  }
  %t = sum.tag %o : !O
  %v = scf.index_switch %t -> i64
  case 0 {
    %p = sum.get %o, 0 : !O -> i64
    scf.yield %p : i64
  }
  default {
    %false = arith.constant false
    cf.assert %false, "unwrap on None"
    %u = ub.poison : i64
    scf.yield %u : i64
  }
  return %v : i64
}

// A bounds-checked read of a tensor: the read stays after its check.
// CHECK-LABEL: func.func @tensor_read_in_the_arm
// CHECK-SAME: (%[[T:.*]]: tensor<?xf32>, %[[I:.*]]: index, %[[N:.*]]: index)
// CHECK-NOT: scf.if
// CHECK: %[[INSIDE:.*]] = arith.cmpi ult, %[[I]], %[[N]]
// CHECK-NEXT: cf.assert %[[INSIDE]], "index out of bounds"
// CHECK-NEXT: %[[V:.*]] = tensor.extract %[[T]][%[[I]]]
// CHECK-NEXT: return %[[V]]
func.func @tensor_read_in_the_arm(%t: tensor<?xf32>, %i: index, %n: index) -> f32 {
  %inside = arith.cmpi ult, %i, %n : index
  %r = scf.if %inside -> f32 {
    %v = tensor.extract %t[%i] : tensor<?xf32>
    scf.yield %v : f32
  } else {
    %p = ub.poison : f32
    scf.yield %p : f32
  }
  cf.assert %inside, "index out of bounds"
  return %r : f32
}

// CHECK-LABEL: func.func @switch_with_an_assertion_of_its_value
// CHECK-SAME: (%[[ARG:.*]]: index, %[[M:.*]]: memref<i64>, %[[X:.*]]: i64)
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
// CHECK-NOT: scf.index_switch
// CHECK: %[[SELECTED:.*]] = arith.cmpi eq, %[[ARG]], %[[ONE]]
// CHECK-NEXT: cf.assert %[[SELECTED]], "not one"
// CHECK-NEXT: %[[V:.*]] = memref.load %[[M]]
// CHECK-NEXT: return %[[V]]
func.func @switch_with_an_assertion_of_its_value(%arg: index, %m: memref<i64>, %x: i64) -> i64 {
  %c1 = arith.constant 1 : index
  %r = scf.index_switch %arg -> i64
  case 0 {
    scf.yield %x : i64
  }
  case 1 {
    %v = memref.load %m[] : memref<i64>
    scf.yield %v : i64
  }
  default {
    %u = ub.poison : i64
    scf.yield %u : i64
  }
  %selected = arith.cmpi eq, %arg, %c1 : index
  cf.assert %selected, "not one"
  return %r : i64
}

// A match of an Option lowered to a switch on its tag, then an unwrap of it:
// the unwrap's assertion that the Option is Some refutes the None arm.
// CHECK-LABEL: func.func @switch_on_a_tag_then_its_unwrap
// CHECK-SAME: (%[[O:.*]]: !sum.sum<(i64, none)>, %[[M:.*]]: memref<i64>)
// CHECK-NOT: scf.index_switch
// CHECK: %[[SOME:.*]] = sum.is_variant %[[O]], 0
// CHECK-NEXT: cf.assert %[[SOME]], "unwrap on None"
// CHECK-NEXT: %[[P:.*]] = sum.get %[[O]], 0
// CHECK-NEXT: memref.store %[[P]], %[[M]]
// CHECK-NEXT: %[[Q:.*]] = sum.get %[[O]], 0
// CHECK-NEXT: return %[[Q]]
func.func @switch_on_a_tag_then_its_unwrap(%o: !O, %m: memref<i64>) -> i64 {
  %t = sum.tag %o : !O
  scf.index_switch %t
  case 0 {
    %p = sum.get %o, 0 : !O -> i64
    memref.store %p, %m[] : memref<i64>
    scf.yield
  }
  default {
    scf.yield
  }
  %some = sum.is_variant %o, 0 : !O
  cf.assert %some, "unwrap on None"
  %q = sum.get %o, 0 : !O -> i64
  return %q : i64
}

// CHECK-LABEL: func.func @store_between_keeps_the_branch
// CHECK: scf.if
// CHECK: memref.store
// CHECK: cf.assert
func.func @store_between_keeps_the_branch(%o: !O, %m: memref<i64>, %x: i64) -> i64 {
  %some = sum.is_variant %o, 0 : !O
  %r = scf.if %some -> i64 {
    %v = memref.load %m[] : memref<i64>
    scf.yield %v : i64
  } else {
    %u = ub.poison : i64
    scf.yield %u : i64
  }
  memref.store %x, %m[] : memref<i64>
  cf.assert %some, "unwrap on None"
  return %r : i64
}

// CHECK-LABEL: func.func @refuted_arm_with_effects_keeps_the_branch
// CHECK: scf.if
// CHECK: } else {
// CHECK-NEXT: memref.store
// CHECK: cf.assert
func.func @refuted_arm_with_effects_keeps_the_branch(%o: !O, %m: memref<i64>, %x: i64) -> i64 {
  %some = sum.is_variant %o, 0 : !O
  %r = scf.if %some -> i64 {
    %v = memref.load %m[] : memref<i64>
    scf.yield %v : i64
  } else {
    memref.store %x, %m[] : memref<i64>
    %u = ub.poison : i64
    scf.yield %u : i64
  }
  cf.assert %some, "unwrap on None"
  return %r : i64
}

// The value 3 selects the default, which also runs for every value no case
// names, so the assertion does not hold exactly where the switch takes it.
// CHECK-LABEL: func.func @switch_with_an_assertion_of_no_case
// CHECK: scf.index_switch
// CHECK: cf.assert
func.func @switch_with_an_assertion_of_no_case(%arg: index, %m: memref<i64>, %x: i64) -> i64 {
  %c3 = arith.constant 3 : index
  %r = scf.index_switch %arg -> i64
  case 0 {
    scf.yield %x : i64
  }
  default {
    %v = memref.load %m[] : memref<i64>
    scf.yield %v : i64
  }
  %selected = arith.cmpi eq, %arg, %c3 : index
  cf.assert %selected, "not three"
  return %r : i64
}
