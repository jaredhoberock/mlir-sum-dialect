// A branch with an arm that leaves its readers nothing, written as what it is.
//
// An arm that asserts a constant false cannot complete: the program ends at the
// assertion, and the values the arm yields after it are never read. A branch one
// of whose arms aborts so is therefore the assertion of the condition under
// which it takes its other arm, followed by that arm: on every input the two
// programs abort at the same assertion with the same message, and otherwise run
// the same operations and compute the same values.
//
// An arm that yields only poison and does nothing else gives its readers no
// value: poison is refined by any value, so the branch's results may be the
// other arm's wherever the branch takes either arm. When that other arm is pure
// it may run where the branch would not take it, and the branch is that arm.
//
// Written so, the branch joins nothing, and a value its live arm computes
// reaches the readers of the branch's result as itself, its producer and its
// type in view, where as the branch's result it is the join of that value with
// the other arm's, which no fold sees through.
#include "BranchArms.hpp"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace mlir::sum {

namespace {

/// The assertion that makes the arm `region` abort: a `cf.assert` of a constant
/// false, every operation before it in the arm free of memory effects. Null when
/// the arm has none, and for an absent arm. An arm that writes memory, or
/// asserts anything else, before its false assertion does not abort here: what
/// it does before the program ends is observable, so the branch is left as it
/// stands.
cf::AssertOp abortingAssertion(Region &region) {
  if (region.empty())
    return {};
  for (Operation &op : region.front()) {
    if (auto assertion = dyn_cast<cf::AssertOp>(op))
      if (matchPattern(assertion.getArg(), m_Zero()))
        return assertion;
    if (!isMemoryEffectFree(&op))
      return {};
  }
  return {};
}

/// Whether every operation of the arm `region` before its terminator is pure:
/// free of memory effects and speculatable, so it terminates and has no
/// undefined behavior. Running such an arm where the branch would not take it,
/// or not running it where the branch would, changes nothing observable. False
/// for an absent arm.
bool isPureArm(Region &region) {
  if (region.empty())
    return false;
  return llvm::all_of(region.front().without_terminator(),
                      [](Operation &op) { return isPure(&op); });
}

/// Whether the arm `region` is pure (isPureArm) and every value it yields is a
/// `ub.poison`: taken, it does nothing and gives its readers no value.
bool yieldsOnlyPoison(Region &region) {
  if (!isPureArm(region))
    return false;
  return llvm::all_of(region.front().getTerminator()->getOperands(),
                      [](Value yielded) {
                        return yielded.getDefiningOp<ub::PoisonOp>();
                      });
}

/// Replaces `op` with the arm `region`: the arm's operations inlined before
/// `op`, and the values the arm yields in place of `op`'s results.
void replaceWithArm(PatternRewriter &rewriter, Operation *op, Region &region) {
  Block *arm = &region.front();
  Operation *yield = arm->getTerminator();
  ValueRange yielded = yield->getOperands();
  rewriter.inlineBlockBefore(arm, op);
  rewriter.replaceOp(op, yielded);
  rewriter.eraseOp(yield);
}

/// An `scf.if` exactly one of whose arms aborts (abortingAssertion) becomes the
/// assertion, with that arm's message and location, of the condition under which
/// the `scf.if` takes its other arm, followed by the other arm. An absent else
/// arm yields nothing and does not abort, so an `scf.if` without one whose then
/// arm aborts becomes the assertion that its condition is false.
struct IfWithAbortingArm : OpRewritePattern<scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp op,
                                PatternRewriter &rewriter) const override {
    cf::AssertOp thenAborts = abortingAssertion(op.getThenRegion());
    cf::AssertOp elseAborts = abortingAssertion(op.getElseRegion());
    if (static_cast<bool>(thenAborts) == static_cast<bool>(elseAborts))
      return rewriter.notifyMatchFailure(op, "not exactly one arm aborts");
    Value taken = op.getCondition();
    cf::AssertOp aborting = elseAborts;
    Region *live = &op.getThenRegion();
    if (thenAborts) {
      Value isTrue = arith::ConstantIntOp::create(rewriter, op.getLoc(), 1, 1);
      taken = arith::XOrIOp::create(rewriter, op.getLoc(), taken, isTrue);
      aborting = thenAborts;
      live = &op.getElseRegion();
    }
    cf::AssertOp::create(rewriter, aborting.getLoc(), taken, aborting.getMsg());
    if (live->empty())
      rewriter.eraseOp(op);
    else
      replaceWithArm(rewriter, op, *live);
    return success();
  }
};

/// An `scf.index_switch` all of whose arms but one abort (abortingAssertion)
/// becomes assertions followed by the live arm: for each aborting case, that the
/// switch's value is not the case's, with the case's message; then, when the
/// default arm aborts, that the value is the live case's, with the default's
/// message. The case values are distinct, so the assertions fail exactly on the
/// values that select an aborting arm, and the first to fail is the one that
/// carries that arm's message.
struct IndexSwitchWithAbortingArms : OpRewritePattern<scf::IndexSwitchOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IndexSwitchOp op,
                                PatternRewriter &rewriter) const override {
    // Each arm with the assertion that makes it abort, the default arm last.
    SmallVector<Region *> arms;
    for (Region &arm : op.getCaseRegions())
      arms.push_back(&arm);
    arms.push_back(&op.getDefaultRegion());
    SmallVector<cf::AssertOp> aborting =
        llvm::map_to_vector(arms, [](Region *arm) {
          return abortingAssertion(*arm);
        });
    if (llvm::count(aborting, cf::AssertOp()) != 1)
      return rewriter.notifyMatchFailure(op, "not all arms but one abort");
    size_t live = llvm::find(aborting, cf::AssertOp()) - aborting.begin();

    Location loc = op.getLoc();
    auto assertSelects = [&](arith::CmpIPredicate predicate, int64_t value,
                             cf::AssertOp arm) {
      Value caseValue = arith::ConstantIndexOp::create(rewriter, loc, value);
      Value holds = arith::CmpIOp::create(rewriter, loc, predicate,
                                          op.getArg(), caseValue);
      cf::AssertOp::create(rewriter, arm.getLoc(), holds, arm.getMsg());
    };
    for (auto [index, value] : llvm::enumerate(op.getCases()))
      if (index != live)
        assertSelects(arith::CmpIPredicate::ne, value, aborting[index]);
    if (live != op.getNumCases())
      assertSelects(arith::CmpIPredicate::eq, op.getCases()[live],
                    aborting.back());
    replaceWithArm(rewriter, op, *arms[live]);
    return success();
  }
};

/// An `scf.if` one of whose arms yields only poison (yieldsOnlyPoison) and whose
/// other arm is pure (isPureArm) becomes that other arm, inlined before it.
/// Where the branch takes the poison arm its results are poison, which the
/// other arm's values refine, and the other arm, pure, may run there.
struct IfWithPoisonArm : OpRewritePattern<scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumResults() == 0)
      return rewriter.notifyMatchFailure(op, "the branch yields nothing");
    Region &then = op.getThenRegion(), &otherwise = op.getElseRegion();
    if (yieldsOnlyPoison(otherwise) && isPureArm(then))
      replaceWithArm(rewriter, op, then);
    else if (yieldsOnlyPoison(then) && isPureArm(otherwise))
      replaceWithArm(rewriter, op, otherwise);
    else
      return rewriter.notifyMatchFailure(
          op, "no arm yields only poison beside a pure arm");
    return success();
  }
};

} // namespace

void populateBranchArmPatterns(RewritePatternSet &patterns) {
  patterns.add<IfWithAbortingArm, IndexSwitchWithAbortingArms, IfWithPoisonArm>(
      patterns.getContext());
}

} // namespace mlir::sum
