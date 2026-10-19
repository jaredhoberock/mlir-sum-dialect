// A branch with an arm that cannot complete, written as what it is.
//
// XXX TODO: the two rewrites here are an upstream `scf.if` and
// `scf.index_switch` canonicalization the estate carries because upstream has
// none: an arm that asserts false, and the assertion that follows a branch on
// its own condition (CombineIfs' adjacency law, with an arm that aborts). They
// are deleted when that canonicalization lands in the fork, one of the fork
// patches docs/plans/ARRAY_TO_TENSOR_PLAN.md lists.
//
// Every rewrite here keeps one law: the branch becomes an assertion of the
// condition under which it takes its live arm, followed by that arm, and
// nothing else moves. Where the assertion holds, the program runs the same
// operations in the same order and computes the same values. Where it fails,
// the program ends at the assertion with the message it ended with before,
// having skipped only pure operations: free of memory effects, whose running is
// not observed once the program ends, and speculatable, so each terminates and
// skipping it does not turn a run that never reached the assertion into one
// that ends there. The live arm runs after the assertion of its own condition,
// where it ran before, so no operation of it runs on a path its guard
// excluded, and nothing is asked of the arm's operations. Whether an operation
// may be skipped or crossed is asked only of the operations the moved
// assertion goes ahead of on the path where it fails.
//
// An arm cannot complete in two ways. It aborts: it asserts a constant false
// after pure operations. Or the continuation refutes it: the operations after
// the branch, past pure ones, assert the condition under which the branch takes
// another arm, so a run that takes this arm fails that assertion if it gets
// there; when the arm is pure, it does, and nothing the arm or those operations
// did is observed before it fails.
//
// Written so, the branch joins nothing, and a value its live arm computes
// reaches the readers of the branch's result as itself, its producer and its
// type in view, where as the branch's result it is the join of that value with
// the other arm's, which no fold sees through.
#include "BranchArms.hpp"
#include "SumOps.hpp"
#include "SumTypeInterface.hpp"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace mlir::sum {

namespace {

/// The assertion that makes the arm `region` abort: a `cf.assert` of a constant
/// false, every operation before it in the arm pure. Null when the arm has
/// none, and for an absent arm. An arm that writes memory, asserts anything
/// else, or runs an operation that may not terminate before its false
/// assertion does not abort here: what it does before the program ends is
/// observable, and it may never reach the assertion, so the branch is left as
/// it stands.
cf::AssertOp abortingAssertion(Region &region) {
  if (region.empty())
    return {};
  for (Operation &op : region.front()) {
    if (auto assertion = dyn_cast<cf::AssertOp>(op))
      if (matchPattern(assertion.getArg(), m_Zero()))
        return assertion;
    if (!isPure(&op))
      return {};
  }
  return {};
}

/// The first assertion after `branch` in its block, every operation between
/// them pure -- free of memory effects and speculatable, so each terminates and
/// skipping it where the assertion fails is not observed. Null when an
/// operation that is not pure, or the end of the block, comes first.
cf::AssertOp continuationAssertion(Operation *branch) {
  for (Operation *op = branch->getNextNode(); op; op = op->getNextNode()) {
    if (auto assertion = dyn_cast<cf::AssertOp>(op))
      return assertion;
    if (!isPure(op))
      return {};
  }
  return {};
}

/// Whether the arm `region` may be skipped where a run that takes it ends at a
/// later assertion: every operation of it pure, so it changes nothing observed
/// and it ends, and the run reaches that assertion. True for an absent arm.
bool skippable(Region &region) {
  if (region.empty())
    return true;
  return llvm::all_of(region.front(),
                      [](Operation &op) { return isPure(&op); });
}

/// Whether `value` is `condition` negated: `arith.xori %condition, true`.
bool isNegationOf(Value value, Value condition) {
  auto negation = value.getDefiningOp<arith::XOrIOp>();
  return negation && negation.getLhs() == condition &&
         matchPattern(negation.getRhs(), m_One());
}

/// The value k such that `condition` holds exactly where the switch's value is
/// k: a constant k compared for equality with the switch's value
/// (`arith.cmpi eq`), or, for a switch on a sum's `sum.tag`, `sum.is_variant`
/// of that sum and k, the form the comparison takes as a question about the
/// sum (TagOp's CompareOfTag). None for any other condition.
std::optional<int64_t> selectedValue(Value condition,
                                     scf::IndexSwitchOp select) {
  Value arg = select.getArg();
  if (auto compare = condition.getDefiningOp<arith::CmpIOp>()) {
    if (compare.getPredicate() != arith::CmpIPredicate::eq)
      return std::nullopt;
    Value other = compare.getLhs() == arg ? compare.getRhs()
                  : compare.getRhs() == arg ? compare.getLhs()
                                            : Value();
    APInt value;
    if (!other || !matchPattern(other, m_ConstantInt(&value)))
      return std::nullopt;
    return value.getSExtValue();
  }
  auto holds = condition.getDefiningOp<IsVariantOp>();
  auto tag = arg.getDefiningOp<TagOp>();
  if (!holds || !tag || holds.getInput() != tag.getInput())
    return std::nullopt;
  return static_cast<int64_t>(holds.getIndex().getZExtValue());
}

/// The position among `select`'s arms (its cases, then its default) of the arm
/// it takes exactly where its value is `value`: the case of `value`; or, for a
/// switch on a sum's `sum.tag` whose cases name every variant of the sum but
/// `value`, the default, which takes the tags no case names, and of a sum's
/// tag, which names one of its variants, that is `value` alone. None
/// otherwise: the default of any other switch also takes values other than
/// `value`.
std::optional<size_t> armTakenAt(int64_t value, scf::IndexSwitchOp select) {
  ArrayRef<int64_t> cases = select.getCases();
  const int64_t *found = llvm::find(cases, value);
  if (found != cases.end())
    return found - cases.begin();
  auto tag = select.getArg().getDefiningOp<TagOp>();
  if (!tag)
    return std::nullopt;
  // The case values are distinct (the switch's verifier), so n - 1 of them
  // within the n variants, none `value`, name every variant but `value`.
  int64_t variants =
      cast<SumTypeInterface>(tag.getInput().getType()).getNumVariants();
  auto isVariant = [&](int64_t variant) {
    return variant >= 0 && variant < variants;
  };
  if (!isVariant(value) || static_cast<int64_t>(cases.size()) + 1 != variants ||
      !llvm::all_of(cases, isVariant))
    return std::nullopt;
  return cases.size();
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

/// An `scf.if` exactly one of whose arms cannot complete becomes the assertion
/// of the condition under which it takes its other arm, followed by the other
/// arm. The arm aborts (abortingAssertion), and the assertion carries that
/// arm's message and location; or, with no arm aborting, the continuation
/// asserts the condition or its negation (continuationAssertion) and the arm it
/// refutes is skippable, and that assertion moves ahead of the live arm with
/// its message and location. An absent else arm yields nothing and does not
/// abort, so an `scf.if` without one whose then arm cannot complete becomes the
/// assertion that its condition is false.
struct IfWithAbortingArm : OpRewritePattern<scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp op,
                                PatternRewriter &rewriter) const override {
    cf::AssertOp thenAborts = abortingAssertion(op.getThenRegion());
    cf::AssertOp elseAborts = abortingAssertion(op.getElseRegion());
    // The continuation's assertion that refutes an arm, moved by the rewrite.
    cf::AssertOp refuting;
    if (!thenAborts && !elseAborts) {
      refuting = continuationAssertion(op);
      if (refuting && refuting.getArg() == op.getCondition() &&
          skippable(op.getElseRegion()))
        elseAborts = refuting;
      else if (refuting && isNegationOf(refuting.getArg(), op.getCondition()) &&
               skippable(op.getThenRegion()))
        thenAborts = refuting;
      else
        return rewriter.notifyMatchFailure(op, "no arm cannot complete");
    }
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
    if (refuting)
      rewriter.eraseOp(refuting);
    if (live->empty())
      rewriter.eraseOp(op);
    else
      replaceWithArm(rewriter, op, *live);
    return success();
  }
};

/// An `scf.index_switch` all of whose arms but one cannot complete becomes
/// assertions followed by the live arm. When every other arm aborts
/// (abortingAssertion): for each aborting case, the assertion that the switch's
/// value is not the case's, with the case's message; then, when the default
/// arm aborts, that the value is the live case's, with the default's message.
/// The case values are distinct, so the assertions fail exactly on the values
/// that select an aborting arm, and the first to fail is the one that carries
/// that arm's message. When no arm aborts, the continuation's assertion
/// (continuationAssertion) that the switch's value is a value (selectedValue)
/// whose arm the switch takes exactly there (armTakenAt) refutes every other
/// arm, each of which must be skippable, and moves ahead of that arm with its
/// message and location.
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
    Location loc = op.getLoc();
    auto assertSelects = [&](arith::CmpIPredicate predicate, int64_t value,
                             Location at, StringRef message) {
      Value caseValue = arith::ConstantIndexOp::create(rewriter, loc, value);
      Value holds = arith::CmpIOp::create(rewriter, loc, predicate,
                                          op.getArg(), caseValue);
      cf::AssertOp::create(rewriter, at, holds, message);
    };

    size_t abortingArms = arms.size() - llvm::count(aborting, cf::AssertOp());
    if (abortingArms == arms.size() - 1) {
      size_t live = llvm::find(aborting, cf::AssertOp()) - aborting.begin();
      for (auto [index, value] : llvm::enumerate(op.getCases()))
        if (index != live)
          assertSelects(arith::CmpIPredicate::ne, value,
                        aborting[index].getLoc(), aborting[index].getMsg());
      if (live != op.getNumCases())
        assertSelects(arith::CmpIPredicate::eq, op.getCases()[live],
                      aborting.back().getLoc(), aborting.back().getMsg());
      replaceWithArm(rewriter, op, *arms[live]);
      return success();
    }
    if (abortingArms != 0)
      return rewriter.notifyMatchFailure(op, "not all arms but one abort");

    // The live arm is the one the switch takes exactly where the
    // continuation's assertion holds.
    cf::AssertOp refuting = continuationAssertion(op);
    std::optional<int64_t> selected =
        refuting ? selectedValue(refuting.getArg(), op) : std::nullopt;
    std::optional<size_t> taken =
        selected ? armTakenAt(*selected, op) : std::nullopt;
    if (!taken)
      return rewriter.notifyMatchFailure(
          op, "the continuation asserts no value one arm alone takes");
    size_t live = *taken;
    for (auto [index, arm] : llvm::enumerate(arms))
      if (index != live && !skippable(*arm))
        return rewriter.notifyMatchFailure(op, "a refuted arm is not pure");
    assertSelects(arith::CmpIPredicate::eq, *selected, refuting.getLoc(),
                  refuting.getMsg());
    rewriter.eraseOp(refuting);
    replaceWithArm(rewriter, op, *arms[live]);
    return success();
  }
};

} // namespace

void populateBranchArmPatterns(RewritePatternSet &patterns) {
  patterns.add<IfWithAbortingArm, IndexSwitchWithAbortingArms>(
      patterns.getContext());
}

} // namespace mlir::sum
