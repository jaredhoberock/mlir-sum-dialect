// A comparison of a value one of two constants against a constant.
//
// XXX TODO: this is an upstream `arith.cmpi` canonicalization the estate
// carries because arith's has only the extension patterns; LLVM's InstCombine
// has it (`foldICmpSelectConstant`). It reads a sum's tag through the select
// a branch between two constant tags canonicalizes to, so a checked read of a
// sum built in one function and unwrapped in another folds whichever is
// inlined first. Deleted when the fold lands in the fork, one of the fork
// patches docs/plans/ARRAY_TO_TENSOR_PLAN.md lists.
#include "CompareOfSelect.hpp"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir::sum {

namespace {

/// `arith.cmpi p, (arith.select %c, k1, k2), k` for constants k1, k2 and k is
/// the select of the two truths `p(k1, k), p(k2, k)` by `%c`: `%c` where they
/// are true and false, its negation where they are false and true, and that
/// truth where they agree. The constant stands on the right, where arith's fold
/// of the comparison puts it.
struct CompareOfSelectedConstants : OpRewritePattern<arith::CmpIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpIOp compare,
                                PatternRewriter &rewriter) const override {
    auto select = compare.getLhs().getDefiningOp<arith::SelectOp>();
    APInt bound, whenTrue, whenFalse;
    if (!select || !compare.getType().isInteger(1) ||
        !select.getCondition().getType().isInteger(1) ||
        !matchPattern(compare.getRhs(), m_ConstantInt(&bound)) ||
        !matchPattern(select.getTrueValue(), m_ConstantInt(&whenTrue)) ||
        !matchPattern(select.getFalseValue(), m_ConstantInt(&whenFalse)))
      return rewriter.notifyMatchFailure(
          compare, "not a comparison of a select between constants");
    bool onTrue =
        arith::applyCmpPredicate(compare.getPredicate(), whenTrue, bound);
    bool onFalse =
        arith::applyCmpPredicate(compare.getPredicate(), whenFalse, bound);
    Value condition = select.getCondition();
    if (onTrue == onFalse)
      rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(compare, onTrue, 1);
    else if (onTrue)
      rewriter.replaceOp(compare, condition);
    else
      rewriter.replaceOpWithNewOp<arith::XOrIOp>(
          compare, condition,
          arith::ConstantIntOp::create(rewriter, compare.getLoc(), 1, 1));
    return success();
  }
};

} // namespace

void populateCompareOfSelectPatterns(RewritePatternSet &patterns) {
  patterns.add<CompareOfSelectedConstants>(patterns.getContext());
}

} // namespace mlir::sum
