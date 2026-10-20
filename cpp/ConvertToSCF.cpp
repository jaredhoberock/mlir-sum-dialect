#include "ConvertToSCF.hpp"
#include "Sum.hpp"
#include "SumOps.hpp"
#include "SumTypeInterface.hpp"
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/WalkPatternRewriteDriver.h>

namespace mlir::sum {

struct MatchOpLowering : OpRewritePattern<MatchOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(MatchOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto sumTy = cast<SumTypeInterface>(op.getInput().getType());
    size_t numVariants = sumTy.getNumVariants();

    Value tag = TagOp::create(rewriter, loc, op.getInput());
    // Variants 0 through n - 2 are the switch's cases and the last its default,
    // so every tag the sum holds has its arm.
    SmallVector<int64_t> cases;
    for (size_t variant = 0; variant + 1 < numVariants; ++variant)
      cases.push_back(variant);
    auto switchOp = scf::IndexSwitchOp::create(
        rewriter, loc, op.getResultTypes(), tag, cases, cases.size());
    for (size_t i = 0; i < numVariants; ++i) {
      Region &srcRegion = op.getCases()[i];
      Region &tgtRegion = i + 1 < numVariants ? switchOp.getCaseRegions()[i]
                                              : switchOp.getDefaultRegion();

      bool isNullary = sumTy.getPayloadTypes(i).empty();

      Block *entryBlock = rewriter.createBlock(&tgtRegion);
      rewriter.setInsertionPointToStart(entryBlock);

      // Inline source region
      rewriter.inlineRegionBefore(srcRegion, tgtRegion, tgtRegion.end());

      if (isNullary) {
        // Nullary variant: no payload, no block args to replace
        Block *secondBlock = &*std::next(tgtRegion.begin());
        rewriter.mergeBlocks(secondBlock, entryBlock, {});
      } else {
        // Extract payload with sum.get
        Value payload = GetOp::create(rewriter, loc, op.getInput(), i);

        // Merge, replacing block arg with payload
        Block *secondBlock = &*std::next(tgtRegion.begin());
        rewriter.mergeBlocks(secondBlock, entryBlock, {payload});
      }

      // Replace sum.yield with scf.yield
      auto yield = cast<YieldOp>(tgtRegion.back().getTerminator());
      rewriter.setInsertionPoint(yield);
      rewriter.replaceOpWithNewOp<scf::YieldOp>(yield, yield.getResults());
    }

    rewriter.replaceOp(op, switchOp.getResults());
    return success();
  }
};

void populateSumToSCFConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<MatchOpLowering>(patterns.getContext());
}

void ConvertSumToSCFPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateSumToSCFConversionPatterns(patterns);
  // A walk driver applies the match lowering once per operation and neither folds
  // nor erases trivially dead ops, so running this step before instantiation
  // changes nothing but the matches it lowers.
  walkAndApplyPatterns(getOperation(),
                       FrozenRewritePatternSet(std::move(patterns)));
}

} // end mlir::sum
