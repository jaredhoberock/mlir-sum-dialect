#include "ConvertToSCF.hpp"
#include "Sum.hpp"
#include "SumOps.hpp"
#include "SumTypeInterface.hpp"
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::sum {

struct MatchOpLowering : OpRewritePattern<MatchOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(MatchOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto sumTy = cast<SumTypeInterface>(op.getInput().getType());
    size_t numVariants = sumTy.getNumVariants();

    // Get tag
    Value tag = rewriter.create<TagOp>(loc, op.getInput());

    // Build case values: 0, 1, ..., N-2 (last goes in default)
    SmallVector<int64_t> caseValues;
    for (size_t i = 0; i + 1 < numVariants; ++i)
      caseValues.push_back(i);

    auto switchOp = rewriter.create<scf::IndexSwitchOp>(
        loc, op.getResultTypes(), tag, caseValues, caseValues.size());

    for (size_t i = 0; i < numVariants; ++i) {
      Region &tgtRegion = (i < numVariants - 1)
          ? switchOp.getCaseRegions()[i]
          : switchOp.getDefaultRegion();
      Region &srcRegion = op.getCases()[i];

      Block *entryBlock = rewriter.createBlock(&tgtRegion);
      rewriter.setInsertionPointToStart(entryBlock);

      // Extract payload with sum.get
      Value payload = rewriter.create<GetOp>(loc, op.getInput(), i);

      // Inline source region
      rewriter.inlineRegionBefore(srcRegion, tgtRegion, tgtRegion.end());

      // Merge, replacing block arg with payload
      Block *secondBlock = &*std::next(tgtRegion.begin());
      rewriter.mergeBlocks(secondBlock, entryBlock, {payload});

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
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

} // end mlir::sum
