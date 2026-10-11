#pragma once

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Pass/Pass.h>

namespace mlir {

class Pass;
class RewritePatternSet;

namespace sum {

struct ConvertSumToSCFPass : public PassWrapper<ConvertSumToSCFPass, OperationPass<>> {
  inline StringRef getArgument() const override { return "convert-sum-to-scf"; }
  inline StringRef getDescription() const override {
    return "Convert sum dialect to scf dialect";
  }

  void runOnOperation() override;
};

void populateSumToSCFConversionPatterns(RewritePatternSet& patterns);

/// An `scf.index_switch` on `tag` yielding `resultTypes`, with one arm per variant
/// of a sum of `numVariants` variants: variants 0 through n - 2 are its cases and
/// the last is its default, so every tag such a sum holds has its arm.
/// `fillArm(variant, arm)` builds the one block of arm `variant`, its terminator
/// included.
scf::IndexSwitchOp
createVariantSwitch(OpBuilder &builder, Location loc, TypeRange resultTypes,
                    Value tag, size_t numVariants,
                    function_ref<void(size_t variant, Region &arm)> fillArm);

inline std::unique_ptr<Pass> createConvertSumToSCFPass() {
  return std::make_unique<ConvertSumToSCFPass>();
}

}
}
