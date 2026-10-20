#pragma once

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

inline std::unique_ptr<Pass> createConvertSumToSCFPass() {
  return std::make_unique<ConvertSumToSCFPass>();
}

}
}
