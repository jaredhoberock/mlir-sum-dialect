#include "ConvertToLLVM.hpp"
#include "LoweringContribution.hpp"
#include "Sum.hpp"
#include "SumOps.hpp"
#include "SumTypes.hpp"
#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include <Sum.cpp.inc>

namespace mlir::sum {

struct ConvertToLLVMInterface : public mlir::ConvertToLLVMPatternInterface {
  using mlir::ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(ConversionTarget& target,
                                               LLVMTypeConverter& typeConverter,
                                               RewritePatternSet& patterns) const override final {
    populateSumToLLVMConversionPatterns(typeConverter, patterns);
  }
};

namespace {
/// The sum dialect's lowering step converts its match operation to structured
/// control flow, minting sum.tag and sum.get, so the dialect's op count cannot
/// witness the claim and the exact operation names it. The step reaches the
/// serialization boundary, the obligation it carries.
struct LoweringContribution : lowering::LoweringContributionInterface {
  using lowering::LoweringContributionInterface::LoweringContributionInterface;
  void contributeSteps(lowering::LoweringStepSink &sink) const override {
    sink.beginStep("convert-sum-to-scf", false, "", false);
    sink.dischargeOperation("sum.match");
    sink.obligation("the module has reached the serialization boundary");
  }
};
} // namespace

void SumDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include <SumOps.cpp.inc>
  >();

  registerTypes();

  addInterfaces<ConvertToLLVMInterface>();
  addInterfaces<LoweringContribution>();
}

}
