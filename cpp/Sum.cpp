#include "ConvertToLLVM.hpp"
#include "ConvertToSCF.hpp"
#include "LoweringContribution.hpp"
#include "Sum.hpp"
#include "SumOps.hpp"
#include "SumTypes.hpp"
#include <mlir/CAPI/Pass.h>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Transforms/InliningUtils.h>

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

/// Declares inlining legal for the sum dialect. MLIR's inliner clones a callee
/// only when every op in it belongs to a dialect registering this interface, so a
/// function holding a sum op is inlinable only once sum provides one. The hooks
/// are permissive: sum's ops carry no state that forbids cloning them into
/// another region.
struct SumInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }
};

namespace {
/// Adds the sum conversion pass to `pm`. It reads no per-invocation options.
void addConvertSumToSCF(MlirOpPassManager pm, void *) {
  mlirOpPassManagerAddOwnedPass(pm, wrap(createConvertSumToSCFPass().release()));
}

/// The sum dialect's lowering step converts its match operation to structured
/// control flow, minting sum.tag and sum.get, so the dialect's op count cannot
/// witness the claim and the exact operation names it.
struct LoweringContribution : lowering::LoweringContributionInterface {
  using lowering::LoweringContributionInterface::LoweringContributionInterface;
  void contributeSteps(lowering::LoweringStepSink &sink) const override {
    sink.beginStep("convert-sum-to-scf", false, "", false);
    sink.passConstructor(&addConvertSumToSCF);
    sink.dischargeOperation("sum.match");
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
  addInterfaces<SumInlinerInterface>();
}

}
