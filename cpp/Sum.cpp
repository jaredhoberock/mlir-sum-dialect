#include "BranchArms.hpp"
#include "ConvertToLLVM.hpp"
#include "ConvertToSCF.hpp"
#include "LoweringContribution.hpp"
#include "LeafConversion.hpp"
#include "PermissiveInlinerInterface.hpp"
#include "Sum.hpp"
#include "SumLeafConversion.hpp"
#include "SumOps.hpp"
#include "SumTypes.hpp"
#include <mlir/CAPI/Pass.h>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/Transforms/DialectConversion.h>
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

struct LeafConversionInterface : public lowering::LeafConversionInterface {
  using lowering::LeafConversionInterface::LeafConversionInterface;

  void populateLeafTypes(TypeConverter &converter) const override final {
    populateSumLeafTypes(converter);
  }
};

namespace {
/// Adds the sum conversion pass to `pm`. It reads no per-invocation options.
void addConvertSumToSCF(MlirOpPassManager pm, void *) {
  mlirOpPassManagerAddOwnedPass(pm, wrap(createConvertSumToSCFPass().release()));
}

/// The legality convert-sum-to-scf hands the driver: sum.match is illegal, every
/// other operation legal, so the step is present exactly while a match stands and
/// has an opinion on every operation (the sum.tag and sum.get it mints are left
/// for the terminal's LLVM lowering). It rewrites no type, so it hands back no
/// converter.
void *sumLegality(MlirOperation, void *targetPtr, void *, void *) {
  auto &target = *static_cast<ConversionTarget *>(targetPtr);
  target.markUnknownOpDynamicallyLegal(
      [](Operation *op) { return !isa<MatchOp>(op); });
  return nullptr;
}

/// The sum dialect's lowering step converts its match operation to structured
/// control flow, minting sum.tag and sum.get.
struct LoweringContribution : lowering::LoweringContributionInterface {
  using lowering::LoweringContributionInterface::LoweringContributionInterface;
  void contributeSteps(lowering::LoweringStepSink &sink) const override {
    sink.beginStep("convert-sum-to-scf", &addConvertSumToSCF, &sumLegality, nullptr);
  }
};
} // namespace

/// The rewrites of a branch one of whose arms cannot complete
/// (populateBranchArmPatterns). A checked read of a sum is such a branch: a
/// match lowers to a switch on the sum's tag, and an unwrap's arm for every
/// other variant aborts, or the assertion after a branch that builds the sum
/// refutes the arm that builds another variant. The rewrites run wherever a
/// canonicalization does, so the value of the live arm reaches the branch's
/// readers as itself whichever order the reads are lowered and inlined in.
void SumDialect::getCanonicalizationPatterns(RewritePatternSet &results) const {
  populateBranchArmPatterns(results);
}

Operation *SumDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}

void SumDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include <SumOps.cpp.inc>
  >();

  registerTypes();

  addInterfaces<ConvertToLLVMInterface>();
  addInterfaces<LeafConversionInterface>();
  addInterfaces<LoweringContribution>();
  addInterfaces<lowering::PermissiveInlinerInterface<SumDialect>>();
}

}
