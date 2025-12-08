#include "ConvertToLLVM.hpp"
#include "Sum.hpp"
#include "SumOps.hpp"
#include "SumTypes.hpp"
#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include "Sum.cpp.inc"

namespace mlir::sum {

struct ConvertToLLVMInterface : public mlir::ConvertToLLVMPatternInterface {
  using mlir::ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(ConversionTarget& target,
                                               LLVMTypeConverter& typeConverter,
                                               RewritePatternSet& patterns) const override final {
    populateSumToLLVMConversionPatterns(typeConverter, patterns);
  }
};

void SumDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "SumOps.cpp.inc"
  >();

  registerTypes();

  addInterfaces<ConvertToLLVMInterface>();
}

}
