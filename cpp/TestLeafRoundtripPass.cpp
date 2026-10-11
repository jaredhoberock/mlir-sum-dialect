#include "TestLeafRoundtripPass.hpp"
#include <LeafConversion.hpp>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>

namespace mlir::sum {

// The leaf conversion asks the LLVM type conversion whether a type is final, and
// building that conversion loads the LLVM dialect, which a pass may not do while
// it runs.
void TestLeafRoundtripPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect>();
}

void TestLeafRoundtripPass::runOnOperation() {
  ModuleOp module = getOperation();
  ConversionTarget target(getContext());
  RewritePatternSet patterns(&getContext());
  std::unique_ptr<TypeConverter> converter = lowering::leafTypeConverter(
      &getContext(), DataLayout(module), target, patterns);
  SmallVector<func::ReturnOp> returns;
  module.walk([&](func::ReturnOp ret) { returns.push_back(ret); });
  for (func::ReturnOp ret : returns) {
    OpBuilder builder(ret);
    SmallVector<Value> rebuilt;
    for (Value value : ret.getOperands()) {
      SmallVector<Value> leaves;
      Value again;
      if (succeeded(lowering::appendLeaves(*converter, builder, ret.getLoc(),
                                           value, leaves)))
        again = lowering::valueFromLeaves(*converter, builder, ret.getLoc(),
                                          value.getType(), leaves);
      if (!again) {
        ret.emitError() << "a returned " << value.getType()
                        << " does not cross as its leaves";
        return signalPassFailure();
      }
      rebuilt.push_back(again);
    }
    ret->setOperands(rebuilt);
  }
}

} // namespace mlir::sum
