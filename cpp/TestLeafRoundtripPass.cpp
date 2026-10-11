#include "TestLeafRoundtripPass.hpp"
#include <LeafConversion.hpp>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>

namespace mlir::sum {

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
      if (failed(lowering::appendLeaves(*converter, builder, ret.getLoc(), value,
                                        leaves)))
        return signalPassFailure();
      Value again = lowering::valueFromLeaves(*converter, builder, ret.getLoc(),
                                              value.getType(), leaves);
      if (!again)
        return signalPassFailure();
      rebuilt.push_back(again);
    }
    ret->setOperands(rebuilt);
  }
}

} // namespace mlir::sum
